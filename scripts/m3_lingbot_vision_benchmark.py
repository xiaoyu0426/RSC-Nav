from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LingBot-Vision variants on Habitat semantic boundaries.")
    parser.add_argument("--frames-metadata", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["small", "base", "large", "giant"])
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from lingbot_vision import extract_patch_tokens, load_image, load_pretrained_backbone

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(Path(args.frames_metadata), args.max_frames)
    weights_root = Path(args.weights_root).expanduser().resolve()
    results: list[dict] = []

    for variant in args.variants:
        weight_dir = weights_root / f"vision_{variant}"
        if not (weight_dir / "model.pt").is_file():
            raise FileNotFoundError(weight_dir / "model.pt")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model, embed_dim = load_pretrained_backbone(
            repo_id_or_path=str(weight_dir),
            variant=variant,
            device=args.device,
            dtype="bf16",
            local_files_only=True,
        )
        load_seconds = time.perf_counter() - load_started
        param_count = sum(p.numel() for p in model.parameters())
        warm_image, _, _ = load_image(
            str(frames[0]["rgb_path"]),
            size=args.image_size,
            patch_size=model.patch_size,
            mode="square",
        )
        extract_patch_tokens(model, warm_image, args.device, torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        inference_seconds = 0.0
        example = None
        for frame_index, frame in enumerate(frames):
            image_t, image_rgb, _ = load_image(
                str(frame["rgb_path"]),
                size=args.image_size,
                patch_size=model.patch_size,
                mode="square",
            )
            started = time.perf_counter()
            tokens, grid = extract_patch_tokens(model, image_t, args.device, torch.bfloat16)
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started

            token_grid = tokens[0].float().cpu().numpy().reshape(grid[0], grid[1], -1)
            token_grid /= np.linalg.norm(token_grid, axis=-1, keepdims=True).clip(1e-8)
            semantic = np.load(frame["semantic_npy"])
            semantic_grid = np.asarray(
                Image.fromarray(semantic.astype(np.int32), mode="I").resize(
                    (grid[1], grid[0]), Image.Resampling.NEAREST
                )
            )
            scores, labels, edge_grid = _neighbor_edges(token_grid, semantic_grid)
            all_scores.append(scores)
            all_labels.append(labels)
            if example is None:
                example = _render_example(image_rgb, token_grid, semantic_grid, edge_grid)

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels).astype(bool)
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        best_index = int(np.nanargmax(f1))
        result = {
            "variant": variant,
            "parameter_count": int(param_count),
            "parameter_millions": round(param_count / 1e6, 2),
            "embed_dim": int(embed_dim),
            "num_frames": len(frames),
            "boundary_positive_rate": float(labels.mean()),
            "boundary_average_precision": float(average_precision_score(labels, scores)),
            "boundary_roc_auc": float(roc_auc_score(labels, scores)),
            "boundary_best_f1": float(f1[best_index]),
            "boundary_best_threshold": float(thresholds[min(best_index, len(thresholds) - 1)]),
            "edge_score_positive_mean": float(scores[labels].mean()),
            "edge_score_negative_mean": float(scores[~labels].mean()),
            "edge_contrast": float(scores[labels].mean() - scores[~labels].mean()),
            "load_seconds": float(load_seconds),
            "inference_seconds": float(inference_seconds),
            "fps": float(len(frames) / max(inference_seconds, 1e-8)),
            "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1e9),
        }
        example_path = out_dir / f"vision_{variant}_boundary_example.png"
        example.save(example_path)
        result["example"] = example_path.name
        results.append(result)
        del model
        torch.cuda.empty_cache()

    payload = {
        "phase": "m3_lingbot_foundation_benchmark",
        "benchmark": "lingbot_vision_parameter_scaling",
        "protocol": "Frozen patch-token neighbor cosine distance versus Habitat semantic-instance boundaries.",
        "frames_metadata": str(Path(args.frames_metadata).expanduser().resolve()),
        "results": results,
    }
    metrics_path = out_dir / "lingbot_vision_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out_dir, payload)
    print(json.dumps(payload, indent=2))


def _load_frames(metadata_path: Path, max_frames: int) -> list[dict]:
    metadata_path = metadata_path.expanduser().resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = [dict(frame) for frame in payload.get("frames", [])[:max_frames]]
    base = metadata_path.parent
    for frame in frames:
        for key in ("rgb_path", "semantic_npy"):
            path = Path(frame[key])
            if not path.is_file():
                candidate = base / "frames" / path.name
                if candidate.is_file():
                    path = candidate
            frame[key] = str(path)
    return frames


def _neighbor_edges(token_grid: np.ndarray, semantic_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizontal_scores = 1.0 - np.sum(token_grid[:, 1:] * token_grid[:, :-1], axis=-1)
    vertical_scores = 1.0 - np.sum(token_grid[1:] * token_grid[:-1], axis=-1)
    horizontal_labels = semantic_grid[:, 1:] != semantic_grid[:, :-1]
    vertical_labels = semantic_grid[1:] != semantic_grid[:-1]
    edge_grid = np.zeros(semantic_grid.shape, dtype=np.float32)
    edge_grid[:, 1:] = np.maximum(edge_grid[:, 1:], horizontal_scores)
    edge_grid[1:] = np.maximum(edge_grid[1:], vertical_scores)
    return (
        np.concatenate([horizontal_scores.ravel(), vertical_scores.ravel()]),
        np.concatenate([horizontal_labels.ravel(), vertical_labels.ravel()]),
        edge_grid,
    )


def _render_example(
    image_rgb: np.ndarray,
    token_grid: np.ndarray,
    semantic_grid: np.ndarray,
    edge_grid: np.ndarray,
) -> Image.Image:
    size = 512
    rgb = Image.fromarray(image_rgb).resize((size, size), Image.Resampling.BILINEAR)
    flat = token_grid.reshape(-1, token_grid.shape[-1])
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    pca = flat @ vt[:3].T
    lo, hi = np.percentile(pca, [2, 98], axis=0)
    pca = ((pca - lo) / np.maximum(hi - lo, 1e-8)).clip(0, 1)
    pca = (pca.reshape(token_grid.shape[:2] + (3,)) * 255).astype(np.uint8)
    pca_img = Image.fromarray(pca).resize((size, size), Image.Resampling.NEAREST)

    edge = edge_grid.copy()
    lo, hi = np.percentile(edge, [5, 99])
    edge = ((edge - lo) / max(hi - lo, 1e-8)).clip(0, 1)
    edge_img = Image.fromarray((edge * 255).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST).convert("RGB")

    semantic_edge = np.zeros_like(semantic_grid, dtype=np.uint8)
    semantic_edge[:, 1:] |= semantic_grid[:, 1:] != semantic_grid[:, :-1]
    semantic_edge[1:] |= semantic_grid[1:] != semantic_grid[:-1]
    semantic_edge = np.asarray(
        Image.fromarray(semantic_edge * 255).resize((size, size), Image.Resampling.NEAREST)
    )
    oracle_overlay = np.asarray(rgb).copy()
    oracle_overlay[semantic_edge > 0] = (235, 59, 56)
    oracle_img = Image.fromarray(oracle_overlay)

    labels = ["RGB", "Frozen feature PCA", "Feature boundary strength", "Habitat oracle boundaries"]
    panels = [rgb, pca_img, edge_img, oracle_img]
    canvas = Image.new("RGB", (size * 2, size * 2 + 64), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(zip(labels, panels)):
        x = (index % 2) * size
        y = (index // 2) * size + 32
        canvas.paste(panel, (x, y))
        draw.text((x + 12, y - 24), label, fill="#202124")
    return canvas


def _write_report(out_dir: Path, payload: dict) -> None:
    rows = []
    cards = []
    for item in payload["results"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['variant'])}</td>"
            f"<td>{item['parameter_millions']:.2f}M</td>"
            f"<td>{item['boundary_average_precision']:.4f}</td>"
            f"<td>{item['boundary_best_f1']:.4f}</td>"
            f"<td>{item['boundary_roc_auc']:.4f}</td>"
            f"<td>{item['edge_contrast']:.4f}</td>"
            f"<td>{item['fps']:.2f}</td>"
            f"<td>{item['peak_vram_gb']:.2f} GB</td>"
            "</tr>"
        )
        cards.append(
            f"<section><h2>{html.escape(item['variant'].title())}</h2>"
            f"<img src='{html.escape(item['example'])}' alt='{html.escape(item['variant'])} boundary evidence'></section>"
        )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>LingBot-Vision Parameter Scaling</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1400px;margin:28px auto;padding:0 22px;color:#202124}}
table{{border-collapse:collapse;width:100%;margin:18px 0 30px}}th,td{{border:1px solid #d5d8dc;padding:9px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}section{{margin:30px 0}}img{{width:100%;height:auto;border:1px solid #d5d8dc}}
.note{{background:#f4f6f8;border-left:4px solid #3b82f6;padding:14px 16px}}
</style></head><body>
<h1>LingBot-Vision 参数规模实测</h1>
<p class="note">同一批 Habitat RGB，使用 frozen patch token 相邻余弦距离预测 Habitat semantic-instance 边界。
该测试衡量空间边界表征，不等价于开放词汇检测能力；这些权重没有 detector head。</p>
<table><thead><tr><th>Variant</th><th>Params</th><th>Boundary AP</th><th>Best F1</th><th>ROC-AUC</th><th>Edge contrast</th><th>FPS</th><th>Peak VRAM</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{''.join(cards)}
<p><a href="lingbot_vision_metrics.json">Raw metrics JSON</a></p>
</body></html>"""
    (out_dir / "lingbot_vision_report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
