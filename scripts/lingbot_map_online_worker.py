from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent LingBot-Map causal inference worker.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-pt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        if not hasattr(torch.nn, "attention") or not hasattr(
            torch.nn.attention, "flex_attention"
        ):
            if not hasattr(torch.nn, "attention"):
                attention_module = types.ModuleType("torch.nn.attention")
                attention_module.__path__ = []
                torch.nn.attention = attention_module
                sys.modules["torch.nn.attention"] = attention_module
            flex_module = types.ModuleType("torch.nn.attention.flex_attention")

            class BlockMask:
                pass

            def create_mask(*_args, **_kwargs):
                raise RuntimeError("FlexAttention is unavailable in this PyTorch build")

            flex_module.BlockMask = BlockMask
            flex_module.create_mask = create_mask
            torch.nn.attention.flex_attention = flex_module
            sys.modules["torch.nn.attention.flex_attention"] = flex_module
        from lingbot_map.models.gct_stream import GCTStream
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        model = GCTStream(
            img_size=518,
            patch_size=14,
            enable_3d_rope=True,
            max_frame_num=4096,
            kv_cache_sliding_window=64,
            kv_cache_scale_frames=int(args.num_scale_frames),
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=True,
            camera_num_iterations=int(args.camera_num_iterations),
        )
        device = torch.device(args.device)
        checkpoint = torch.load(
            Path(args.model_pt).expanduser().resolve(),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        model = model.to(device)
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
        model.eval()

    state = {
        "bootstrapped": False,
        "frame_count": 0,
        "scale_frames": int(args.num_scale_frames),
    }
    emit(
        {
            "type": "ready",
            "model": "LingBot-Map-long",
            "num_scale_frames": state["scale_frames"],
            "device": str(args.device),
        }
    )

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_type = request.get("type")
            if request_type == "shutdown":
                emit({"type": "shutdown"})
                return
            if request_type == "bootstrap":
                paths = [Path(value).expanduser().resolve() for value in request["rgb_paths"]]
                if len(paths) != state["scale_frames"]:
                    raise ValueError(
                        f"bootstrap requires {state['scale_frames']} frames, got {len(paths)}"
                    )
                model.clean_kv_cache()
                images = load_images(load_and_preprocess_images, paths)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward(
                        images.unsqueeze(0).to(device),
                        num_frame_for_scale=state["scale_frames"],
                        num_frame_per_block=state["scale_frames"],
                        causal_inference=True,
                    )
                results = save_predictions(
                    output,
                    output_dir=output_dir,
                    start_index=0,
                    image_shape=tuple(int(value) for value in images.shape[-2:]),
                )
                state["bootstrapped"] = True
                state["frame_count"] = len(results)
                emit(
                    {
                        "type": "bootstrap_result",
                        "request_id": request.get("request_id"),
                        "results": results,
                    }
                )
                continue
            if request_type == "infer":
                if not state["bootstrapped"]:
                    raise RuntimeError("worker must be bootstrapped before infer")
                path = Path(request["rgb_path"]).expanduser().resolve()
                images = load_images(load_and_preprocess_images, [path])
                frame_index = int(request.get("frame_index", state["frame_count"]))
                is_keyframe = (
                    int(args.keyframe_interval) <= 1
                    or (frame_index - state["scale_frames"]) % int(args.keyframe_interval) == 0
                )
                if not is_keyframe:
                    model._set_skip_append(True)
                try:
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        output = model.forward(
                            images.unsqueeze(0).to(device),
                            num_frame_for_scale=state["scale_frames"],
                            num_frame_per_block=1,
                            causal_inference=True,
                        )
                finally:
                    if not is_keyframe:
                        model._set_skip_append(False)
                results = save_predictions(
                    output,
                    output_dir=output_dir,
                    start_index=frame_index,
                    image_shape=tuple(int(value) for value in images.shape[-2:]),
                )
                state["frame_count"] += 1
                emit(
                    {
                        "type": "result",
                        "request_id": request.get("request_id"),
                        "result": results[0],
                        "is_keyframe": is_keyframe,
                    }
                )
                continue
            raise ValueError(f"unknown request type: {request_type!r}")
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "request_id": request.get("request_id") if "request" in locals() else None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


def load_images(load_and_preprocess_images: Any, paths: list[Path]):
    with contextlib.redirect_stdout(sys.stderr):
        return load_and_preprocess_images(
            [str(path) for path in paths],
            mode="crop",
            image_size=518,
            patch_size=14,
        )


def save_predictions(
    output: dict[str, Any],
    output_dir: Path,
    start_index: int,
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    pose_enc = output["pose_enc"].detach().float().cpu()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_shape)
    depth = output["depth"].detach().float().cpu().numpy()
    confidence = output["depth_conf"].detach().float().cpu().numpy()
    extrinsic = extrinsic.detach().float().cpu().numpy()
    intrinsic = intrinsic.detach().float().cpu().numpy()
    depth = normalize_frame_axis(depth)
    confidence = normalize_frame_axis(confidence)
    extrinsic = normalize_frame_axis(extrinsic)
    intrinsic = normalize_frame_axis(intrinsic)

    results = []
    for offset in range(depth.shape[0]):
        frame_index = int(start_index + offset)
        depth_frame = np.asarray(depth[offset]).squeeze().astype(np.float32)
        confidence_frame = np.asarray(confidence[offset]).squeeze().astype(np.float32)
        depth_path = output_dir / f"frame_{frame_index:04d}_lingbot_depth.npy"
        confidence_path = output_dir / f"frame_{frame_index:04d}_lingbot_conf.npy"
        np.save(depth_path, depth_frame)
        np.save(confidence_path, confidence_frame)
        results.append(
            {
                "frame_index": frame_index,
                "depth_npy": str(depth_path),
                "depth_conf_npy": str(confidence_path),
                "extrinsic_c2w": np.asarray(extrinsic[offset]).tolist(),
                "intrinsic": np.asarray(intrinsic[offset]).tolist(),
            }
        )
    return results


def normalize_frame_axis(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim >= 2 and value.shape[0] == 1:
        value = value[0]
    return value


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
