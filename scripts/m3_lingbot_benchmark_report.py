from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the combined LingBot foundation-model benchmark report.")
    parser.add_argument("--map-metrics", required=True)
    parser.add_argument("--depth-metrics", required=True)
    parser.add_argument("--vision-metrics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    map_metrics = _read(args.map_metrics)
    depth_metrics = _read(args.depth_metrics)
    vision_metrics = _read(args.vision_metrics)
    output.write_text(_report(output, map_metrics, depth_metrics, vision_metrics), encoding="utf-8")
    print(output)


def _read(path: str) -> dict:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _rel(output: Path, path: Path) -> str:
    return path.resolve().relative_to(output.parent.resolve()).as_posix()


def _report(output: Path, map_metrics: dict, depth_metrics: dict, vision_metrics: dict) -> str:
    map_report = map_metrics["outputs"].get("report")
    map_dir = Path(map_report).parent if map_report else output.parent / "map_vs_vggt_first16_20260723_v2"
    if not map_dir.is_dir():
        map_dir = output.parent / "map_vs_vggt_first16_20260723_v2"
    depth_dir = Path(output.parent / "depth_recovery_first16_20260723")
    vision_dir = Path(output.parent / "vision_scaling_96_20260723")

    current = map_metrics["bev_comparison"]
    baseline = map_metrics["vggt_baseline"]
    baseline_bev = baseline["bev_comparison"]
    baseline_ate = baseline["alignment"]["ate_rmse_m"]
    current_ate = map_metrics["alignment"]["ate_rmse_m"]
    ate_gain = 100.0 * (baseline_ate - current_ate) / baseline_ate
    map_rows = f"""
<tr><td>VGGT-1B</td><td>{baseline_ate:.3f}</td><td>{baseline_bev['explored_iou']:.3f}</td>
<td>{baseline_bev['free_iou']:.3f}</td><td>{baseline_bev['occupied_iou']:.3f}</td><td>-</td><td>-</td></tr>
<tr class="winner"><td>LingBot-Map-long</td><td>{current_ate:.3f}</td><td>{current['explored_iou']:.3f}</td>
<td>{current['free_iou']:.3f}</td><td>{current['occupied_iou']:.3f}</td>
<td>{map_metrics['runtime']['fps']:.2f}</td><td>{map_metrics['runtime']['peak_vram_gb']:.2f} GB</td></tr>"""

    nearest = depth_metrics["baselines"]["nearest_fill"]
    depth_rows = [
        f"<tr><td>Nearest fill</td><td>{nearest['mae_m']:.3f}</td><td>{nearest['rmse_m']:.3f}</td>"
        f"<td>{nearest['abs_rel']:.3f}</td><td>{nearest['delta1']:.3f}</td>"
        f"<td>{nearest['bev_comparison']['free_iou']:.3f}</td><td>{nearest['bev_comparison']['occupied_iou']:.3f}</td>"
        "<td>-</td></tr>"
    ]
    for item in depth_metrics["results"]:
        depth_rows.append(
            f"<tr><td>{html.escape(item['model'])}</td><td>{item['mae_m']:.3f}</td><td>{item['rmse_m']:.3f}</td>"
            f"<td>{item['abs_rel']:.3f}</td><td>{item['delta1']:.3f}</td>"
            f"<td>{item['bev_comparison']['free_iou']:.3f}</td><td>{item['bev_comparison']['occupied_iou']:.3f}</td>"
            f"<td>{item['fps']:.2f}</td></tr>"
        )

    vision_rows = []
    for item in vision_metrics["results"]:
        vision_rows.append(
            f"<tr><td>{html.escape(item['variant'].title())}</td><td>{item['parameter_millions']:.2f}M</td>"
            f"<td>{item['boundary_average_precision']:.3f}</td><td>{item['boundary_best_f1']:.3f}</td>"
            f"<td>{item['boundary_roc_auc']:.3f}</td><td>{item['edge_contrast']:.3f}</td>"
            f"<td>{item['fps']:.1f}</td><td>{item['peak_vram_gb']:.2f} GB</td></tr>"
        )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>LingBot 基模对 RSC-Nav 的可用性实测</title>
<style>
:root{{--ink:#202124;--muted:#5f6368;--line:#d8dde3;--blue:#2463eb;--green:#16794b;--soft:#f5f7fa}}
body{{font-family:Arial,"Noto Sans SC",sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;color:var(--ink);line-height:1.55}}
h1{{font-size:32px;margin-bottom:8px}}h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.lede{{font-size:18px;color:var(--muted)}}.verdict{{background:#eef7f2;border-left:5px solid var(--green);padding:16px 18px}}
.warning{{background:#fff7e8;border-left:5px solid #d97706;padding:14px 18px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.inventory{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.item{{border:1px solid var(--line);padding:14px;background:white}}
.item strong{{display:block;margin-bottom:6px}}table{{border-collapse:collapse;width:100%;margin:16px 0 24px}}
th,td{{border:1px solid var(--line);padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:var(--soft)}}tr.winner{{background:#eef7f2}}img{{width:100%;height:auto;border:1px solid var(--line)}}
a{{color:var(--blue)}}code{{background:#f1f3f4;padding:2px 5px}}@media(max-width:900px){{.grid,.inventory{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>LingBot 基模对 RSC-Nav 的可用性实测</h1>
<p class="lede">固定 Habitat RGB-D/semantic oracle，分别测试 RGB-only 几何、深度修复和 dense spatial backbone 参数扩展。</p>
<p class="verdict"><strong>结论：</strong>现在最值得接入的是 <strong>LingBot-Map-long</strong>，可替换当前 VGGT RGB-only geometry MVP；
LingBot-Depth v0.5 适合作为未来真实 RGB-D 传感器修复支路，但本组 BEV 指标没有超过简单基线；
LingBot-Vision 暂不替换 GroundingDINO，因为其开源权重只有 backbone，无开放词汇检测头，且 21.6M 到 1.13B 参数的边界指标没有单调提升。</p>

<h2>当前可用模型</h2>
<div class="inventory">
<div class="item"><strong>LingBot-Map</strong>RGB sequence -> pose/depth/point cloud。与当前 VGGT 前端同类，已完成可执行对照。</div>
<div class="item"><strong>LingBot-Depth v0.5 / DC</strong>RGB + 不完整 depth -> refined metric depth。需要原始 depth，不是纯 RGB 深度模型。</div>
<div class="item"><strong>LingBot-Vision S/B/L/G</strong>21.6M / 85.7M / 303M / 1.13B frozen ViT backbone。需要另接 detector/segmenter head。</div>
<div class="item"><strong>LingBot-VLA 4B / VLA 2.0 6B</strong>机器人 action policy；需要匹配 embodiment 和后训练，不直接输出 RSC-Nav waypoint。</div>
<div class="item"><strong>LingBot-World / VA</strong>视频世界模型与 video-action 模型；不直接产生当前 BEV 或开放词汇对象证据。</div>
<div class="item"><strong>LingBot-Depth 2.0</strong>官方 Vision 报告展示了结果，但当前公开 Depth model zoo 仍是 v0.5/DC，未作为本次可下载 checkpoint 测试。</div>
</div>

<h2>1. LingBot-Map vs VGGT</h2>
<p>同一连续 16 帧，均以 Sim(3) 对齐 Habitat world frame 后进入相同 DenseBEVMapper。LingBot-Map 将 ATE RMSE 降低
<strong>{ate_gain:.1f}%</strong>，且三个 BEV IoU 全部提高。</p>
<table><thead><tr><th>Geometry frontend</th><th>ATE RMSE m</th><th>Explored IoU</th><th>Free IoU</th>
<th>Occupied IoU</th><th>FPS</th><th>Peak VRAM</th></tr></thead><tbody>{map_rows}</tbody></table>
<div class="grid"><img src="{_rel(output, map_dir / 'oracle_bev.png')}"><img src="{_rel(output, map_dir / 'lingbot_map_bev.png')}"></div>
<img src="{_rel(output, map_dir / 'lingbot_map_depth_contact_sheet.png')}">
<p><a href="{_rel(output, map_dir / 'lingbot_map_report.html')}">打开 Map 完整报告</a></p>

<h2>2. LingBot-Depth 传感器修复</h2>
<p>对 Habitat gold depth 加 3cm 高斯噪声、12% 随机掉点与块状空洞。General v0.5 的像素误差优于 nearest fill，
但 BEV free/occupied IoU 未超过简单填充；DC 在这组“稠密但破损”的协议上更差，符合其偏向 sparse completion 的定位。</p>
<table><thead><tr><th>Method</th><th>MAE m</th><th>RMSE m</th><th>AbsRel</th><th>delta1</th>
<th>Free IoU</th><th>Occupied IoU</th><th>FPS</th></tr></thead><tbody>{''.join(depth_rows)}</tbody></table>
<img src="{_rel(output, depth_dir / 'depth_refinement_contact_sheet.png')}">
<p class="warning">这里不能说“Depth 对当前 Habitat oracle 有提升”：oracle 本身就是无噪声 gold。它的价值是未来真机 RGB-D
出现玻璃、反光、空洞和飞点时提供修复候选，还需用真实传感器数据复验。</p>
<p><a href="{_rel(output, depth_dir / 'lingbot_depth_report.html')}">打开 Depth 完整报告</a></p>

<h2>3. LingBot-Vision 参数扩展</h2>
<p>96 帧 frozen patch token，以相邻 token 余弦距离预测 Habitat semantic-instance 边界。Small 的 AP 最高，
Giant 的 edge contrast 最高，但 Giant 参数量约为 Small 的 52.5 倍、吞吐约为 36%。这组代理任务不支持“参数越大越好”。</p>
<table><thead><tr><th>Variant</th><th>Params</th><th>Boundary AP</th><th>Best F1</th><th>ROC-AUC</th>
<th>Edge contrast</th><th>FPS</th><th>Peak VRAM</th></tr></thead><tbody>{''.join(vision_rows)}</tbody></table>
<div class="grid"><img src="{_rel(output, vision_dir / 'vision_small_boundary_example.png')}">
<img src="{_rel(output, vision_dir / 'vision_giant_boundary_example.png')}"></div>
<p><a href="{_rel(output, vision_dir / 'lingbot_vision_report.html')}">打开 Vision 完整报告</a></p>

<h2>主线建议</h2>
<ol>
<li>新增 <code>LingBot-Map-long</code> 为 M3 的首选 RGB-only geometry candidate，保留 VGGT 为 baseline。</li>
<li>下一次做 96 帧重叠窗口/streaming 长序列测试，再把 GroundingDINO+SAM evidence 投影到 LingBot-Map geometry。</li>
<li>LingBot-Depth 保持 optional real-RGB-D refinement，不进入当前 Habitat oracle 默认路径。</li>
<li>LingBot-Vision Small 可作为轻量 dense-feature 研究支路；在训练 detection/segmentation head 前不替换 GroundingDINO。</li>
<li>VLA/VLA 2.0 暂不纳入当前 planner：现阶段 qwen3-max 输出语义任务规划与 waypoint，底层 navmesh 执行，接口更匹配。</li>
</ol>
</body></html>"""


if __name__ == "__main__":
    main()
