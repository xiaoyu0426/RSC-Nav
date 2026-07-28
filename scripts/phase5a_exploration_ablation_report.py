from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exploration-policy ablation runs.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = [load_run(label, Path(path).expanduser().resolve()) for label, path in args.run]
    output.write_text(render_report(runs, output.parent), encoding="utf-8")
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"html": str(output), "metrics": str(metrics_path)}, indent=2))


def load_run(label: str, run_dir: Path) -> dict:
    summary = read_json(run_dir / "online_summary.json")
    coverage = read_json(run_dir / "posthoc_coverage_metrics.json")
    trace = [
        json.loads(line)
        for line in (run_dir / "online_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    explored = np.asarray(
        [row.get("bev", {}).get("num_explored_cells", 0) for row in trace],
        dtype=np.float64,
    )
    increments = np.diff(explored, prepend=0.0)
    actions = Counter(str(row.get("action")) for row in trace)
    modes = Counter(str(row.get("interest", {}).get("mode")) for row in trace)
    moved_m = float(sum(float(row.get("moved_m", 0.0)) for row in trace))
    final_coverage = float(coverage.get("navmesh_observation_coverage", 0.0))
    low_gain_ratio = float(np.mean(increments < 20.0)) if increments.size else 0.0
    explored_auc = float(np.mean(explored)) if explored.size else 0.0
    report_dir = run_dir / "report"
    return {
        "label": label,
        "run_dir": str(run_dir),
        "frontier_strategy": summary.get("frontier_strategy"),
        "geometry_source": summary.get("geometry_source", "habitat_rgbd"),
        "steps": len(trace),
        "stop_reason": summary.get("stop_reason"),
        "navmesh_coverage": final_coverage,
        "explored_cells": int(explored[-1]) if explored.size else 0,
        "explored_cells_auc": explored_auc,
        "low_gain_frame_ratio": low_gain_ratio,
        "moved_m": moved_m,
        "coverage_per_meter": final_coverage / max(moved_m, 1e-6),
        "move_forward_actions": int(actions.get("move_forward", 0)),
        "scan_actions": int(
            modes.get("initial_panorama_scan", 0)
            + modes.get("coverage_viewpoint_scan", 0)
            + modes.get("coverage_completion_scan", 0)
            + modes.get("semantic_surface_scan", 0)
        ),
        "collisions": int(summary.get("num_detected_collisions", 0)),
        "stuck_events": int(summary.get("num_stuck_events", 0)),
        "candidate_cups": int(summary.get("num_candidate_cups", 0)),
        "scanned_surface_regions": int(summary.get("num_scanned_surface_regions", 0)),
        "loop_mean_ms": float(
            summary.get("timing_ms", {}).get("total", {}).get("mean", 0.0)
        ),
        "policy_mean_ms": float(
            summary.get("timing_ms", {}).get("policy", {}).get("mean", 0.0)
        ),
        "lingbot_depth_scale": summary.get("lingbot_depth_scale"),
        "final_bev": str(run_dir / "bev_frames" / f"frame_{len(trace) - 1:04d}_bev.png"),
        "report_html": str(report_dir / "online_interest_exploration.html"),
        "report_gif": str(report_dir / "online_interest_exploration.gif"),
    }


def render_report(runs: list[dict], base: Path) -> str:
    findings = ""
    if len(runs) >= 3:
        baseline, hierarchical, lingbot = runs[:3]
        policy_coverage_delta = 100.0 * (
            hierarchical["navmesh_coverage"] - baseline["navmesh_coverage"]
        )
        lingbot_coverage_delta = 100.0 * (
            lingbot["navmesh_coverage"] - hierarchical["navmesh_coverage"]
        )
        policy_low_gain_delta = 100.0 * (
            hierarchical["low_gain_frame_ratio"] - baseline["low_gain_frame_ratio"]
        )
        lingbot_latency_delta = (
            lingbot["loop_mean_ms"] - hierarchical["loop_mean_ms"]
        )
        findings = f"""
<section class="findings">
<h2>本轮结论</h2>
<p><strong>层次策略 vs. 贪心：</strong>后验覆盖差
<b>{policy_coverage_delta:+.2f} pp</b>，低增益帧差
<b>{policy_low_gain_delta:+.1f} pp</b>，扫描动作
<b>{hierarchical['scan_actions']} vs. {baseline['scan_actions']}</b>。
覆盖率若接近，只能说明策略达到非劣，不能表述为显著提升。</p>
<p><strong>LingBot 深度 vs. Habitat 深度：</strong>同策略下后验覆盖差
<b>{lingbot_coverage_delta:+.2f} pp</b>，单步闭环延迟增加
<b>{lingbot_latency_delta:+.0f} ms</b>。该组证明预测几何能接入在线闭环；
仍使用 Habitat exact pose 和一次性因果尺度标定，不等同于完整 RGB-only SLAM。</p>
<p class="caveat">“探索格”和“低增益帧”来自各自在线地图。预测深度可能把噪声投成已探索区域，
因此跨几何前端的主指标采用仅在实验后计算的 navmesh observation coverage。</p>
</section>"""
    rows = "".join(
        "<tr>"
        f"<th>{html.escape(run['label'])}</th>"
        f"<td>{html.escape(str(run['frontier_strategy']))}</td>"
        f"<td>{html.escape(str(run['geometry_source']))}</td>"
        f"<td>{run['steps']}</td>"
        f"<td>{100.0 * run['navmesh_coverage']:.2f}%</td>"
        f"<td>{run['explored_cells']:,}</td>"
        f"<td>{100.0 * run['low_gain_frame_ratio']:.1f}%</td>"
        f"<td>{run['moved_m']:.1f} m</td>"
        f"<td>{run['scan_actions']}</td>"
        f"<td>{run['collisions']} / {run['stuck_events']}</td>"
        f"<td>{run['loop_mean_ms']:.0f} ms</td>"
        "</tr>"
        for run in runs
    )
    cards = "".join(
        "<article>"
        f"<h2>{html.escape(run['label'])}</h2>"
        f"<div class='coverage'><span style='width:{100.0 * run['navmesh_coverage']:.2f}%'></span></div>"
        f"<p class='coverage-label'>Post-hoc coverage: <b>{100.0 * run['navmesh_coverage']:.2f}%</b></p>"
        f"<img src='{html.escape(relative(base, Path(run['final_bev'])))}'>"
        "<p>"
        f"<a href='{html.escape(relative(base, Path(run['report_html'])))}'>HTML report</a> · "
        f"<a href='{html.escape(relative(base, Path(run['report_gif'])))}'>GIF</a>"
        "</p>"
        f"<code>{html.escape(run['stop_reason'])}</code>"
        "</article>"
        for run in runs
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RSC-Nav Exploration Ablation</title>
<style>
body{{margin:0;background:#eef2f5;color:#17212b;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1440px;margin:auto;padding:32px}} h1{{font-size:32px;margin:0 0 8px}}
.note{{max-width:980px;color:#4c5c68}} .findings{{margin:24px 0;padding:20px 0;border-top:2px solid #17212b}}
.findings p{{max-width:1100px}} .caveat{{color:#586773}} table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:12px;border-bottom:1px solid #dde3e8;text-align:left}} thead th{{background:#17212b;color:white}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:24px}}
article{{background:white;border:1px solid #d8e0e6;padding:16px;border-radius:6px}} article img{{width:100%;display:block}}
.coverage{{height:10px;background:#dfe6eb;margin:8px 0}} .coverage span{{display:block;height:100%;background:#15806f}}
.coverage-label{{margin:0 0 12px;color:#43535e}}
a{{color:#086ba8}} code{{color:#50616e}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>实时兴趣探索 × LingBot-Map 对照</h1>
<p class="note">A/B 只改变 frontier 策略；B/C 只改变深度几何来源。LingBot 组使用
8 帧因果启动、固定深度尺度标定和 Habitat exact pose，因此属于 RGB-depth diagnostic，
不是完整 RGB-only 导航结论。三组从相同位姿出发并使用相同主步数预算。
Habitat navmesh 只负责统一低层执行与 episode 后覆盖评估。</p>
{findings}
<table><thead><tr><th>组别</th><th>策略</th><th>几何</th><th>步数</th><th>后验覆盖</th>
<th>探索格</th><th>低增益帧</th><th>位移</th><th>扫描动作</th><th>碰撞/卡住</th><th>均值延迟</th>
</tr></thead><tbody>{rows}</tbody></table>
<div class="grid">{cards}</div>
</main></body></html>"""


def relative(base: Path, target: Path) -> str:
    import os

    return os.path.relpath(target, base)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
