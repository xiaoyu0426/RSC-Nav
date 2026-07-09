from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter M2.5 grounding candidates for multi-view consistency and confidence.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--labels", default="", help="Optional comma-separated label allowlist.")
    parser.add_argument("--max-per-label", type=int, default=0, help="Optional top-N candidates per label after filtering.")
    parser.add_argument("--min-sam-iou", type=float, default=0.0)
    parser.add_argument("--min-depth-valid-ratio", type=float, default=0.0)
    args = parser.parse_args()

    src = Path(args.input_json).expanduser().resolve()
    dst = Path(args.out_json).expanduser().resolve()
    payload = json.loads(src.read_text(encoding="utf-8"))
    allow_labels = {item.strip().lower() for item in args.labels.split(",") if item.strip()}

    filtered = []
    dropped = []
    for item in payload.get("items", []):
        reasons = []
        label = str(item.get("label", "")).lower()
        views = len(item.get("source_view_ids", []) or [])
        confidence = float(item.get("confidence", 0.0))
        raw = item.get("raw", {}) or {}
        sam_iou = raw.get("mean_sam_iou_score")
        depth_valid_ratio = raw.get("mean_depth_valid_ratio")
        if allow_labels and label not in allow_labels:
            reasons.append("label_not_allowed")
        if views < int(args.min_views):
            reasons.append("too_few_views")
        if confidence < float(args.min_confidence):
            reasons.append("low_confidence")
        if sam_iou is not None and float(sam_iou) < float(args.min_sam_iou):
            reasons.append("low_sam_iou")
        if depth_valid_ratio is not None and float(depth_valid_ratio) < float(args.min_depth_valid_ratio):
            reasons.append("low_depth_valid_ratio")
        if reasons:
            dropped.append(
                {
                    "id": item.get("id"),
                    "label": label,
                    "confidence": confidence,
                    "views": views,
                    "sam_iou": sam_iou,
                    "depth_valid_ratio": depth_valid_ratio,
                    "reasons": reasons,
                }
            )
        else:
            filtered.append(item)

    if int(args.max_per_label) > 0:
        kept = []
        overflow = []
        by_label: dict[str, list[dict[str, Any]]] = {}
        for item in filtered:
            by_label.setdefault(str(item.get("label", "")), []).append(item)
        for label, rows in by_label.items():
            rows.sort(key=lambda item: (float(item.get("confidence", 0.0)), len(item.get("source_view_ids", []) or [])), reverse=True)
            kept.extend(rows[: int(args.max_per_label)])
            for item in rows[int(args.max_per_label) :]:
                overflow.append(
                    {
                        "id": item.get("id"),
                        "label": label,
                        "confidence": float(item.get("confidence", 0.0)),
                        "views": len(item.get("source_view_ids", []) or []),
                        "reasons": ["max_per_label"],
                    }
                )
        filtered = kept
        dropped.extend(overflow)

    out_payload = dict(payload)
    out_payload["items"] = filtered
    metadata = dict(payload.get("metadata", {}))
    metadata["post_filter"] = {
        "source": str(src),
        "min_views": int(args.min_views),
        "min_confidence": float(args.min_confidence),
        "labels": sorted(allow_labels),
        "max_per_label": int(args.max_per_label),
        "min_sam_iou": float(args.min_sam_iou),
        "min_depth_valid_ratio": float(args.min_depth_valid_ratio),
        "input_count": len(payload.get("items", [])),
        "output_count": len(filtered),
        "dropped_count": len(dropped),
    }
    out_payload["metadata"] = metadata
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    (dst.parent / f"{dst.stem}_filter_report.json").write_text(
        json.dumps({"metadata": metadata["post_filter"], "dropped": dropped}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata["post_filter"], indent=2))


if __name__ == "__main__":
    main()
