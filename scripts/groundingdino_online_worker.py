from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

from m25_groundingdino_export import _detect, _load_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent JSONL GroundingDINO inference worker.")
    parser.add_argument("--model-id", default="downloads/hf_models/grounding-dino-tiny")
    parser.add_argument("--labels", default="cup,mug,bottle,table,counter,sink")
    parser.add_argument("--box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--max-detections", type=int, default=16)
    args = parser.parse_args()

    labels = [item.strip().lower() for item in args.labels.split(",") if item.strip()]
    load_started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        detector = _load_detector(args.model_id)
    _write(
        {
            "type": "ready",
            "model_id": args.model_id,
            "labels": labels,
            "load_ms": (time.perf_counter() - load_started) * 1000.0,
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("type") == "shutdown":
            _write({"type": "shutdown_ack"})
            return
        request_id = request.get("request_id")
        image_path = Path(request["rgb_path"]).expanduser().resolve()
        request_labels = [
            str(item).strip().lower()
            for item in request.get("labels", labels)
            if str(item).strip()
        ]
        started = time.perf_counter()
        try:
            image = Image.open(image_path).convert("RGB")
            with contextlib.redirect_stdout(sys.stderr):
                detections = _detect(
                    detector,
                    image,
                    request_labels,
                    box_threshold=float(request.get("box_threshold", args.box_threshold)),
                    text_threshold=float(request.get("text_threshold", args.text_threshold)),
                    max_detections=int(request.get("max_detections", args.max_detections)),
                )
            _write(
                {
                    "type": "result",
                    "request_id": request_id,
                    "rgb_path": str(image_path),
                    "labels": request_labels,
                    "detections": detections,
                    "inference_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        except Exception as exc:
            _write(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "inference_ms": (time.perf_counter() - started) * 1000.0,
                }
            )


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
