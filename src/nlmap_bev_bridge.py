from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import json
import math
import struct

import numpy as np


HorizontalAxes = Literal["xz", "xy"]


@dataclass
class BridgeObject:
    id: str
    label: str
    position_3d: tuple[float, float, float]
    bev_position: tuple[float, float]
    confidence: float
    context_id: str
    source: str = "imported"
    source_view_ids: list[str] = field(default_factory=list)
    bbox: dict[str, float] | None = None
    status: str = "active"
    freshness: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["position_3d"] = [float(value) for value in self.position_3d]
        data["bev_position"] = [float(value) for value in self.bev_position]
        return data

    def to_rsc_memory_item(self, semantic_id: int, step: int = 0) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.label,
            "semantic_id": int(semantic_id),
            "object_id": self.id,
            "position_3d": [float(value) for value in self.position_3d],
            "bev_position": [float(value) for value in self.bev_position],
            "centroid_xz": [float(value) for value in self.bev_position],
            "confidence": float(self.confidence),
            "freshness": float(self.freshness),
            "status": self.status,
            "context_id": self.context_id,
            "source": self.source,
            "source_view_ids": list(self.source_view_ids),
            "bbox": self.bbox,
            "first_seen_step": int(step),
            "last_seen_step": int(step),
            "visible_steps": [int(step)],
        }


@dataclass
class BridgeGrid:
    resolution_m: float
    origin: tuple[float, float]
    width: int
    height: int
    horizontal_axes: HorizontalAxes

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution_m": float(self.resolution_m),
            "origin": [float(self.origin[0]), float(self.origin[1])],
            "width": int(self.width),
            "height": int(self.height),
            "horizontal_axes": self.horizontal_axes,
        }


def load_nlmap_objects(path: str | Path, context_id: str, horizontal_axes: HorizontalAxes = "xz") -> list[BridgeObject]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "items" in data:
            raw_objects = list(data["items"])
        elif "objects" in data:
            raw_objects = list(data["objects"])
        else:
            raw_objects = [_inventory_entry_to_object(name, value) for name, value in data.items()]
    else:
        raw_objects = list(data)
    return normalize_nlmap_objects(raw_objects, context_id=context_id, horizontal_axes=horizontal_axes)


def load_pointcloud_json(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = data.get("points", data) if isinstance(data, dict) else data
    return _points_array(points)


def load_ascii_pcd(path: str | Path, max_points: int = 6000) -> np.ndarray:
    return load_pcd(path, max_points=max_points)


def load_pcd(path: str | Path, max_points: int = 6000) -> np.ndarray:
    pcd_path = Path(path)
    with pcd_path.open("rb") as handle:
        header, data_offset = _read_pcd_header(handle)
    data_mode = " ".join(header.get("DATA", [])).lower()
    if data_mode == "ascii":
        return _load_ascii_pcd_sampled(pcd_path, header=header, data_offset=data_offset, max_points=max_points)
    if data_mode == "binary":
        return _load_binary_pcd_sampled(pcd_path, header=header, data_offset=data_offset, max_points=max_points)
    raise ValueError(f"Unsupported PCD DATA mode in {path}: {data_mode or 'missing'}")


def _load_ascii_pcd_sampled(path: Path, header: dict[str, list[str]], data_offset: int, max_points: int) -> np.ndarray:
    fields = header.get("FIELDS", ["x", "y", "z"])
    total_points = _pcd_point_count(header)
    stride = max(1, total_points // max(1, int(max_points)))
    fields: list[str] = []
    fields.extend(header.get("FIELDS", ["x", "y", "z"]))
    try:
        x_idx, y_idx, z_idx = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError as exc:
        raise ValueError(f"PCD file {path} does not contain x/y/z fields") from exc
    points = []
    with path.open("rb") as handle:
        handle.seek(data_offset)
        for row_idx, raw_line in enumerate(handle):
            if row_idx % stride:
                continue
            values = raw_line.decode("utf-8", errors="ignore").strip().split()
            if len(values) <= max(x_idx, y_idx, z_idx):
                continue
            point = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx])]
            if all(math.isfinite(value) for value in point):
                points.append(point)
            if len(points) >= max_points:
                break
    return _points_array(points)


def _load_binary_pcd_sampled(path: Path, header: dict[str, list[str]], data_offset: int, max_points: int) -> np.ndarray:
    fields = header.get("FIELDS", ["x", "y", "z"])
    sizes = [int(value) for value in header.get("SIZE", ["4"] * len(fields))]
    types = header.get("TYPE", ["F"] * len(fields))
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if len(sizes) != len(fields) or len(types) != len(fields) or len(counts) != len(fields):
        raise ValueError(f"Malformed PCD header in {path}")
    field_offsets: dict[str, int] = {}
    offset = 0
    for field, size, count in zip(fields, sizes, counts, strict=True):
        field_offsets[field] = offset
        offset += size * count
    record_size = offset
    if record_size <= 0:
        raise ValueError(f"Invalid PCD record size in {path}")
    try:
        x_offset, y_offset, z_offset = field_offsets["x"], field_offsets["y"], field_offsets["z"]
    except KeyError as exc:
        raise ValueError(f"PCD file {path} does not contain x/y/z fields") from exc
    total_points = _pcd_point_count(header)
    stride = max(1, total_points // max(1, int(max_points)))
    points = []
    with path.open("rb") as handle:
        for point_index in range(0, total_points, stride):
            handle.seek(data_offset + point_index * record_size)
            record = handle.read(record_size)
            if len(record) < record_size:
                break
            point = [
                _unpack_pcd_scalar(record, x_offset, sizes[fields.index("x")], types[fields.index("x")]),
                _unpack_pcd_scalar(record, y_offset, sizes[fields.index("y")], types[fields.index("y")]),
                _unpack_pcd_scalar(record, z_offset, sizes[fields.index("z")], types[fields.index("z")]),
            ]
            if all(math.isfinite(value) for value in point):
                points.append(point)
            if len(points) >= max_points:
                break
    return _points_array(points)


def _read_pcd_header(handle) -> tuple[dict[str, list[str]], int]:
    header: dict[str, list[str]] = {}
    while True:
        line_bytes = handle.readline()
        if not line_bytes:
            break
        line = line_bytes.decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        header[key] = parts[1:]
        if key == "DATA":
            return header, handle.tell()
    raise ValueError("Could not find DATA header in PCD file")


def _pcd_point_count(header: dict[str, list[str]]) -> int:
    if header.get("POINTS"):
        return int(header["POINTS"][0])
    width = int(header.get("WIDTH", ["0"])[0])
    height = int(header.get("HEIGHT", ["1"])[0])
    return width * height


def _unpack_pcd_scalar(record: bytes, offset: int, size: int, value_type: str) -> float:
    type_key = value_type.upper()
    if type_key == "F" and size == 4:
        return float(struct.unpack_from("<f", record, offset)[0])
    if type_key == "F" and size == 8:
        return float(struct.unpack_from("<d", record, offset)[0])
    if type_key == "I" and size == 4:
        return float(struct.unpack_from("<i", record, offset)[0])
    if type_key == "U" and size == 4:
        return float(struct.unpack_from("<I", record, offset)[0])
    if type_key == "I" and size == 2:
        return float(struct.unpack_from("<h", record, offset)[0])
    if type_key == "U" and size == 2:
        return float(struct.unpack_from("<H", record, offset)[0])
    if type_key == "I" and size == 1:
        return float(struct.unpack_from("<b", record, offset)[0])
    if type_key == "U" and size == 1:
        return float(struct.unpack_from("<B", record, offset)[0])
    raise ValueError(f"Unsupported PCD scalar type: TYPE={value_type} SIZE={size}")


def normalize_nlmap_objects(
    raw_objects: Iterable[dict[str, Any]],
    context_id: str,
    horizontal_axes: HorizontalAxes = "xz",
) -> list[BridgeObject]:
    objects = []
    for index, raw in enumerate(raw_objects):
        normalized = _normalize_one_object(raw, index=index, context_id=context_id, horizontal_axes=horizontal_axes)
        if normalized is not None:
            objects.append(normalized)
    objects.sort(key=lambda item: item.id)
    return objects


def bridge_mock_scene(context_id: str = "nlmap_mock_A", horizontal_axes: HorizontalAxes = "xz") -> tuple[list[BridgeObject], np.ndarray]:
    raw_objects = [
        {"id": "nlmap_sofa_001", "label": "sofa", "position_3d": [-2.0, 0.45, -1.25], "confidence": 0.91, "image_id": "mock_000"},
        {"id": "nlmap_table_001", "label": "table", "position_3d": [0.0, 0.55, -0.8], "confidence": 0.86, "image_id": "mock_001"},
        {"id": "nlmap_chair_001", "label": "chair", "position_3d": [1.25, 0.5, -0.55], "confidence": 0.83, "image_id": "mock_002"},
        {"id": "nlmap_bed_001", "label": "bed", "position_3d": [-2.35, 0.65, 1.85], "confidence": 0.88, "image_id": "mock_003"},
        {"id": "nlmap_door_001", "label": "door", "position_3d": [2.6, 1.0, 1.6], "confidence": 0.79, "image_id": "mock_004"},
        {"id": "nlmap_cup_001", "label": "cup", "position_3d": [0.35, 0.95, -0.72], "confidence": 0.76, "image_id": "mock_005"},
    ]
    objects = normalize_nlmap_objects(raw_objects, context_id=context_id, horizontal_axes=horizontal_axes)
    pointcloud = _mock_room_pointcloud()
    return objects, pointcloud


def make_bridge_grid(
    objects: Iterable[BridgeObject],
    pointcloud: np.ndarray | None,
    resolution_m: float = 0.25,
    padding_m: float = 0.75,
    horizontal_axes: HorizontalAxes = "xz",
) -> BridgeGrid:
    points_2d = []
    if pointcloud is not None and len(pointcloud):
        points_2d.extend(_points_to_bev(pointcloud, horizontal_axes=horizontal_axes).tolist())
    points_2d.extend([list(item.bev_position) for item in objects])
    if not points_2d:
        points_2d = [[0.0, 0.0]]
    array = np.asarray(points_2d, dtype=float)
    mins = array.min(axis=0) - float(padding_m)
    maxs = array.max(axis=0) + float(padding_m)
    resolution_m = max(1e-3, float(resolution_m))
    width = max(1, int(math.ceil((maxs[0] - mins[0]) / resolution_m)) + 1)
    height = max(1, int(math.ceil((maxs[1] - mins[1]) / resolution_m)) + 1)
    return BridgeGrid(
        resolution_m=resolution_m,
        origin=(float(mins[0]), float(mins[1])),
        width=width,
        height=height,
        horizontal_axes=horizontal_axes,
    )


def rasterize_occupancy(pointcloud: np.ndarray | None, grid: BridgeGrid) -> np.ndarray:
    occupancy = np.zeros((grid.height, grid.width), dtype=np.uint8)
    if pointcloud is None or len(pointcloud) == 0:
        return occupancy
    for x, y in _points_to_bev(pointcloud, horizontal_axes=grid.horizontal_axes):
        cell = world_to_cell((float(x), float(y)), grid)
        if cell is not None:
            col, row = cell
            occupancy[row, col] = 1
    return occupancy


def rasterize_semantic(
    objects: Iterable[BridgeObject],
    grid: BridgeGrid,
    label_to_id: dict[str, int],
    radius_m: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    semantic = np.zeros((grid.height, grid.width), dtype=np.int32)
    confidence = np.zeros((grid.height, grid.width), dtype=np.float32)
    radius_cells = max(1, int(math.ceil(float(radius_m) / grid.resolution_m)))
    for item in objects:
        cell = world_to_cell(item.bev_position, grid)
        if cell is None:
            continue
        col, row = cell
        label_id = int(label_to_id[item.label])
        for rr in range(max(0, row - radius_cells), min(grid.height, row + radius_cells + 1)):
            for cc in range(max(0, col - radius_cells), min(grid.width, col + radius_cells + 1)):
                if (rr - row) ** 2 + (cc - col) ** 2 > radius_cells**2:
                    continue
                if item.confidence >= confidence[rr, cc]:
                    semantic[rr, cc] = label_id
                    confidence[rr, cc] = float(item.confidence)
    return semantic, confidence


def semantic_label_ids(objects: Iterable[BridgeObject]) -> dict[str, int]:
    labels = sorted({item.label for item in objects})
    return {label: index + 1 for index, label in enumerate(labels)}


def world_to_cell(position: tuple[float, float], grid: BridgeGrid) -> tuple[int, int] | None:
    col = int(round((float(position[0]) - grid.origin[0]) / grid.resolution_m))
    row = int(round((float(position[1]) - grid.origin[1]) / grid.resolution_m))
    if col < 0 or row < 0 or col >= grid.width or row >= grid.height:
        return None
    return col, row


def export_rsc_memory(objects: Iterable[BridgeObject], label_to_id: dict[str, int], scene_id: str) -> dict[str, Any]:
    items = [item.to_rsc_memory_item(semantic_id=label_to_id[item.label]) for item in objects]
    return {
        "scene_id": scene_id,
        "source": "nlmap_style_bridge",
        "items": items,
    }


def _normalize_one_object(
    raw: dict[str, Any],
    index: int,
    context_id: str,
    horizontal_axes: HorizontalAxes,
) -> BridgeObject | None:
    label = str(raw.get("label") or raw.get("name") or raw.get("category") or "").strip().lower()
    if not label:
        return None
    position = raw.get("position_3d", raw.get("position", raw.get("centroid_3d")))
    if position is None:
        bev_position_raw = raw.get("bev_position", raw.get("centroid_xz"))
        if bev_position_raw is None:
            return None
        bev_position = (float(bev_position_raw[0]), float(bev_position_raw[1]))
        position_3d = (bev_position[0], 0.0, bev_position[1])
    else:
        position_3d = _position_3d(position)
        bev_position = _position_to_bev(position_3d, horizontal_axes=horizontal_axes)
    object_id = str(raw.get("id") or raw.get("object_id") or f"nlmap_{label}_{index:03d}").replace(" ", "_")
    image_id = raw.get("image_id")
    source_view_ids = raw.get("source_view_ids", raw.get("view_ids", []))
    if image_id is not None and str(image_id) not in [str(value) for value in source_view_ids]:
        source_view_ids = [*source_view_ids, str(image_id)]
    bbox = raw.get("bbox", raw.get("bounding_box"))
    return BridgeObject(
        id=object_id,
        label=label,
        position_3d=position_3d,
        bev_position=bev_position,
        confidence=_clip(float(raw.get("confidence", raw.get("score", 0.75)))),
        context_id=str(raw.get("context_id") or context_id),
        source=str(raw.get("source") or "imported"),
        source_view_ids=[str(value) for value in source_view_ids],
        bbox=bbox,
        status=str(raw.get("status") or "active"),
        freshness=_clip(float(raw.get("freshness", 1.0))),
        raw=dict(raw),
    )


def _inventory_entry_to_object(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        item = dict(value)
    else:
        item = {"position": value}
    item.setdefault("label", name)
    item.setdefault("id", f"nlmap_{str(name).lower().replace(' ', '_')}")
    return item


def _position_3d(value: Any) -> tuple[float, float, float]:
    if isinstance(value, dict):
        return (float(value["x"]), float(value["y"]), float(value["z"]))
    if len(value) < 3:
        raise ValueError(f"Expected 3D position, got {value}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _position_to_bev(position: tuple[float, float, float], horizontal_axes: HorizontalAxes) -> tuple[float, float]:
    if horizontal_axes == "xy":
        return (float(position[0]), float(position[1]))
    return (float(position[0]), float(position[2]))


def _points_to_bev(points: np.ndarray, horizontal_axes: HorizontalAxes) -> np.ndarray:
    if horizontal_axes == "xy":
        return points[:, [0, 1]]
    return points[:, [0, 2]]


def _points_array(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.zeros((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("Point cloud must be an Nx3 array or list.")
    return array[:, :3]


def _mock_room_pointcloud() -> np.ndarray:
    points = []
    xs = np.linspace(-3.0, 3.0, 61)
    zs = np.linspace(-2.5, 2.5, 51)
    for x in xs:
        points.append([x, 0.0, -2.5])
        points.append([x, 0.0, 2.5])
    for z in zs:
        points.append([-3.0, 0.0, z])
        points.append([3.0, 0.0, z])
    for x in np.linspace(-2.4, 2.4, 25):
        points.append([x, 0.0, 0.15])
    for z in np.linspace(-1.8, 1.8, 25):
        points.append([0.95, 0.0, z])
    return np.asarray(points, dtype=float)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
