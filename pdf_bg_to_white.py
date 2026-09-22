#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dark WeChat screenshot -> light theme (v9)."""

from __future__ import annotations
import argparse
import ast
import io
import math
import platform
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

TARGET_BG = np.array([245, 245, 245], dtype=np.float32)
PROFILE_DIVIDER = np.array([215, 215, 215], dtype=np.float32)
CARD_BG = np.array([255, 255, 255], dtype=np.float32)
BUBBLE_GREEN = np.array([149, 236, 105], dtype=np.float32)
INK = 18.0

AVATAR_DETECTION_CLASSES = frozenset(
    {"profile_avatar", "chat_avatar_left", "chat_avatar_right"}
)
FILE_BROWSE_DETECTION_CLASSES = frozenset({"file_item", "file_path_bar"})
CHAT_BUBBLE_CLASSES = frozenset({"chat_bubble_left", "chat_bubble_right"})
CHAT_VOICE_CLASSES = frozenset({"chat_voice_left", "chat_voice_right"})
# Only image messages are eligible for byte-for-byte preservation. Voice
# detections go through the neutral-bubble recolor path after visual recheck.
CHAT_MEDIA_CLASSES = frozenset(
    {
        "chat_image_left",
        "chat_image_right",
    }
)


@dataclass(frozen=True)
class Detection:
    class_id: int
    name: str
    confidence: float
    box: tuple[float, float, float, float]

class YoloDetector:
    """Run the exported YOLO detector directly through ONNX Runtime."""

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ):
        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"YOLO 模型不存在: {resolved_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("运行 YOLO 检测需要已安装 onnxruntime") from exc

        self.session = ort.InferenceSession(
            str(resolved_path),
            providers=["CPUExecutionProvider"],
        )
        input_meta = self.session.get_inputs()[0]
        input_shape = input_meta.shape
        if len(input_shape) != 4 or input_shape[1] not in (3, "3"):
            raise RuntimeError(f"不支持的 YOLO 输入形状: {input_shape}")

        model_height, model_width = input_shape[2:4]
        if isinstance(model_height, int) and isinstance(model_width, int):
            if (model_height, model_width) != (imgsz, imgsz):
                raise ValueError(
                    f"模型固定输入为 {model_width}x{model_height}，"
                    f"--imgsz={imgsz} 不匹配"
                )
            self.input_height = model_height
            self.input_width = model_width
        else:
            self.input_height = imgsz
            self.input_width = imgsz

        self.input_name = input_meta.name
        self.names = self._load_names()
        self.confidence = confidence
        self.iou = iou

    def _load_names(self) -> dict[int, str]:
        raw_names = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw_names:
            return {}
        try:
            names = ast.literal_eval(raw_names)
            if isinstance(names, dict):
                return {int(class_id): str(name) for class_id, name in names.items()}
            if isinstance(names, (list, tuple)):
                return {class_id: str(name) for class_id, name in enumerate(names)}
        except (SyntaxError, TypeError, ValueError):
            pass
        return {}

    @staticmethod
    def _letterbox(
        rgb_u8: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> tuple[np.ndarray, float, int, int]:
        original_height, original_width = rgb_u8.shape[:2]
        ratio = min(target_height / original_height, target_width / original_width)
        new_width = round(original_width * ratio)
        new_height = round(original_height * ratio)
        dw = target_width - new_width
        dh = target_height - new_height
        dw /= 2
        dh /= 2
        left = round(dw - 0.1)
        right = round(dw + 0.1)
        top = round(dh - 0.1)
        bottom = round(dh + 0.1)

        if (new_width, new_height) != (original_width, original_height):
            resized = cv2.resize(
                rgb_u8,
                (new_width, new_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = rgb_u8
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return tensor, ratio, left, top

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        # ponytail: keep a local NumPy NMS until the ONNX graph is exported with NMS included.
        order = scores.argsort()[::-1]
        keep: list[int] = []
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            intersection_left = np.maximum(boxes[current, 0], boxes[remaining, 0])
            intersection_top = np.maximum(boxes[current, 1], boxes[remaining, 1])
            intersection_right = np.minimum(boxes[current, 2], boxes[remaining, 2])
            intersection_bottom = np.minimum(boxes[current, 3], boxes[remaining, 3])
            intersection_width = np.maximum(0.0, intersection_right - intersection_left)
            intersection_height = np.maximum(0.0, intersection_bottom - intersection_top)
            intersection = intersection_width * intersection_height
            union = areas[current] + areas[remaining] - intersection
            overlap = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[overlap <= iou_threshold]
        return np.asarray(keep, dtype=np.int64)

    def detect(self, rgb_u8: np.ndarray) -> list[Detection]:
        input_tensor, ratio, pad_x, pad_y = self._letterbox(
            rgb_u8,
            self.input_height,
            self.input_width,
        )
        prediction = np.asarray(
            self.session.run(None, {self.input_name: input_tensor})[0],
            dtype=np.float32,
        )
        if prediction.ndim == 3:
            if prediction.shape[0] != 1:
                raise RuntimeError(f"不支持的 YOLO batch 输出: {prediction.shape}")
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise RuntimeError(f"不支持的 YOLO 输出形状: {prediction.shape}")
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            raise RuntimeError(f"不支持的 YOLO 输出形状: {prediction.shape}")

        boxes_xywh = prediction[:, :4]
        class_scores = prediction[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        candidate_mask = scores > self.confidence
        if not np.any(candidate_mask):
            return []

        boxes_xywh = boxes_xywh[candidate_mask]
        class_ids = class_ids[candidate_mask]
        scores = scores[candidate_mask]
        boxes = np.empty_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, rgb_u8.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, rgb_u8.shape[0])

        keep: list[int] = []
        for class_id in np.unique(class_ids):
            class_indices = np.flatnonzero(class_ids == class_id)
            class_keep = self._nms(boxes[class_indices], scores[class_indices], self.iou)
            keep.extend(class_indices[class_keep].tolist())
        if not keep:
            return []
        keep_array = np.asarray(keep, dtype=np.int64)
        keep_array = keep_array[np.argsort(scores[keep_array])[::-1]][:300]

        detections: list[Detection] = []
        for index in keep_array:
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    class_id=class_id,
                    name=self.names.get(class_id, str(class_id)),
                    confidence=float(scores[index]),
                    box=tuple(float(value) for value in boxes[index]),
                )
            )
        return detections


def _detection_mask(
    shape: tuple[int, int],
    detections: list[Detection] | None,
    class_names: frozenset[str],
    padding: int = 0,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not detections:
        return mask
    height, width = shape
    for detection in detections:
        if detection.name not in class_names:
            continue
        x0, y0, x1, y1 = detection.box
        left = max(0, math.floor(x0) - padding)
        top = max(0, math.floor(y0) - padding)
        right = min(width, math.ceil(x1) + padding)
        bottom = min(height, math.ceil(y1) + padding)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def _classify_detections(detections: list[Detection] | None) -> str | None:
    if not detections:
        return None
    names = {detection.name for detection in detections}

    # Require two independent profile cues. The avatar label itself is not
    # trusted as a profile marker because the model can confuse it with a chat
    # avatar; profile_moments supplies the page-level context.
    has_avatar = bool(names & AVATAR_DETECTION_CLASSES)
    if has_avatar and "profile_moments" in names:
        return "profile"

    if "normal_photo" in names:
        return "document"
    if names & FILE_BROWSE_DETECTION_CLASSES:
        return "phone_document"

    # A single avatar, a single input bar, or chat content without the page
    # chrome is ambiguous. Require both object types before calling it chat.
    if has_avatar and "chat_input_bar" in names:
        return "chat"
    return None


def estimate_bg_level(lum: np.ndarray) -> float:
    h, w = lum.shape
    samples = np.concatenate(
        [
            lum[int(0.12 * h) : int(0.88 * h), : max(8, w // 40)].ravel(),
            lum[int(0.12 * h) : int(0.88 * h), -max(8, w // 40) :].ravel(),
            lum[int(0.34 * h) : int(0.46 * h), int(0.34 * w) : int(0.66 * w)].ravel(),
        ]
    )
    return float(np.median(samples))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255).copy()
    flood = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    cv2.floodFill(m, flood, (0, 0), 128)
    holes = m == 0
    out = mask.copy()
    out[holes] = True
    return out


def protect_side_avatars(rgb: np.ndarray, green: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    color = (s > 28) & (v > 35) & (v < 250) & ~green
    out = np.zeros((h, w), dtype=bool)
    y_min = int(0.11 * h)
    y_max = int(0.88 * h)  # never treat footer/input chrome as avatars
    for x0, x1 in ((0, int(0.22 * w)), (int(0.78 * w), w)):
        band = np.zeros((h, w), dtype=np.uint8)
        band[:, x0:x1][color[:, x0:x1]] = 255
        n, _, st, _ = cv2.connectedComponentsWithStats(band)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if y < y_min or y > y_max:
                continue
            if area < 900 or area > 0.05 * h * w:
                continue
            aspect = bw / max(bh, 1)
            if aspect < 0.55 or aspect > 1.8:
                continue
            if abs(bw - bh) > 0.45 * max(bw, bh):
                continue
            if bw < 48 or bh < 48 or bw > 170 or bh > 170:
                continue
            out[y : y + bh, x : x + bw] = True
    return out


def protect_green_bubbles(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    seed = ((hh >= 35) & (hh <= 95) & (ss >= 40) & (vv >= 70)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, st, _ = cv2.connectedComponentsWithStats(seed)
    out = np.zeros((h, w), dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 500 or bw < 35 or bh < 22:
            continue
        if bw > 0.92 * w and bh < 0.08 * h:
            continue
        blob = labels == i
        out |= fill_holes(blob)
        out |= cv2.dilate(blob.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return out


def protect_white_tables(rgb: np.ndarray, lum: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    bright = ((lum > 170) & (rgb.max(2) - rgb.min(2) < 35)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, st, _ = cv2.connectedComponentsWithStats(bright)
    out = np.zeros((h, w), dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 2500 or bw < 0.28 * w or bh < 40:
            continue
        if y < 0.05 * h and bh < 0.08 * h:
            continue
        if y > 0.9 * h:
            continue
        blob = labels == i
        if area / float(bw * bh) < 0.35:
            continue
        out |= fill_holes(blob)
        out |= cv2.dilate(blob.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return out


def protect_gray_ui(lum: np.ndarray, chroma: np.ndarray, bg_level: float, green: np.ndarray, avatar: np.ndarray) -> np.ndarray:
    h, w = lum.shape
    lo = max(18.0, bg_level + 4.0)
    hi = 105.0
    cand = (lum > lo) & (lum < hi) & (chroma < 30) & ~green & ~avatar
    cand[cand.mean(axis=1) > 0.72] = False
    # Status + title bar must never become gray_ui (elevated header bg on
    # pages like img_0003 would soft-remap 「太子湾…」 and look blurry).
    # First chat bubbles typically start ~y>=0.11h (e.g. 嗯嗯好的 ~288).
    cand[: int(h * 0.108)] = False
    cand[int(h * 0.90) :] = False

    u8 = cand.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, st, _ = cv2.connectedComponentsWithStats(u8)
    out = np.zeros((h, w), dtype=bool)
    y_chat_min = int(0.110 * h)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if y < y_chat_min:
            continue
        if area < 600 or bw < 70 or bh < 28:
            continue
        if bw > 0.82 * w or bh > 0.55 * h:
            continue
        if bw / max(bh, 1) > 14 and bh < 55:
            continue
        # Reject timestamp / glyph-sized blobs (not chat bubbles or file cards)
        if area < 3500 and bh < 55 and bw < 220:
            continue
        blob = labels == i
        if area / float(max(bw * bh, 1)) < 0.18:
            continue
        out |= fill_holes(blob)
        out |= cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return out


def build_protect(rgb: np.ndarray, lum: np.ndarray, chroma: np.ndarray, bg_level: float):
    green = protect_green_bubbles(rgb)
    avatar = protect_side_avatars(rgb, green)
    tables = protect_white_tables(rgb, lum)
    gray_ui = protect_gray_ui(lum, chroma, bg_level, green, avatar)
    hard = green | avatar | tables | gray_ui
    soft = cv2.dilate(hard.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~hard
    return hard, soft, green, avatar, tables, gray_ui


def convert_gray_ui_cards(
    out: np.ndarray,
    gray_ui: np.ndarray,
    green: np.ndarray,
    lum: np.ndarray,
    chroma: np.ndarray,
    rgb: np.ndarray | None = None,
) -> None:
    card = gray_ui & ~green
    if not np.any(card):
        return
    accent = np.zeros(lum.shape, dtype=bool)
    if rgb is not None:
        hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        accent = card & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 60)
        out[accent] = rgb[accent]
    body = card & ~accent & (lum < 75) & (chroma < 28)
    out[body] = CARD_BG
    ink = card & ~accent & (lum >= 55) & (chroma < 55)
    if np.any(ink):
        # Steeper curve: solid cores, less gray AA mush (reads as 模糊)
        t = np.clip((lum[ink] - 70.0) / 150.0, 0.0, 1.0).astype(np.float32)
        t = np.power(t, 0.55)
        for c in range(3):
            out[:, :, c][ink] = CARD_BG[c] * (1.0 - t) + INK * t
        # Bright glyph cores -> solid ink
        core = card & ~accent & (lum >= 150) & (chroma < 55)
        out[core] = INK


def _alpha_paint(out: np.ndarray, alpha: np.ndarray, protect: np.ndarray) -> None:
    alpha = alpha.copy()
    alpha[protect] = 0.0
    m = alpha > 0.05
    if not np.any(m):
        return
    am = alpha[m]
    for c in range(3):
        out[:, :, c][m] = TARGET_BG[c] * (1.0 - am) + INK * am


def _cc_filter(u8: np.ndarray, min_area: int, max_area: int, w: int, max_bw_frac: float = 0.55) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(u8)
    out = np.zeros_like(u8)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < min_area or area > max_area:
            continue
        if bw > max_bw_frac * w and bh > 40:
            continue
        fill = area / float(max(bw * bh, 1))
        if fill > 0.88 and bw > 0.35 * w:
            continue
        out[lab == i] = 255
    return out


def _fill_tiny_holes(u8: np.ndarray, max_hole: int = 48) -> np.ndarray:
    clean = u8.copy()
    inv = cv2.bitwise_not(clean)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv)
    for i in range(1, n2):
        if 1 <= st2[i, 4] <= max_hole:
            clean[lab2 == i] = 255
    return clean


def paint_header_chrome(
    out: np.ndarray,
    lum: np.ndarray,
    chroma: np.ndarray,
    protect: np.ndarray,
    bg_level: float,
) -> None:
    """Solid header chrome — slightly thinner than source, avoid stroke adhesion."""
    h, w = lum.shape
    editable = ~protect
    # Stop above first chat bubbles (title ends ~y=255 on 1179x2556)
    y1 = max(1, int(0.102 * h))
    top = np.zeros((h, w), dtype=bool)
    top[:y1] = True

    header_band = lum[8 : max(9, int(0.055 * h)), int(0.08 * w) : int(0.92 * w)]
    header_bg = float(np.median(header_band)) if header_band.size else float(bg_level)
    denom = max(200.0 - header_bg, 80.0)
    contrast = np.clip((lum - header_bg) / denom, 0.0, 1.0).astype(np.float32)
    contrast[~(top & editable & (chroma < 55))] = 0.0

    y_split = int(0.055 * h)
    status_band = np.zeros((h, w), dtype=bool)
    title_band = np.zeros((h, w), dtype=bool)
    status_band[:y_split] = True
    title_band[y_split:y1] = True

    def _clean_seed(seed: np.ndarray, blur: bool) -> np.ndarray:
        if not np.any(seed):
            return seed
        m = seed
        if blur:
            m = cv2.medianBlur(m, 3)
        # light open only — avoid closing gaps inside 商/湾
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return _cc_filter(m, min_area=3, max_area=int(0.02 * h * w), w=w)

    def _peel(mask: np.ndarray, min_dist: float) -> np.ndarray:
        """Peel outer stroke crust via distance transform (opens adhered gaps)."""
        if not np.any(mask):
            return mask
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        peeled = ((dist >= min_dist) & (mask > 0)).astype(np.uint8) * 255
        return _cc_filter(peeled, min_area=2, max_area=int(0.02 * h * w), w=w)

    # Status/time: peel a little so clock/icons are not heavy
    status_mask = ((contrast >= 0.38) & status_band & top & editable).astype(np.uint8) * 255
    status_mask[:, int(0.32 * w) : int(0.68 * w)] = 0
    status_mask = _clean_seed(status_mask, blur=True)
    status_mask = _peel(status_mask, min_dist=1.05)

    # Title: stronger peel to break 商/湾/苹 internal adhesion
    title_mask = ((contrast >= 0.50) & title_band & top & editable).astype(np.uint8) * 255
    title_mask = _clean_seed(title_mask, blur=False)
    title_mask = _peel(title_mask, min_dist=1.45)

    shaped = np.maximum(status_mask, title_mask)
    ink_mask = shaped > 0
    out[ink_mask] = np.array([INK, INK, INK], dtype=np.float32)

    wipe = top & editable & ~ink_mask
    wipe &= (chroma < 50)
    wipe &= (lum > header_bg - 8.0) | (out.mean(axis=2) < 230)
    out[wipe] = TARGET_BG



def paint_timestamps(
    out: np.ndarray,
    lum: np.ndarray,
    chroma: np.ndarray,
    protect: np.ndarray,
    bg_level: float,
) -> np.ndarray:
    """Solid timestamp glyphs; wipe dust so 「2025年…」 has no pepper dots."""
    h, w = lum.shape
    editable = ~protect
    y1 = max(1, int(0.12 * h))
    top = np.zeros((h, w), dtype=bool)
    top[:y1] = True

    mid = editable & ~top & (chroma < 40) & (lum > 55) & (lum < 200)
    u8 = mid.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((2, 41), np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(u8)

    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bh < 12 or bh > 52 or bw < 60 or bw > 0.75 * w:
            continue
        if area / float(max(bw * bh, 1)) < 0.05:
            continue
        cx = x + bw * 0.5
        if abs(cx - 0.5 * w) > 0.30 * w:
            continue
        if y < 0.12 * h or y > 0.88 * h:
            continue
        if bw / max(bh, 1) < 2.2:
            continue
        boxes.append((x, y, bw, bh))

    if not boxes:
        return np.zeros((h, w), dtype=bool)

    denom = max(200.0 - float(bg_level), 80.0)
    contrast = np.clip((lum - bg_level) / denom, 0.0, 1.0).astype(np.float32)

    ink_all = np.zeros((h, w), dtype=np.uint8)
    wipe = np.zeros((h, w), dtype=bool)

    for x, y, bw, bh in boxes:
        pad_x, pad_y = 12, 10
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1b = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        roi = np.zeros((h, w), dtype=bool)
        roi[y0:y1b, x0:x1] = True
        wipe |= roi & editable

        # Core glyphs: thr high enough to skip AA pepper, low enough to keep strokes
        seed = ((contrast >= 0.32) & roi & editable & (chroma < 45)).astype(np.uint8) * 255
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        seed = _cc_filter(seed, min_area=5, max_area=int(0.02 * h * w), w=w, max_bw_frac=0.80)
        if not np.any(seed):
            seed = ((contrast >= 0.26) & roi & editable & (chroma < 45)).astype(np.uint8) * 255
            seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            seed = _cc_filter(seed, min_area=4, max_area=int(0.02 * h * w), w=w, max_bw_frac=0.80)
        ink_all = np.maximum(ink_all, seed)

    ink = ink_all > 0
    if not np.any(ink):
        return np.zeros((h, w), dtype=bool)

    out[wipe & ~ink] = TARGET_BG
    out[ink] = np.array([INK, INK, INK], dtype=np.float32)

    # Anything dark outside a 1px halo of the ink core is pepper — wipe it
    halo = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    exterior = wipe & ~halo & (out.mean(axis=2) < 140)
    out[exterior] = TARGET_BG
    # Tiny isolated dots still touching the halo
    dark = (wipe & (out.mean(axis=2) < 100)).astype(np.uint8) * 255
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(dark)
    for i in range(1, n2):
        if 1 <= st2[i, 4] <= 3:
            out[lab2 == i] = TARGET_BG
    return wipe


def remap_chrome(out: np.ndarray, lum: np.ndarray, chroma: np.ndarray, protect: np.ndarray, bg_level: float) -> None:
    """Soft alpha remap for body chrome; solid paint for header + timestamps."""
    h, w = lum.shape
    editable = ~protect

    paint_header_chrome(out, lum, chroma, protect, bg_level)
    ts_zone = paint_timestamps(out, lum, chroma, protect, bg_level)

    y1 = max(1, int(0.12 * h))
    top = np.zeros((h, w), dtype=bool)
    top[:y1] = True

    # Body chrome: skip timestamp boxes (already solid + wiped)
    rest = editable & ~top & ~ts_zone & (chroma < 45) & (lum > bg_level + 8.0)
    alpha = np.zeros((h, w), dtype=np.float32)
    delta = np.clip(lum - bg_level, 0.0, None)
    a = np.clip(delta / 55.0, 0.0, 1.0)
    a = np.power(a, 0.75)
    alpha[rest] = a[rest]
    dust = (alpha > 0.18).astype(np.uint8) * 255
    dust = cv2.morphologyEx(dust, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha[dust == 0] = 0.0
    # No Gaussian blur here — blur spreads pepper dots around glyphs
    alpha = np.clip(alpha * 1.15, 0.0, 1.0)
    _alpha_paint(out, alpha, protect)


def cleanup_green_shell(out: np.ndarray, rgb: np.ndarray, green: np.ndarray, avatar: np.ndarray, bg: np.ndarray) -> None:
    if not np.any(green):
        return
    dil = cv2.dilate(green.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    shell = dil & ~green & ~avatar
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    greenish = shell & (((hh >= 35) & (hh <= 95) & (ss >= 15)) | (ss < 45))
    if np.any(green & (ss >= 40)):
        mean_g = rgb[green & (ss >= 40)].mean(axis=0)
    else:
        mean_g = BUBBLE_GREEN
    out[greenish] = mean_g
    # Exterior dark fringe -> bg (kills black outline)
    ring = cv2.dilate(dil.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~dil & ~avatar
    dark_ring = ring & (out.mean(axis=2) < 130) & ((out.max(2) - out.min(2)) < 45)
    out[dark_ring] = bg
    # Dilated green mask often swallows dark chat-bg; those restore as black
    # stripes on the bubble rim. Only clean the rim — never the bubble interior
    # (black text on green is also low-sat/low-v).
    er = cv2.erode(green.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    rim = green & ~avatar & ~er
    fake = rim & (ss < 50) & (vv < 120)
    out[fake] = bg
    rim_dark = rim & (out.mean(axis=2) < 120) & ((out.max(2) - out.min(2)) < 55)
    out[rim_dark] = bg
    # Top edge of each green column: wipe 1–2px dark crumbs above/on the rim
    has = green.any(axis=0)
    first = np.argmax(green, axis=0)
    yy = np.arange(green.shape[0], dtype=np.int32)[:, None]
    top_rows = has[None, :] & (yy >= first[None, :] - 1) & (yy <= first[None, :] + 2)
    top_dark = top_rows & ~avatar & (ss < 55) & (out.mean(axis=2) < 110) & ((out.max(2) - out.min(2)) < 55)
    out[top_dark] = bg


def cleanup_gray_ui_fringe(out: np.ndarray, gray_ui: np.ndarray, avatar: np.ndarray, green: np.ndarray, bg: np.ndarray) -> None:
    if not np.any(gray_ui):
        return
    dil = cv2.dilate(gray_ui.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ring = dil & ~gray_ui & ~avatar & ~green
    dark = ring & (out.mean(axis=2) < 120) & ((out.max(2) - out.min(2)) < 40)
    out[dark] = bg
    # Dark AA just inside card edge -> white (removes black bubble stroke)
    er = cv2.erode(gray_ui.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    inner = gray_ui & ~er & ~avatar
    edge_dark = inner & (out.mean(axis=2) < 90) & ((out.max(2) - out.min(2)) < 40)
    out[edge_dark] = CARD_BG


def fix_avatar_rounded_corners(out: np.ndarray, rgb: np.ndarray, avatar: np.ndarray, bg_level: float, bg: np.ndarray) -> None:
    """Remove dark square frame + rounded outline around avatars."""
    if not np.any(avatar):
        return
    u8 = avatar.astype(np.uint8)
    n, _labels, st, _ = cv2.connectedComponentsWithStats(u8)
    for i in range(1, n):
        x, y, bw, bh, _area = st[i]
        if bw < 8 or bh < 8:
            continue
        # Synthetic rounded-rect: anything in bbox outside it is chat-bg frame
        rr = np.zeros((bh, bw), dtype=np.uint8)
        rad = max(6, int(0.18 * min(bw, bh)))
        cv2.rectangle(rr, (rad, 0), (bw - rad - 1, bh - 1), 255, -1)
        cv2.rectangle(rr, (0, rad), (bw - 1, bh - rad - 1), 255, -1)
        for cx, cy in ((rad, rad), (bw - rad - 1, rad), (rad, bh - rad - 1), (bw - rad - 1, bh - rad - 1)):
            cv2.circle(rr, (cx, cy), rad, 255, -1)
        outside = rr == 0
        out[y : y + bh, x : x + bw][outside] = bg
        # Dark ring on the rounded perimeter
        er = cv2.erode(rr, np.ones((3, 3), np.uint8))
        peri = (rr > 0) & (er == 0)
        roi = out[y : y + bh, x : x + bw]
        r_lum = roi.mean(axis=2)
        r_ch = roi.max(2) - roi.min(2)
        kill = peri & (r_lum < 100) & (r_ch < 60)
        roi[kill] = bg
        out[y : y + bh, x : x + bw] = roi
        # Also plain near-bg corners from original
        src_roi = rgb[y : y + bh, x : x + bw].astype(np.float32)
        s_lum = 0.299 * src_roi[:, :, 0] + 0.587 * src_roi[:, :, 1] + 0.114 * src_roi[:, :, 2]
        s_ch = src_roi.max(2) - src_roi.min(2)
        corner = (s_lum < bg_level + 22.0) & (s_ch < 25.0)
        out[y : y + bh, x : x + bw][corner] = bg
    dil = cv2.dilate(u8, np.ones((5, 5), np.uint8)).astype(bool)
    ring = dil & ~avatar
    edge_lum = out.mean(axis=2)
    edge_ch = out.max(2) - out.min(2)
    out[ring & (edge_lum < 130) & (edge_ch < 40)] = bg


def _add_rounded_rect(mask: np.ndarray, x: int, y: int, width: int, height: int) -> None:
    """Add a clipped rounded rectangle to a boolean mask."""
    h, w = mask.shape
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + width)), min(h, int(y + height))
    if x1 <= x0 or y1 <= y0:
        return
    roi_w, roi_h = x1 - x0, y1 - y0
    radius = max(4, int(0.10 * min(roi_w, roi_h)))
    rr = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.rectangle(rr, (radius, 0), (roi_w - radius - 1, roi_h - 1), 255, -1)
    cv2.rectangle(rr, (0, radius), (roi_w - 1, roi_h - radius - 1), 255, -1)
    for cx, cy in (
        (radius, radius),
        (roi_w - radius - 1, radius),
        (radius, roi_h - radius - 1),
        (roi_w - radius - 1, roi_h - radius - 1),
    ):
        cv2.circle(rr, (cx, cy), radius, 255, -1)
    mask[y0:y1, x0:x1] |= rr.astype(bool)


def _classify_screenshot(rgb_u8: np.ndarray, detections: list[Detection] | None = None) -> str:
    """Classify the layouts that need different levels of recoloring."""
    detected_kind = _classify_detections(detections)
    if detected_kind == "chat" and _find_profile_avatar_box(rgb_u8) is not None:
        # A large avatar in the profile identity block is a safer signal than
        # a spurious chat-avatar/input-bar pair on the same page.
        return "profile"
    if detected_kind is not None:
        return detected_kind

    h, w = rgb_u8.shape[:2]
    aspect = h / max(w, 1)
    if not 1.80 <= aspect <= 2.40:
        return "document"

    rgb = rgb_u8.astype(np.float32)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    top = lum[: max(1, int(0.08 * h))]
    if float((top < 90.0).mean()) < 0.72:
        return "document"

    body = lum[int(0.11 * h) : int(0.88 * h)]
    body_chroma = chroma[int(0.11 * h) : int(0.88 * h)]
    if float(np.median(body)) > 130.0 or float((body > 150.0).mean()) > 0.35:
        return "phone_document"

    # A profile page has one large square avatar in the upper-left identity
    # block.  Chat avatars are deliberately much smaller, so this is a more
    # stable signal than global chroma (dark chat pages may contain no green
    # outgoing bubble at all).
    if _find_profile_avatar_box(rgb_u8) is not None:
        return "profile"

    # Standard WeChat incoming bubbles are neutral gray and therefore do not
    # show up reliably in a percentile-chroma test.  Large green bubbles are
    # an additional strong signal, but they are not required for a chat page.
    bg_level = estimate_bg_level(lum)
    neutral = (chroma < 24.0) & (lum >= bg_level + 7.0) & (lum < 115.0)
    neutral[: int(0.105 * h)] = False
    neutral[int(0.90 * h) :] = False
    neutral_u8 = cv2.morphologyEx(
        neutral.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(neutral_u8)
    gray_bubbles = 0
    for x, y, bw, bh, area in stats[1:]:
        fill = area / float(max(bw * bh, 1))
        if (
            area >= 3000
            and bw >= 100
            and bh >= 28
            and x >= 0.14 * w
            and x + bw <= 0.94 * w
            and fill >= 0.25
        ):
            gray_bubbles += 1
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    green = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 70)
    )
    green[: int(0.105 * h)] = False
    green[int(0.90 * h) :] = False
    green_u8 = cv2.morphologyEx(
        green.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(green_u8)
    green_bubbles = sum(
        int(area >= 5000 and bw >= 80 and bh >= 35)
        for _x, _y, bw, bh, area in stats[1:]
    )
    if gray_bubbles or green_bubbles or float(np.percentile(body_chroma, 90)) > 15.0:
        return "chat"
    return "document"


def _find_profile_avatar_box(rgb_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the large identity avatar used by a WeChat profile page."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    bg_level = estimate_bg_level(lum)

    # Restrict the search to the identity panel. This prevents a chat's
    # 123px side avatar or a document body from becoming a profile avatar.
    visual = ((np.abs(lum - bg_level) > 18.0) | (chroma > 15.0)).astype(np.uint8) * 255
    roi = np.zeros_like(visual)
    roi[int(0.10 * h) : int(0.32 * h), : int(0.34 * w)] = 255
    visual &= roi
    visual = cv2.morphologyEx(visual, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(visual)
    candidates: list[tuple[float, int, int, int, int]] = []
    for x, y, bw, bh, area in stats[1:]:
        if x >= int(0.15 * w):
            continue
        fill = area / float(max(bw * bh, 1))
        ratio = bw / max(bh, 1)
        if not 150 <= bw <= 240 or not 150 <= bh <= 240:
            continue
        if not 0.78 <= ratio <= 1.28 or fill < 0.72:
            continue
        score = float(area) + 5000.0 * fill - 20.0 * abs(ratio - 1.0) * min(bw, bh)
        candidates.append((score, int(x), int(y), int(bw), int(bh)))
    if not candidates:
        return None
    _score, x, y, bw, bh = max(candidates)
    return x, y, bw, bh


def _profile_avatar_mask(rgb_or_shape: np.ndarray | tuple[int, int]) -> np.ndarray:
    if isinstance(rgb_or_shape, tuple):
        h, w = rgb_or_shape
        box = None
    else:
        h, w = rgb_or_shape.shape[:2]
        box = _find_profile_avatar_box(rgb_or_shape)
    mask = np.zeros((h, w), dtype=bool)
    if box is None:
        # Only a last-resort fallback is needed for old profile screenshots;
        # normal profile pages are required to pass the component detector.
        side = max(32, int(round(0.164 * w)))
        x = int(round((0.041 if w >= 1160 else 0.064) * w))
        y = int(round(0.140 * h))
    else:
        x, y, bw, bh = box
        side = min(bw, bh)
    _add_rounded_rect(mask, x, y, side, side)
    return mask


def _map_dark_theme_ink(lum: np.ndarray) -> np.ndarray:
    """Map light-mode foreground from a dark-theme luminance value."""
    return np.clip(245.0 - 1.70 * (lum - 35.0), 25.0, 245.0)


def _profile_separator_mask(lum: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """Find one-pixel neutral separators without treating the dark canvas as ink."""
    h, _w = lum.shape
    row_lum = np.median(lum, axis=1)
    row_chroma = np.percentile(chroma, 90, axis=1)
    neighbours = (np.roll(row_lum, 1) + np.roll(row_lum, -1)) * 0.5
    rows = (
        (row_lum >= 35.0)
        & (row_lum <= 90.0)
        & (row_chroma < 8.0)
        & ((row_lum - neighbours) > 6.0)
    )
    rows[: int(0.20 * h)] = False
    rows[-1] = False
    return np.broadcast_to(rows[:, None], lum.shape)


def _convert_profile_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert a profile page while copying the profile avatar byte-for-byte."""
    rgb = rgb_u8.astype(np.float32)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    avatar = _profile_avatar_mask(rgb_u8)

    out = np.empty_like(rgb)
    out[:] = TARGET_BG
    neutral = chroma < 45.0
    ink = neutral & (lum >= 60.0) & ~avatar
    mapped = _map_dark_theme_ink(lum)
    for c in range(3):
        out[:, :, c][ink] = mapped[ink]

    # Profile pages have colored status/badge controls but no colored chat
    # background, so keeping chromatic pixels is safer than recoloring them.
    color = ~neutral & ~avatar
    out[color] = rgb[color]
    out[avatar] = rgb[avatar]
    separators = _profile_separator_mask(lum, chroma) & ~avatar
    out[separators] = PROFILE_DIVIDER
    return np.clip(out, 0, 255).astype(np.uint8)


def _find_phone_header_end(lum: np.ndarray) -> int:
    h = lum.shape[0]
    start = int(0.065 * h)
    stop = min(h, int(0.17 * h))
    for y in range(start, stop):
        row = lum[y]
        if float(np.median(row)) > 150.0 and float((row > 150.0).mean()) > 0.35:
            return y
    return max(1, int(0.09 * h))


def _convert_phone_document_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """Only recolor the dark phone/app chrome above a light document."""
    rgb = rgb_u8.astype(np.float32)
    h, _w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    header_end = _find_phone_header_end(lum)

    out = rgb.copy()
    out[:header_end] = TARGET_BG
    neutral = (chroma[:header_end] < 65.0) & (lum[:header_end] >= 70.0)
    mapped = _map_dark_theme_ink(lum[:header_end])
    for c in range(3):
        out[:header_end, :, c][neutral] = mapped[neutral]
    color = (chroma[:header_end] >= 45.0) & (lum[:header_end] > 60.0)
    out[:header_end][color] = rgb[:header_end][color]
    return np.clip(out, 0, 255).astype(np.uint8)


def _append_avatar_box(
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    h, w = mask.shape
    x, y, width, height = int(x), int(y), int(width), int(height)
    if width < 70 or height < 70 or x < 0 or y < 0 or x + width > w or y + height > h:
        return
    for bx, by, bw, bh in boxes:
        overlap_x = max(0, min(x + width, bx + bw) - max(x, bx))
        overlap_y = max(0, min(y + height, by + bh) - max(y, by))
        if overlap_x > 0.55 * min(width, bw) and overlap_y > 0.35 * min(height, bh):
            return
    boxes.append((x, y, width, height))
    _add_rounded_rect(mask, x, y, width, height)


def _find_low_bg_chat_avatars(
    rgb_u8: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Find the fixed-size side tiles used by the newer dark WeChat layout."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bg_level = estimate_bg_level(lum)
    side = max(80, int(round(0.114 * w)))
    avatar = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for x in (int(round(0.028 * w)), int(round(0.858 * w))):
        if x < 0 or x + side > w:
            continue
        # Chat background is nearly uniform at this x. A tile has a strong
        # row-wise departure even when the portrait itself is dark.
        activity = np.mean(np.abs(lum[:, x : x + side] - bg_level), axis=1)
        good = activity > max(4.0, 0.10 * max(float(activity.max()), 1.0))
        good[: int(0.105 * h)] = False
        good[int(0.90 * h) :] = False
        good = cv2.morphologyEx(
            good.astype(np.uint8)[:, None],
            cv2.MORPH_CLOSE,
            np.ones((5, 1), np.uint8),
        )[:, 0].astype(bool)
        changes = np.diff(np.r_[False, good, False].astype(np.int8))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        for y0, y1 in zip(starts, ends):
            height = y1 - y0
            if height < int(0.65 * side) or height > int(1.45 * side):
                continue
            if float(activity[y0:y1].mean()) < 18.0:
                continue
            _append_avatar_box(avatar, boxes, x, y0, side, side)
    return avatar, boxes


def _find_chat_avatars(
    rgb_u8: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Detect light and dark avatar tiles without classifying their contents."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    if estimate_bg_level(lum) < 35.0:
        return _find_low_bg_chat_avatars(rgb_u8)
    visual = (lum > 120.0) | ((chroma > 45.0) & (lum > 60.0))
    avatar = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for side in ("left", "right"):
        if side == "left":
            x0, x1 = 0, int(0.22 * w)
        else:
            x0, x1 = int(0.78 * w), w
        sub = visual[:, x0:x1]
        row_score = sub.mean(axis=1)
        good = row_score > 0.40
        good = cv2.morphologyEx(
            good.astype(np.uint8)[:, None],
            cv2.MORPH_CLOSE,
            np.ones((17, 1), np.uint8),
        )[:, 0].astype(bool)
        changes = np.diff(np.r_[False, good, False].astype(np.int8))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        for y0, y1 in zip(starts, ends):
            if not 80 <= y1 - y0 <= 240:
                continue
            y0, y1 = max(0, y0), min(h, y1)
            col_score = sub[y0:y1].mean(axis=0)
            col_good = col_score > 0.35
            changes_x = np.diff(np.r_[False, col_good, False].astype(np.int8))
            x_starts, x_ends = np.where(changes_x == 1)[0], np.where(changes_x == -1)[0]
            for xa, xb in zip(x_starts, x_ends):
                width, height = xb - xa, y1 - y0
                if not 75 <= width <= 230:
                    continue
                if not 0.55 <= width / max(height, 1) <= 1.55:
                    continue
                _append_avatar_box(avatar, boxes, xa + x0, y0, width, height)

    # A dark portrait can be too dark for the bright-square pass. Look for a
    # high-variance square in the same left avatar band, but only add it when
    # it is clearly more textured than the surrounding chat background.
    avatar_side = max(80, int(round(0.105 * w)))
    left_x = int(round(0.031 * w))
    sub = visual[:, : int(0.22 * w)]
    row_score = sub.mean(axis=1)
    weak = row_score > 0.15
    weak = cv2.morphologyEx(
        weak.astype(np.uint8)[:, None], cv2.MORPH_CLOSE, np.ones((17, 1), np.uint8)
    )[:, 0].astype(bool)
    changes = np.diff(np.r_[False, weak, False].astype(np.int8))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    for y0, y1 in zip(starts, ends):
        if not 80 <= y1 - y0 <= 180:
            continue
        scan_start = max(int(0.10 * h), y0 - 8)
        scan_stop = min(int(0.90 * h) - avatar_side, y1)
        if scan_stop <= scan_start:
            continue
        best: tuple[float, int, float] | None = None
        context = 8
        for candidate_y in range(scan_start, scan_stop + 1, 4):
            win_lum = lum[candidate_y : candidate_y + avatar_side, left_x : left_x + avatar_side]
            win_chroma = chroma[candidate_y : candidate_y + avatar_side, left_x : left_x + avatar_side]
            bright_fraction = float((win_lum > 100.0).mean())
            if bright_fraction <= 0.20:
                continue
            top_context = lum[
                max(0, candidate_y - context) : candidate_y,
                left_x : left_x + avatar_side,
            ]
            bottom_context = lum[
                candidate_y + avatar_side : candidate_y + avatar_side + context,
                left_x : left_x + avatar_side,
            ]
            boundary = abs(float(win_lum[:context].mean()) - float(top_context.mean()))
            boundary += abs(float(win_lum[-context:].mean()) - float(bottom_context.mean()))
            score = float(win_lum.std()) + 30.0 * bright_fraction
            score += 0.15 * float(np.percentile(win_chroma, 90))
            score += 1.5 * boundary
            if best is None or score > best[0]:
                best = (score, candidate_y, bright_fraction)
        if best is None:
            continue
        _score, candidate_y, bright_fraction = best
        win_lum = lum[candidate_y : candidate_y + avatar_side, left_x : left_x + avatar_side]
        if win_lum.std() > 35.0:
            _append_avatar_box(avatar, boxes, left_x, candidate_y, avatar_side, avatar_side)

    return avatar, boxes


def _find_chat_green_bubbles(rgb_u8: np.ndarray) -> np.ndarray:
    """Return complete outgoing green bubbles, including their dark text."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    seed = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 70)
    )
    seed[: int(0.105 * h)] = False
    seed[int(0.90 * h) :] = False
    u8 = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8)
    out = np.zeros((h, w), dtype=bool)
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if area < 5000 or bw < 80 or bh < 35:
            continue
        blob = labels == i
        out |= fill_holes(blob)
        out |= cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return out


def _find_chat_voice_bubbles(
    rgb_u8: np.ndarray,
    detections: list[Detection] | None,
) -> np.ndarray:
    """Return YOLO voice boxes that pass a lightweight waveform recheck."""
    # ponytail: keep this OpenCV shape check local until the detector is retrained
    # with enough hard-negative text bubbles to separate voice and text classes.
    h, w = rgb_u8.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    if not detections:
        return out

    rgb = rgb_u8.astype(np.float32)
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)

    for detection in detections:
        if detection.name not in CHAT_VOICE_CLASSES:
            continue
        x0, y0, x1, y1 = detection.box
        left = max(0, int(math.floor(x0)))
        top = max(int(0.105 * h), int(math.floor(y0)))
        right = min(w, int(math.ceil(x1)))
        bottom = min(int(0.90 * h), int(math.ceil(y1)))
        if right - left < 70 or bottom - top < 55:
            continue

        roi_lum = lum[top:bottom, left:right]
        roi_chroma = chroma[top:bottom, left:right]
        local_bg = float(np.median(roi_lum))
        bright_ink = (roi_lum > max(70.0, local_bg + 35.0)) & (roi_chroma < 65.0)
        if detection.name.endswith("_right"):
            bright_ink = bright_ink[:, ::-1]

        # WeChat's voice glyph is three nested, narrow arcs. Text glyphs tend
        # to be wider and do not grow monotonically from left to right.
        ink_u8 = bright_ink.astype(np.uint8) * 255
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(ink_u8)
        roi_h, roi_w = bright_ink.shape
        wave_components: list[tuple[int, int, int, int, int]] = []
        max_component_width = max(16, int(round(0.065 * roi_w)))
        x_min, x_max = 0.10 * roi_w, 0.36 * roi_w
        for component_x, component_y, component_w, component_h, area in stats[1:]:
            if area < 35 or component_w > max_component_width:
                continue
            if component_h < max(8, int(round(0.08 * roi_h))):
                continue
            if not x_min <= component_x < x_max:
                continue
            wave_components.append(
                (component_x, component_y, component_w, component_h, area)
            )
        wave_components.sort(key=lambda component: component[0])

        verified = False
        for first, second, third in zip(
            wave_components,
            wave_components[1:],
            wave_components[2:],
        ):
            heights = (first[3], second[3], third[3])
            centers = (
                first[1] + first[3] / 2.0,
                second[1] + second[3] / 2.0,
                third[1] + third[3] / 2.0,
            )
            gaps = (second[0] - first[0], third[0] - second[0])
            if (
                heights[1] >= 0.75 * heights[0]
                and heights[2] >= 0.75 * heights[1]
                and heights[2] - heights[0] >= 0.15 * roi_h
                and max(centers) - min(centers) <= 0.25 * roi_h
                and max(gaps) <= 0.10 * roi_w
            ):
                verified = True
                break
        if verified:
            out[top:bottom, left:right] = True
    return out


def _find_chat_bubbles(
    rgb_u8: np.ndarray,
    avatar: np.ndarray,
    green: np.ndarray,
    detections: list[Detection] | None = None,
) -> np.ndarray:
    """Find neutral incoming bubbles without absorbing the dark canvas."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    bg_level = estimate_bg_level(lum)
    seed = (chroma < 26.0) & (lum >= bg_level + 7.0) & (lum < 115.0)
    seed[: int(0.105 * h)] = False
    seed[int(0.90 * h) :] = False
    seed[avatar | green] = False
    u8 = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8)
    detected_bubbles = _detection_mask(rgb_u8.shape[:2], detections, CHAT_BUBBLE_CLASSES)
    # The detector's right-bubble box often wraps a green outgoing bubble;
    # remove that box plus a narrow halo so it cannot create a gray rectangle.
    green_neighborhood = cv2.dilate(green.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    detected_bubbles &= ~green_neighborhood
    out = detected_bubbles
    out |= _find_chat_voice_bubbles(rgb_u8, detections)
    out[: int(0.105 * h)] = False
    out[int(0.90 * h) :] = False
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        fill = area / float(max(bw * bh, 1))
        pixels = lum[labels == i]
        spread = float(np.percentile(pixels, 90) - np.percentile(pixels, 10))
        if area < 600 or bw < 70 or bh < 24:
            continue
        if x < 0.12 * w or x + bw > 0.95 * w or bw > 0.86 * w or bh > 0.55 * h:
            continue
        if fill < 0.22:
            continue
        # A centered date is made of several short horizontal glyph groups;
        # it must not become a fake message card. Textured portrait/image
        # messages are similarly rejected here and handled by media below.
        if area < 3500 and bh < 55 and bw < 220:
            continue
        if bh < 70 and bw < 340 and abs((x + 0.5 * bw) - 0.5 * w) < 0.25 * w:
            continue
        # A tall, text-heavy message can have a large luminance spread even
        # though its neutral component is a solid card.  Only reject the
        # low-fill variant, which is more likely to be a textured image.
        if spread > 32.0 and bw / max(bh, 1) < 1.50 and fill < 0.82:
            continue
        blob = labels == i
        out |= fill_holes(blob)
        out |= cv2.dilate(blob.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)

        # The small triangular pointer is darker than the card body and is
        # normally disconnected from the neutral component.
        if x < 0.45 * w:
            cy = y + bh // 2
            triangle = np.array([[max(0, x - 14), cy], [x + 2, cy - 17], [x + 2, cy + 17]])
            pointer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(pointer, [triangle], 1)
            out |= pointer.astype(bool)
    return out


def _find_chat_media(
    rgb_u8: np.ndarray,
    avatar: np.ndarray,
    bubbles: np.ndarray,
    green: np.ndarray,
) -> np.ndarray:
    """Keep image messages and colored file icons byte-for-byte."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    bg_level = estimate_bg_level(lum)
    # The generic component pass is for colored media/icons only. Neutral
    # glyphs (especially centered dates) are handled by the text pass and
    # must never be restored as original dark pixels.
    candidate = chroma > 22.0
    candidate[: int(0.105 * h)] = False
    candidate[int(0.90 * h) :] = False
    candidate[avatar | bubbles | green] = False
    u8 = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8)
    out = np.zeros((h, w), dtype=bool)
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        fill = area / float(max(bw * bh, 1))
        aspect = bw / max(bh, 1)
        if area < 180 or x < 0.12 * w or y < int(0.105 * h):
            continue
        if bw > 0.82 * w or bh > 0.65 * h or fill < 0.10:
            continue
        if not 0.12 <= aspect <= 8.0:
            continue
        blob = labels == i
        out |= blob
        # A small dilation catches anti-aliased edges of file icons and
        # thumbnails while remaining inside the surrounding dark canvas.
        out |= cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # File icons sit inside a neutral gray bubble, so the bubble mask above
    # must not hide their red/blue/green artwork. Keep these compact accents
    # independently from the surrounding card.
    accent = (chroma > 28.0) & (lum > 30.0)
    accent[: int(0.105 * h)] = False
    accent[int(0.90 * h) :] = False
    accent[avatar | green] = False
    accent_u8 = cv2.morphologyEx(
        accent.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(accent_u8)
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if area < 80 or bw < 10 or bh < 10 or bw > 0.40 * w or bh > 0.30 * h:
            continue
        if x < 0.12 * w:
            continue
        blob = labels == i
        out |= blob
        out |= cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # Bright, textured rectangles are sent image messages. Use their complete
    # bounding rectangle so dark portions of a photo are copied too, instead
    # of being mistaken for the dark chat background.
    bright = (lum > 70.0) | (chroma > 22.0)
    bright[: int(0.105 * h)] = False
    bright[int(0.90 * h) :] = False
    bright[avatar | bubbles | green] = False
    bright_u8 = cv2.morphologyEx(
        bright.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
    )
    bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright_u8)
    for _i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        fill = area / float(max(bw * bh, 1))
        if area < 5000 or x < 0.14 * w or x > 0.82 * w:
            continue
        if bw < 120 or bh < 70 or bw > 0.82 * w or bh > 0.65 * h:
            continue
        if fill < 0.15:
            continue
        out[y : y + bh, x : x + bw] = True
    return out


def _find_chat_text(
    rgb_u8: np.ndarray,
    body_end: int,
    avatar: np.ndarray,
    bubbles: np.ndarray,
    green: np.ndarray,
    media: np.ndarray,
) -> np.ndarray:
    """Recover light-theme ink from header, dates and neutral chat bubbles."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    baseline = cv2.GaussianBlur(lum, (31, 31), 0)
    seed = (lum > 82.0) & (chroma < 65.0) & ((lum - baseline) > 24.0)
    header_end = int(0.09 * h)
    seed[:header_end] = (lum[:header_end] > 78.0) & (chroma[:header_end] < 70.0)
    seed[body_end:] = False

    # Text inside a gray bubble is measured against the bubble itself rather
    # than the page background. This keeps file names and voice durations.
    bg_level = estimate_bg_level(lum)
    bubble_text = bubbles & (lum > bg_level + 45.0) & (chroma < 70.0)
    body_text = (
        (lum > max(58.0, bg_level + 28.0))
        & (lum < 220.0)
        & (chroma < 55.0)
        & ((lum - baseline) > 20.0)
        & ~bubbles
    )
    seed |= bubble_text | body_text
    seed[green | avatar | media] = False

    u8 = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8)
    out = np.zeros((h, w), dtype=bool)
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        fill = area / float(max(bw * bh, 1))
        if area < 3 or area > 18000 or bw > 0.80 * w or bh > 0.30 * h:
            continue
        if area > 300 and bw > 80 and bh > 40 and fill > 0.70:
            continue
        if float(lum[labels == i].mean()) < 70.0:
            continue
        out |= labels == i
    # Keep small voice-duration glyphs from absorbing too much of the dark
    # bubble around their antialiased edges.
    out = cv2.dilate(out.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    out[green | avatar | media] = False
    return out


def _paint_chat_dates(
    out: np.ndarray,
    lum: np.ndarray,
    chroma: np.ndarray,
    protect: np.ndarray,
    bg_level: float,
) -> None:
    """Clear centered gray date rows before painting their complete strokes."""
    h, w = lum.shape
    if bg_level >= 35.0:
        baseline = cv2.GaussianBlur(lum, (31, 31), 0)
        seed = (
            (chroma < 50.0)
            & (lum > bg_level + 8.0)
            & (lum < 180.0)
            & ((lum - baseline) > 12.0)
            & ~protect
        )
        seed[: int(0.10 * h)] = False
    else:
        seed = (chroma < 50.0) & (lum > bg_level + 8.0) & (lum < 180.0) & ~protect
        seed[: int(0.12 * h)] = False
    seed[int(0.90 * h) :] = False
    if bg_level >= 35.0:
        seed = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    x0, x1 = int(0.12 * w), int(0.88 * w)
    row_count = seed[:, x0:x1].sum(axis=1)
    good = row_count > 8
    good = cv2.morphologyEx(
        good.astype(np.uint8)[:, None],
        cv2.MORPH_CLOSE,
        np.ones((7, 1), np.uint8),
    )[:, 0].astype(bool)
    changes = np.diff(np.r_[False, good, False].astype(np.int8))
    starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
    mapped = _map_dark_theme_ink(lum)
    for y0, y1 in zip(starts, ends):
        if not 18 <= y1 - y0 <= 65:
            continue
        yy, xx = np.where(seed[y0:y1, x0:x1])
        if len(xx) < 100:
            continue
        left, right = int(xx.min() + x0), int(xx.max() + x0 + 1)
        span = right - left
        center = (left + right) * 0.5
        if span < 120 or span > 0.75 * w or abs(center - 0.5 * w) > 0.32 * w:
            continue
        pad_x, pad_y = 8, 4
        xa, xb = max(0, left - pad_x), min(w, right + pad_x)
        ya, yb = max(0, y0 - pad_y), min(h, y1 + pad_y)
        roi = np.zeros((h, w), dtype=bool)
        roi[ya:yb, xa:xb] = True
        editable = roi & ~protect
        out[editable] = TARGET_BG
        ink_seed = editable & (chroma < 55.0) & (lum > bg_level + 8.0) & (lum < 180.0)
        if bg_level >= 35.0:
            ink_u8 = cv2.morphologyEx(ink_seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            n_ink, ink_labels, ink_stats, _ = cv2.connectedComponentsWithStats(ink_u8)
            ink = np.zeros((h, w), dtype=bool)
            for i, (_ix, _iy, ibw, ibh, iarea) in enumerate(ink_stats[1:], 1):
                if 20 <= iarea <= 1600 and 5 <= ibw <= 60 and 10 <= ibh <= 55:
                    ink |= ink_labels == i
        else:
            ink = ink_seed
        date_mapped = mapped if bg_level >= 35.0 else np.clip(mapped - 65.0, 65.0, 170.0)
        out[ink] = np.repeat(date_mapped[ink, None], 3, axis=1)


def _convert_wallpaper_chat_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """Convert an elevated-gray chat page whose canvas contains wallpaper."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    bg_level = estimate_bg_level(lum)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    body_start, body_end = int(0.105 * h), int(0.90 * h)

    avatar, avatar_boxes = _find_chat_avatars(rgb_u8)

    # Dark teal wallpaper is also classified as HSV green. Keep only bright,
    # saturated components, which covers outgoing bubbles and file icons but
    # rejects the low-value wallpaper texture.
    color_seed = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 82)
        & (hsv[:, :, 1] >= 40)
        & (hsv[:, :, 2] >= 70)
    )
    color_seed[:body_start] = False
    color_seed[body_end:] = False
    color_u8 = cv2.morphologyEx(
        color_seed.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(color_u8)
    green = np.zeros((h, w), dtype=bool)
    for i, (_x, _y, bw, bh, area) in enumerate(stats[1:], 1):
        if area < 500 or bw > 0.85 * w or bh > 0.65 * h:
            continue
        pixels = hsv[labels == i]
        if float(pixels[:, 1].mean()) < 100.0 or float(pixels[:, 2].mean()) < 100.0:
            continue
        blob = labels == i
        green |= fill_holes(blob)
        green |= cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # Find the stable horizontal runs that form neutral dark message cards.
    # Wallpaper runs usually touch the image edge; message runs start inside
    # the message column and have a bounded width.
    neutral = (chroma < 12.0) & (lum > 25.0) & (lum < 90.0)
    neutral[:body_start] = False
    neutral[body_end:] = False
    neutral_u8 = cv2.morphologyEx(
        neutral.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    neutral_u8 = cv2.morphologyEx(neutral_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    runs: list[tuple[int, int, int]] = []
    for y in range(int(0.11 * h), body_end):
        row = neutral_u8[y] > 0
        row[: int(0.16 * w)] = False
        row[int(0.94 * w) :] = False
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
        for x0, x1 in zip(starts, ends):
            width = int(x1 - x0)
            if int(0.25 * w) <= width <= int(0.82 * w):
                runs.append((y, int(x0), int(x1)))

    run_groups: list[list[tuple[int, int, int]]] = []
    start = 0
    for i in range(1, len(runs) + 1):
        if i == len(runs) or runs[i][0] > runs[i - 1][0] + 1:
            if i > start:
                run_groups.append(runs[start:i])
            start = i

    card_rects: list[tuple[int, int, int, int]] = []
    for group in run_groups:
        if len(group) < 70:
            continue
        y0, y1 = group[0][0], group[-1][0] + 1
        x0 = int(np.median([item[1] for item in group]))
        x1 = int(np.median([item[2] for item in group]))
        if x1 - x0 < 100 or x1 - x0 > 0.82 * w or x0 < 0.16 * w:
            continue
        y0 = max(body_start, y0 - 8)
        y1 = min(body_end, y1 + 35)
        card_rects.append((x0, y0, x1 - x0, y1 - y0))

    # If a bubble background merges into the wallpaper, use the bright text
    # immediately below its avatar as a bounded fallback rectangle.
    for ax, ay, aw, ah in avatar_boxes:
        if ax >= 0.5 * w:
            continue
        x0, x1 = min(w - 1, ax + aw + 25), min(w, int(0.82 * w))
        y0, y1 = min(h, int(ay + 0.20 * ah)), min(body_end, int(ay + 1.70 * ah))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = np.zeros((h, w), dtype=bool)
        roi[y0:y1, x0:x1] = True
        text_seed = roi & (lum > max(70.0, bg_level + 25.0)) & (lum < 220.0) & (chroma < 70.0)
        text_u8 = cv2.morphologyEx(text_seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n2, _labels2, stats2, _ = cv2.connectedComponentsWithStats(text_u8)
        boxes = [
            (int(tx), int(ty), int(tw), int(th))
            for tx, ty, tw, th, area in stats2[1:]
            if 4 <= area <= 3000 and tw <= 220 and th <= 80
        ]
        if not boxes:
            continue
        bx0 = max(x0, min(item[0] for item in boxes) - 35)
        bx1 = min(x1, max(item[0] + item[2] for item in boxes) + 45)
        by0 = max(y0, min(item[1] for item in boxes) - 35)
        by1 = min(y1, max(item[1] + item[3] for item in boxes) + 35)
        if bx1 - bx0 >= 100 and by1 - by0 >= 70:
            card_rects.append((bx0, max(body_start, by0 - 6), bx1 - bx0, min(body_end, by1 + 30) - max(body_start, by0 - 6)))

    # Bright compact accents extend the neutral card to the file icon without
    # restoring low-saturation wallpaper pixels.
    icon_seed = (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 55)
    icon_seed[:body_start] = False
    icon_seed[body_end:] = False
    icon_seed[avatar | green] = False
    icon_u8 = cv2.morphologyEx(icon_seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n3, labels3, stats3, _ = cv2.connectedComponentsWithStats(icon_u8)
    icon_mask = np.zeros((h, w), dtype=bool)
    for i, (x, y, bw, bh, area) in enumerate(stats3[1:], 1):
        fill = area / float(max(bw * bh, 1))
        if not 250 <= area <= 40000 or not 20 <= bw <= 220 or not 20 <= bh <= 220:
            continue
        if not 0.25 <= bw / max(bh, 1) <= 2.5 or fill < 0.12:
            continue
        pixels = hsv[labels3 == i]
        mean_s, mean_v = float(pixels[:, 1].mean()), float(pixels[:, 2].mean())
        colorful = (mean_s > 120.0 and mean_v > 100.0) or (mean_s > 70.0 and mean_v > 150.0)
        neutral_file_icon = bw >= 70 and bh >= 70 and fill > 0.55 and mean_s > 30.0 and mean_v > 70.0
        if x < 0.14 * w or not (colorful or neutral_file_icon):
            continue
        blob = labels3 == i
        icon_mask |= blob
        best = None
        for rect_i, (rx, ry, rw, rh) in enumerate(card_rects):
            overlap = min(y + bh, ry + rh + 60) - max(y, ry - 60)
            if overlap <= 0 or x + bw < rx - 30 or x > rx + rw + 180:
                continue
            distance = abs((rx + rw) - x)
            if best is None or distance < best[0]:
                best = (distance, rect_i)
        if best is not None:
            rect_i = best[1]
            rx, ry, rw, rh = card_rects[rect_i]
            nx = min(w, max(rx + rw, x + bw + 45))
            ny0, ny1 = min(ry, y - 45), max(ry + rh, y + bh + 45)
            card_rects[rect_i] = (rx, max(0, ny0), nx - rx, min(h, ny1) - max(0, ny0))

    cards = np.zeros((h, w), dtype=bool)
    for x, y, bw, bh in card_rects:
        cards[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)] = True

    out = np.empty_like(rgb)
    out[:] = TARGET_BG
    out[cards] = np.array([232.0, 232.0, 232.0], dtype=np.float32)
    out[avatar] = rgb[avatar]
    out[green] = rgb[green]

    # Only restore selected icon components. Dark wallpaper inside a translucent
    # file bubble must follow the new light bubble instead of being copied back.
    icon_mask &= cards & ~avatar & ~green
    out[icon_mask] = rgb[icon_mask]
    mapped = _map_dark_theme_ink(lum)
    # Wallpaper can show through translucent message cards. Keep foreground
    # components that are locally brighter than the card, rather than copying
    # every bright wallpaper pixel as gray speckles.
    card_text_seed = cards & (lum > 82.0) & (chroma < 70.0) & ~icon_mask
    card_baseline = cv2.GaussianBlur(lum, (31, 31), 0)
    card_text_seed &= (lum - card_baseline) > 18.0
    card_text_u8 = cv2.morphologyEx(card_text_seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n_text, text_labels, text_stats, _ = cv2.connectedComponentsWithStats(card_text_u8)
    card_text = np.zeros((h, w), dtype=bool)
    for i, (_x, _y, bw, bh, area) in enumerate(text_stats[1:], 1):
        if 4 <= area <= 18000 and bw <= 0.80 * w and bh <= 0.30 * h:
            card_text |= text_labels == i
    out[card_text] = np.repeat(mapped[card_text, None], 3, axis=1)

    protect = cards | avatar | green | icon_mask
    paint_header_chrome(out, lum, chroma, protect, bg_level)
    _paint_chat_dates(out, lum, chroma, protect, bg_level)

    # Sender labels sit outside the message card but inside the avatar row.
    sender = np.zeros((h, w), dtype=bool)
    n4, _labels4, stats4, _ = cv2.connectedComponentsWithStats(avatar.astype(np.uint8) * 255)
    for ax, ay, aw, ah, area in stats4[1:]:
        if area < 1000:
            continue
        x0, x1 = int(ax + aw + 15), min(w, int(ax + aw + 360))
        y0, y1 = max(int(0.11 * h), int(ay - 0.15 * ah)), min(body_end, int(ay + 0.45 * ah))
        roi = np.zeros((h, w), dtype=bool)
        roi[y0:y1, x0:x1] = True
        seed = roi & (lum > max(78.0, bg_level + 25.0)) & (lum < 220.0) & (chroma < 55.0) & ~protect
        seed_u8 = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n5, labels5, stats5, _ = cv2.connectedComponentsWithStats(seed_u8)
        for i, (_x, _y, bw, bh, component_area) in enumerate(stats5[1:], 1):
            if 15 <= component_area <= 3000 and 8 <= bw <= 200 and 8 <= bh <= 70:
                sender |= labels5 == i
    sender = cv2.dilate(sender.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    sender &= ~protect
    out[sender] = np.repeat(mapped[sender, None], 3, axis=1)

    footer_start = int(0.90 * h)
    out[footer_start:] = np.array([236.0, 236.0, 236.0], dtype=np.float32)
    input_y0, input_y1 = int(0.91 * h), int(0.975 * h)
    input_x0, input_x1 = int(0.11 * w), int(0.79 * w)
    out[input_y0:input_y1, input_x0:input_x1] = np.array([245.0, 245.0, 245.0], dtype=np.float32)
    footer_ink = (lum[footer_start:] > 78.0) & (chroma[footer_start:] < 70.0)
    footer_ink &= ~avatar[footer_start:] & ~green[footer_start:]
    for c in range(3):
        out[footer_start:, :, c][footer_ink] = mapped[footer_start:][footer_ink]

    out[avatar | green | icon_mask] = rgb[avatar | green | icon_mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def _convert_legacy_dark_chat_rgb(
    rgb_u8: np.ndarray,
    detections: list[Detection] | None = None,
) -> np.ndarray:
    """Convert the older, elevated-gray chat theme with the legacy masks."""
    rgb = rgb_u8.astype(np.float32)
    out = rgb.copy()
    h, w = out.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    bg_level = estimate_bg_level(lum)
    if bg_level >= 35.0:
        return _convert_wallpaper_chat_rgb(rgb_u8)
    bg_tol = 10.0

    hard, soft, green, avatar, tables, gray_ui = build_protect(rgb_u8, lum, chroma, bg_level)
    detected_media = _detection_mask(rgb_u8.shape[:2], detections, CHAT_MEDIA_CLASSES, padding=2)
    model_preserve = detected_media
    hard |= model_preserve
    soft |= cv2.dilate(model_preserve.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~hard
    protect_for_bg = hard | soft
    protect_chrome = hard | tables | green | avatar | gray_ui

    near_bg = (np.abs(lum - bg_level) <= bg_tol) & (chroma < 28) & ~protect_for_bg
    out[near_bg] = TARGET_BG

    # Header/status panel: some pages use elevated dark-gray (~30–40) that misses
    # global near_bg. Replace locally through the nav bar under the title so pages
    # like img_0003 don't leave a black slab between title and first bubbles.
    y_head = max(1, int(0.112 * h))
    head_band = lum[8 : max(9, int(0.055 * h)), int(0.08 * w) : int(0.92 * w)]
    header_bg = float(np.median(head_band)) if head_band.size else float(bg_level)
    head_tol = max(14.0, abs(header_bg - bg_level) + 8.0)
    protect_strict = hard
    head_bg_mask = np.zeros((h, w), dtype=bool)
    head_bg_mask[:y_head] = True
    head_bg_mask &= ~protect_strict & (chroma < 30) & (np.abs(lum - header_bg) <= head_tol) & (lum < 90)
    out[head_bg_mask] = TARGET_BG

    gap = np.zeros((h, w), dtype=bool)
    gap[int(0.09 * h) : y_head] = True
    gap &= ~protect_strict & (chroma < 28) & (lum < 55) & (lum > bg_level + 3.0)
    out[gap] = TARGET_BG

    y0_sep, y1_sep = int(0.108 * h), min(int(0.120 * h), h)
    for y in range(y0_sep, y1_sep):
        free = ~protect_strict[y] & (chroma[y] < 35)
        dark = free & (lum[y] < 75)
        if dark.mean() >= 0.12:
            out[y, dark] = TARGET_BG
            mid = free & (lum[y] >= 75) & (lum[y] < 140)
            if mid.mean() >= 0.05:
                out[y, mid] = np.array([220, 220, 220], dtype=np.float32)

    convert_gray_ui_cards(out, gray_ui, green, lum, chroma, rgb_u8)
    remap_chrome(out, lum, chroma, protect_chrome, bg_level)

    out[green] = rgb[green]
    out[avatar] = rgb[avatar]
    out[tables] = rgb[tables]
    out[model_preserve] = rgb[model_preserve]

    convert_gray_ui_cards(out, gray_ui, green, lum, chroma, rgb_u8)
    cleanup_gray_ui_fringe(out, gray_ui, avatar, green, TARGET_BG)
    cleanup_green_shell(out, rgb_u8, green, avatar, TARGET_BG)
    fix_avatar_rounded_corners(out, rgb_u8, avatar, bg_level, TARGET_BG)

    y0_sep, y1_sep = int(0.108 * h), min(int(0.120 * h), h)
    out_lum = out.mean(axis=2)
    out_ch = out.max(2) - out.min(2)
    for y in range(y0_sep, y1_sep):
        crumb = ~green[y] & ~avatar[y] & (out_ch[y] < 40) & (out_lum[y] < 90)
        if crumb.mean() >= 0.08:
            out[y, crumb] = TARGET_BG

    foot = np.zeros((h, w), dtype=bool)
    foot[int(0.90 * h) :, :] = True
    foot &= ~avatar & ~green
    foot_body = foot & (chroma < 35) & (lum > bg_level - 2) & (lum < 70)
    out[foot_body] = np.array([236, 236, 236], dtype=np.float32)
    foot_ink = foot & (chroma < 50) & (lum >= 70)
    if np.any(foot_ink):
        t = np.clip((lum[foot_ink] - 70.0) / 160.0, 0.0, 1.0)
        for c in range(3):
            out[:, :, c][foot_ink] = 236.0 * (1.0 - t) + INK * t

    frame = (lum > 200) & (chroma < 20)
    ys = np.any(frame, axis=1)
    xs = np.any(frame, axis=0)
    if ys.any() and xs.any():
        y0, y1 = int(np.argmax(ys)), int(h - np.argmax(ys[::-1]))
        x0, x1 = int(np.argmax(xs)), int(w - np.argmax(xs[::-1]))
        if y0 <= 4 and x0 <= 4 and (y1 - y0) > 0.9 * h and (x1 - x0) > 0.9 * w:
            border = np.zeros((h, w), dtype=bool)
            border[: y0 + 3, :] = True
            border[y1 - 3 :, :] = True
            border[:, : x0 + 3] = True
            border[:, x1 - 3 :] = True
            out[border & frame] = rgb[border & frame]

    return np.clip(np.nan_to_num(out, nan=TARGET_BG[0]), 0, 255).astype(np.uint8)


def _convert_chat_rgb(
    rgb_u8: np.ndarray,
    detections: list[Detection] | None = None,
) -> np.ndarray:
    """Rebuild a dark chat page on a light canvas while preserving media."""
    rgb = rgb_u8.astype(np.float32)
    h, w = rgb.shape[:2]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(2) - rgb.min(2)
    if estimate_bg_level(lum) >= 35.0:
        return _convert_legacy_dark_chat_rgb(rgb_u8, detections)
    footer_start = int(0.90 * h)

    out = np.empty_like(rgb)
    out[:] = TARGET_BG
    avatar, _avatar_boxes = _find_chat_avatars(rgb_u8)
    green = _find_chat_green_bubbles(rgb_u8)
    bubbles = _find_chat_bubbles(rgb_u8, avatar, green, detections)
    media = _find_chat_media(rgb_u8, avatar, bubbles, green)
    media |= _detection_mask(rgb_u8.shape[:2], detections, CHAT_MEDIA_CLASSES, padding=2)
    text = _find_chat_text(rgb_u8, footer_start, avatar, bubbles, green, media)
    mapped = _map_dark_theme_ink(lum)

    # Neutral message cards remain slightly darker than the page canvas so
    # their boundaries survive the theme conversion.
    out[bubbles] = np.array([232.0, 232.0, 232.0], dtype=np.float32)
    for c in range(3):
        out[:, :, c][text] = mapped[text]

    # Footer/input chrome has a different light-theme gray than the chat body.
    out[footer_start:] = np.array([236.0, 236.0, 236.0], dtype=np.float32)
    input_y0, input_y1 = int(0.91 * h), int(0.975 * h)
    input_x0, input_x1 = int(0.11 * w), int(0.79 * w)
    out[input_y0:input_y1, input_x0:input_x1] = np.array([245.0, 245.0, 245.0], dtype=np.float32)
    footer_ink = (lum[footer_start:] > 78.0) & (chroma[footer_start:] < 70.0)
    footer_ink &= ~media[footer_start:] & ~avatar[footer_start:] & ~green[footer_start:]
    for c in range(3):
        out[footer_start:, :, c][footer_ink] = mapped[footer_start:][footer_ink]

    # Dates are gray text on the dark canvas. Wipe their complete centered
    # row, then map all antialiased strokes without leaving a dark-theme halo.
    _paint_chat_dates(out, lum, chroma, avatar | media | green | bubbles, estimate_bg_level(lum))

    # Restore outgoing bubbles, image messages, colored file icons and the
    # complete avatar tiles after all neutral-background operations finish.
    out[green | avatar | media] = rgb[green | avatar | media]
    return np.clip(out, 0, 255).astype(np.uint8)


def process_rgb(rgb_u8: np.ndarray, detections: list[Detection] | None = None) -> np.ndarray:
    kind = _classify_screenshot(rgb_u8, detections)
    if kind == "document":
        return rgb_u8.copy()
    if kind == "phone_document":
        return _convert_phone_document_rgb(rgb_u8)
    if kind == "chat":
        return _convert_chat_rgb(rgb_u8, detections)
    if kind == "profile":
        return _convert_profile_rgb(rgb_u8)

    return _convert_legacy_dark_chat_rgb(rgb_u8, detections)


def _to_rgb_array(img: Image.Image) -> np.ndarray:
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.split()[-1])
        img = rgb
    else:
        img = img.convert("RGB")
    return np.asarray(img)


def _emit_log(message: str, callback: Callable[[str], None] | None = None) -> None:
    if callback is not None:
        callback(message)
    elif sys.stdout is not None:
        print(message)


def _process_image(
    img: Image.Image,
    detector: YoloDetector | None = None,
) -> tuple[np.ndarray, str, list[Detection]]:
    rgb = _to_rgb_array(img)
    detections = detector.detect(rgb) if detector is not None else []
    kind = _classify_screenshot(rgb, detections)
    return process_rgb(rgb, detections), kind, detections


def process_image_bytes(data: bytes, detector: YoloDetector | None = None) -> bytes:
    with Image.open(io.BytesIO(data)) as img:
        arr, _kind, _detections = _process_image(img, detector)
    out = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=98, subsampling=0, optimize=True)
    return buf.getvalue()


def convert_image(
    input_image: Path,
    output_image: Path,
    detector: YoloDetector | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    if not input_image.is_file():
        raise FileNotFoundError(f"输入图片不存在: {input_image}")
    with Image.open(input_image) as img:
        arr, kind, detections = _process_image(img, detector)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    out = Image.fromarray(arr, "RGB")
    suffix = output_image.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        out.save(output_image, format="JPEG", quality=98, subsampling=0, optimize=True)
    elif suffix == ".png":
        out.save(output_image, format="PNG", optimize=True)
    else:
        raise ValueError("单图输出格式仅支持 .jpg/.jpeg/.png")
    _emit_log(f"Saved: {output_image} (kind={kind}, detections={len(detections)})", log)


def process_pdf(
    input_pdf: Path,
    output_pdf: Path,
    dump_dir: Path | None = None,
    processed_dir: Path | None = None,
    detector: YoloDetector | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    import fitz

    doc = fitz.open(input_pdf)
    xref_cache: dict[int, bytes] = {}
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)

    img_i = 0
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in xref_cache:
                new_bytes = xref_cache[xref]
            else:
                try:
                    raw = doc.extract_image(xref)
                except Exception as e:
                    _emit_log(f"[skip] page={page_index} xref={xref}: {e}", log)
                    continue
                data = raw["image"]
                if dump_dir is not None:
                    ext = raw.get("ext", "bin")
                    (dump_dir / f"img_{img_i:04d}.{ext}").write_bytes(data)
                try:
                    new_bytes = process_image_bytes(data, detector)
                except Exception as e:
                    _emit_log(f"[fail] page={page_index} xref={xref}: {e}", log)
                    img_i += 1
                    continue
                xref_cache[xref] = new_bytes
                if processed_dir is not None:
                    (processed_dir / f"img_{img_i:04d}.jpg").write_bytes(new_bytes)
                _emit_log(f"[ok] page={page_index} xref={xref} -> img_{img_i:04d}", log)
            try:
                page.replace_image(xref, stream=new_bytes)
            except Exception as e:
                _emit_log(f"[replace-fail] page={page_index} xref={xref}: {e}", log)
            img_i += 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_pdf, deflate=True, garbage=3)
    doc.close()
    _emit_log(f"Saved: {output_pdf}", log)


def rebuild_pdf_from_processed(
    input_pdf: Path,
    processed_dir: Path,
    output_pdf: Path,
    log: Callable[[str], None] | None = None,
) -> None:
    """Rebuild a PDF from already-processed images, using input.pdf as layout template.

    Image indices follow the exact traversal order used by process_pdf (per page,
    per image, repeated xrefs reuse the first occurrence's processed file).
    """
    if not input_pdf.exists():
        raise FileNotFoundError(f"模板 PDF 不存在: {input_pdf}")
    if not processed_dir.is_dir():
        raise NotADirectoryError(f"处理后图片目录不存在: {processed_dir}")

    import fitz

    doc = fitz.open(input_pdf)
    xref_cache: dict[int, bytes] = {}
    img_i = 0
    replaced = 0
    missing = 0
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in xref_cache:
                new_bytes = xref_cache[xref]
            else:
                path = processed_dir / f"img_{img_i:04d}.jpg"
                if not path.exists():
                    _emit_log(
                        f"[missing] page={page_index} xref={xref}: {path.name} 不存在，保留原图",
                        log,
                    )
                    missing += 1
                    img_i += 1
                    continue
                new_bytes = path.read_bytes()
                xref_cache[xref] = new_bytes
            try:
                page.replace_image(xref, stream=new_bytes)
                replaced += 1
            except Exception as e:
                _emit_log(f"[replace-fail] page={page_index} xref={xref}: {e}", log)
            img_i += 1

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_pdf, deflate=True, garbage=3)
    doc.close()
    _emit_log(f"Saved: {output_pdf} (replaced={replaced}, missing={missing})", log)


def _create_detector(
    base: Path,
    model: str | None,
    confidence: float,
    iou: float,
    imgsz: int,
    no_yolo: bool,
) -> YoloDetector | None:
    if no_yolo:
        return None

    if model:
        model_path = Path(model).expanduser()
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
    else:
        model_path = base / "best.onnx"
    return YoloDetector(model_path, confidence=confidence, iou=iou, imgsz=imgsz)


def run_windows_gui() -> None:
    """Run the Windows-only Tkinter front end."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    base = Path(__file__).resolve().parent

    class WindowsApp:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("WeChat2White")
            self.root.geometry("780x650")
            self.root.minsize(700, 560)
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(5, weight=1)

            self.mode = tk.StringVar(value="image")
            self.input_path = tk.StringVar()
            self.output_path = tk.StringVar()
            self.image_output_path = tk.StringVar()
            self.model_path = tk.StringVar()
            self.confidence = tk.StringVar(value="0.25")
            self.iou = tk.StringVar(value="0.45")
            self.imgsz = tk.StringVar(value="640")
            self.no_yolo = tk.BooleanVar(value=False)
            self.status = tk.StringVar(value="请选择输入图片或 PDF 文件")
            self.events: queue.Queue[tuple[str, object]] = queue.Queue()

            self._build_widgets()
            self._on_mode_changed()
            self.root.after(100, self._poll_events)

        def _build_widgets(self) -> None:
            mode_frame = ttk.LabelFrame(self.root, text="转换类型", padding=8)
            mode_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
            ttk.Radiobutton(
                mode_frame,
                text="转换单张图片",
                variable=self.mode,
                value="image",
                command=self._on_mode_changed,
            ).grid(row=0, column=0, padx=(0, 24), sticky="w")
            ttk.Radiobutton(
                mode_frame,
                text="转换单个 PDF",
                variable=self.mode,
                value="pdf",
                command=self._on_mode_changed,
            ).grid(row=0, column=1, sticky="w")

            input_frame = ttk.LabelFrame(self.root, text="输入", padding=8)
            input_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
            input_frame.columnconfigure(1, weight=1)
            self.input_label = ttk.Label(input_frame, width=12)
            self.input_label.grid(row=0, column=0, sticky="w")
            ttk.Entry(input_frame, textvariable=self.input_path).grid(
                row=0, column=1, sticky="ew", padx=8
            )
            self.input_button = ttk.Button(input_frame, command=self._browse_input)
            self.input_button.grid(row=0, column=2, sticky="e")

            output_frame = ttk.LabelFrame(self.root, text="输出", padding=8)
            output_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
            output_frame.columnconfigure(1, weight=1)
            self.output_label = ttk.Label(output_frame, width=12)
            self.output_label.grid(row=0, column=0, sticky="w")
            ttk.Entry(output_frame, textvariable=self.output_path).grid(
                row=0, column=1, sticky="ew", padx=8
            )
            self.output_button = ttk.Button(output_frame, command=self._browse_output)
            self.output_button.grid(row=0, column=2, sticky="e")

            self.image_output_frame = ttk.LabelFrame(self.root, text="图片输出", padding=8)
            self.image_output_frame.grid(row=3, column=0, sticky="ew", padx=12, pady=6)
            self.image_output_frame.columnconfigure(1, weight=1)
            ttk.Label(self.image_output_frame, text="图片输出目录", width=12).grid(
                row=0, column=0, sticky="w"
            )
            ttk.Entry(self.image_output_frame, textvariable=self.image_output_path).grid(
                row=0, column=1, sticky="ew", padx=8
            )
            ttk.Button(
                self.image_output_frame,
                text="选择目录",
                command=self._browse_image_output,
            ).grid(row=0, column=2, sticky="e")

            params_frame = ttk.LabelFrame(self.root, text="处理参数", padding=8)
            params_frame.grid(row=4, column=0, sticky="ew", padx=12, pady=6)
            params_frame.columnconfigure(1, weight=1)
            ttk.Label(params_frame, text="YOLO 模型").grid(row=0, column=0, sticky="w")
            ttk.Entry(params_frame, textvariable=self.model_path).grid(
                row=0, column=1, columnspan=4, sticky="ew", padx=8
            )
            ttk.Button(params_frame, text="选择文件", command=self._browse_model).grid(
                row=0, column=5, sticky="e"
            )
            ttk.Label(params_frame, text="置信度").grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(params_frame, textvariable=self.confidence, width=10).grid(
                row=1, column=1, sticky="w", padx=8, pady=(8, 0)
            )
            ttk.Label(params_frame, text="IoU").grid(row=1, column=2, sticky="w", pady=(8, 0))
            ttk.Entry(params_frame, textvariable=self.iou, width=10).grid(
                row=1, column=3, sticky="w", padx=8, pady=(8, 0)
            )
            ttk.Label(params_frame, text="推理尺寸").grid(row=1, column=4, sticky="w", pady=(8, 0))
            ttk.Entry(params_frame, textvariable=self.imgsz, width=10).grid(
                row=1, column=5, sticky="w", padx=8, pady=(8, 0)
            )
            ttk.Checkbutton(
                params_frame,
                text="禁用 YOLO，使用启发式判断",
                variable=self.no_yolo,
            ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))

            log_frame = ttk.LabelFrame(self.root, text="处理日志", padding=8)
            log_frame.grid(row=5, column=0, sticky="nsew", padx=12, pady=6)
            log_frame.rowconfigure(0, weight=1)
            log_frame.columnconfigure(0, weight=1)
            self.log = tk.Text(log_frame, height=12, wrap="word", state="normal")
            self.log.grid(row=0, column=0, sticky="nsew")
            log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
            log_scrollbar.grid(row=0, column=1, sticky="ns")
            self.log.configure(yscrollcommand=log_scrollbar.set)

            action_frame = ttk.Frame(self.root, padding=(12, 6, 12, 12))
            action_frame.grid(row=6, column=0, sticky="ew")
            action_frame.columnconfigure(1, weight=1)
            self.start_button = ttk.Button(action_frame, text="开始转换", command=self._start)
            self.start_button.grid(row=0, column=0, sticky="w")
            self.progress = ttk.Progressbar(action_frame, mode="determinate", length=240)
            self.progress.grid(row=0, column=1, sticky="ew", padx=12)
            ttk.Label(action_frame, textvariable=self.status).grid(row=0, column=2, sticky="e")

        def _on_mode_changed(self) -> None:
            is_image = self.mode.get() == "image"
            self.input_label.configure(text="图片文件" if is_image else "PDF 文件")
            self.output_label.configure(text="输出图片" if is_image else "输出 PDF")
            self.input_button.configure(text="选择文件" if is_image else "选择 PDF")
            self.output_button.configure(text="选择文件" if is_image else "选择输出")
            if is_image:
                self.image_output_frame.grid_remove()
            else:
                self.image_output_frame.grid()
            self.input_path.set("")
            self.output_path.set("")
            self.image_output_path.set("")

        def _browse_input(self) -> None:
            if self.mode.get() == "image":
                selected = filedialog.askopenfilename(
                    title="选择待转换图片",
                    filetypes=[
                        ("图片文件", "*.jpg *.jpeg *.png"),
                        ("所有文件", "*.*"),
                    ],
                )
            else:
                selected = filedialog.askopenfilename(
                    title="选择待转换 PDF",
                    filetypes=[("PDF 文件", "*.pdf"), ("所有文件", "*.*")],
                )
            if not selected:
                return
            self.input_path.set(selected)
            input_path = Path(selected)
            if self.mode.get() == "image":
                suffix = input_path.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png"}:
                    suffix = ".png"
                self.output_path.set(str(input_path.with_name(f"{input_path.stem}_white{suffix}")))
            else:
                self.output_path.set(str(input_path.with_name(f"{input_path.stem}_white.pdf")))
                self.image_output_path.set(str(input_path.with_name(f"{input_path.stem}_images")))

        def _browse_output(self) -> None:
            if self.mode.get() == "image":
                current = Path(self.output_path.get()) if self.output_path.get() else None
                selected = filedialog.asksaveasfilename(
                    title="选择输出图片",
                    initialdir=str(current.parent) if current else None,
                    initialfile=current.name if current else "output_white.png",
                    defaultextension=".png",
                    filetypes=[
                        ("PNG 图片", "*.png"),
                        ("JPEG 图片", "*.jpg *.jpeg"),
                    ],
                )
            else:
                current = Path(self.output_path.get()) if self.output_path.get() else None
                selected = filedialog.asksaveasfilename(
                    title="选择输出 PDF",
                    initialdir=str(current.parent) if current else None,
                    initialfile=current.name if current else "output_white.pdf",
                    defaultextension=".pdf",
                    filetypes=[("PDF 文件", "*.pdf")],
                )
            if selected:
                self.output_path.set(selected)

        def _browse_image_output(self) -> None:
            selected = filedialog.askdirectory(title="选择图片输出目录")
            if selected:
                self.image_output_path.set(selected)

        def _browse_model(self) -> None:
            selected = filedialog.askopenfilename(
                title="选择 YOLO ONNX 模型",
                filetypes=[("ONNX 模型", "*.onnx"), ("所有文件", "*.*")],
            )
            if selected:
                self.model_path.set(selected)

        def _collect_config(self) -> dict[str, object]:
            input_value = self.input_path.get().strip()
            if not input_value:
                raise ValueError("请选择输入图片或 PDF 文件")
            input_path = Path(input_value).expanduser()

            try:
                confidence = float(self.confidence.get().strip())
                iou = float(self.iou.get().strip())
                imgsz = int(self.imgsz.get().strip())
            except ValueError as exc:
                raise ValueError("置信度、IoU 必须是小数，推理尺寸必须是整数") from exc
            if not 0 < confidence <= 1:
                raise ValueError("置信度必须大于 0 且不超过 1")
            if not 0 <= iou <= 1:
                raise ValueError("IoU 必须在 0 到 1 之间")
            if imgsz <= 0:
                raise ValueError("推理尺寸必须大于 0")

            if self.mode.get() == "image":
                if not input_path.is_file():
                    raise ValueError(f"图片不存在: {input_path}")
                if input_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    raise ValueError("单图输入仅支持 .jpg、.jpeg、.png")
                output_value = self.output_path.get().strip()
                output_path = Path(output_value).expanduser() if output_value else input_path.with_name(
                    f"{input_path.stem}_white{input_path.suffix}"
                )
                if output_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    raise ValueError("单图输出仅支持 .jpg、.jpeg、.png")
                return {
                    "mode": "image",
                    "input": input_path,
                    "output": output_path,
                    "model": self.model_path.get().strip() or None,
                    "confidence": confidence,
                    "iou": iou,
                    "imgsz": imgsz,
                    "no_yolo": self.no_yolo.get(),
                }

            if not input_path.is_file():
                raise ValueError(f"PDF 文件不存在: {input_path}")
            if input_path.suffix.lower() != ".pdf":
                raise ValueError("PDF 输入必须是 .pdf 文件")
            output_value = self.output_path.get().strip()
            output_path = Path(output_value).expanduser() if output_value else input_path.with_name(
                f"{input_path.stem}_white.pdf"
            )
            if output_path.suffix.lower() != ".pdf":
                raise ValueError("PDF 输出必须使用 .pdf 扩展名")
            image_output_value = self.image_output_path.get().strip()
            image_output_dir = (
                Path(image_output_value).expanduser()
                if image_output_value
                else input_path.with_name(f"{input_path.stem}_images")
            )
            if image_output_dir.exists() and not image_output_dir.is_dir():
                raise ValueError(f"图片输出路径不是文件夹: {image_output_dir}")
            return {
                "mode": "pdf",
                "input": input_path,
                "output": output_path,
                "image_output_dir": image_output_dir,
                "model": self.model_path.get().strip() or None,
                "confidence": confidence,
                "iou": iou,
                "imgsz": imgsz,
                "no_yolo": self.no_yolo.get(),
            }

        def _start(self) -> None:
            try:
                config = self._collect_config()
            except ValueError as exc:
                messagebox.showerror("参数错误", str(exc))
                return

            item_count = 1
            self.log.delete("1.0", "end")
            self.progress.configure(maximum=item_count, value=0)
            self.status.set("正在处理...")
            self.start_button.configure(state="disabled")
            threading.Thread(target=self._convert, args=(config,), daemon=True).start()

        def _queue_log(self, message: str) -> None:
            self.events.put(("log", message))

        def _convert(self, config: dict[str, object]) -> None:
            try:
                detector = _create_detector(
                    base,
                    config["model"],
                    config["confidence"],
                    config["iou"],
                    config["imgsz"],
                    config["no_yolo"],
                )
                if config["mode"] == "image":
                    self._queue_log(f"开始转换: {config['input']}")
                    convert_image(
                        config["input"],
                        config["output"],
                        detector=detector,
                        log=self._queue_log,
                    )
                    self.events.put(("progress", 1))
                    self.events.put(("done", f"转换完成: {config['output']}"))
                    return

                self._queue_log(f"开始转换: {config['input']}")
                image_output_dir = config["image_output_dir"]
                process_pdf(
                    config["input"],
                    config["output"],
                    dump_dir=image_output_dir / "extracted_images",
                    processed_dir=image_output_dir / "processed_images",
                    detector=detector,
                    log=self._queue_log,
                )
                self.events.put(("progress", 1))
                self.events.put(
                    (
                        "done",
                        f"转换完成: {config['output']}\n图片目录: {image_output_dir}",
                    )
                )
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        def _append_log(self, message: str) -> None:
            self.log.insert("end", f"{message}\n")
            self.log.see("end")

        def _poll_events(self) -> None:
            try:
                while True:
                    event, payload = self.events.get_nowait()
                    if event == "log":
                        self._append_log(str(payload))
                    elif event == "progress":
                        self.progress.configure(value=payload)
                    elif event == "done":
                        self.start_button.configure(state="normal")
                        self.status.set("处理完成")
                        self._append_log(str(payload))
                        messagebox.showinfo("转换完成", str(payload))
                    elif event == "error":
                        self.start_button.configure(state="normal")
                        self.status.set("处理失败")
                        self._append_log(str(payload))
                        messagebox.showerror("转换失败", str(payload))
            except queue.Empty:
                pass
            self.root.after(100, self._poll_events)

    root = tk.Tk()
    WindowsApp(root)
    root.mainloop()


def main() -> None:
    if platform.system() == "Windows":
        run_windows_gui()
        return

    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="微信深色截图 PDF/图片 -> 浅色主题；使用 best.onnx 做页面检测分类。",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="process",
        choices=("process", "rebuild", "image"),
        help="process: 处理 input.pdf（默认）；rebuild: 从 processed_images 重建；image: 转换单张图片",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="rebuild 模式下的输出路径；image 模式下的输入图片路径",
    )
    parser.add_argument(
        "image_output",
        nargs="?",
        default=None,
        help="image 模式下的输出图片路径（默认在输入文件名后加 _white）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO ONNX 模型路径，默认使用脚本目录下的 best.onnx",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU 阈值，默认 0.45")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸，默认 640")
    parser.add_argument("--no-yolo", action="store_true", help="禁用 YOLO，使用原有启发式分类")
    args = parser.parse_args()

    detector = None
    if args.command != "rebuild":
        detector = _create_detector(
            base,
            args.model,
            args.conf,
            args.iou,
            args.imgsz,
            args.no_yolo,
        )

    if args.command == "rebuild":
        if args.image_output is not None:
            parser.error("rebuild 模式只接受一个输出路径")
        out_path = Path(args.path) if args.path else base / "out.pdf"
        rebuild_pdf_from_processed(base / "input.pdf", base / "processed_images", out_path)
    elif args.command == "image":
        if args.path is None:
            parser.error("image 模式需要输入图片路径")
        input_path = Path(args.path)
        if args.image_output:
            output_path = Path(args.image_output)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_white{input_path.suffix}")
        convert_image(input_path, output_path, detector)
    else:
        if args.path is not None or args.image_output is not None:
            parser.error("process 模式不接受位置参数")
        process_pdf(
            base / "input.pdf",
            base / "output.pdf",
            base / "extracted_images",
            base / "processed_images",
            detector,
        )


if __name__ == "__main__":
    main()
