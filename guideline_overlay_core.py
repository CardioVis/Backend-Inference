"""Guideline overlay geometry + visualization (from predict_frames_guideline_overlay notebook).

Call :func:`set_overlay_runtime` with the same keys the notebook used as UPPER_SNAKE globals
before using :func:`preprocess_frame` or :func:`overlay_guideline`.
"""
from __future__ import annotations

import types
from typing import Any, Optional

import cv2
import numpy as np
import torch
from skimage.restoration import denoise_tv_chambolle

_RT: Any = None


def set_overlay_runtime(**kwargs: Any) -> None:
    """Bind DEVICE, IMG_SIZE, CLASS_NAMES, GUIDELINE_ANCHOR_CLASS_ID, … (notebook globals)."""
    global _RT
    _RT = types.SimpleNamespace(**kwargs)


def _cfg() -> Any:
    if _RT is None:
        raise RuntimeError("guideline_overlay_core: call set_overlay_runtime(...) first")
    return _RT


def preprocess_frame(frame):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (_cfg().IMG_SIZE, _cfg().IMG_SIZE))
    img = np.clip(img, 0, 255).astype(np.float32) / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(_cfg().DEVICE)
    return img, (h, w)


def draw_dashed_line(img, pt1, pt2, color, thickness, dash_len, gap_len):
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length < 1e-6:
        return
    step = dash_len + gap_len
    n = int(length / step) + 1
    for i in range(n):
        t0 = min(i * step / length, 1.0)
        t1 = min((i * step + dash_len) / length, 1.0)
        if t0 >= 1:
            break
        p0 = (int(x1 + t0 * (x2 - x1)), int(y1 + t0 * (y2 - y1)))
        p1 = (int(x1 + t1 * (x2 - x1)), int(y1 + t1 * (y2 - y1)))
        cv2.line(img, p0, p1, color, thickness)


def get_pericardium_centerline(mask, gap_threshold=5, smooth_win=3):
    """Lấy đường centerline trong lòng mask: mỗi cột x lấy y_min, y_max của mask rồi y_mid = (min+max)/2."""
    h, w = mask.shape[:2]
    points_by_x = []
    for x in range(w):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            y_mid = (ys.min() + ys.max()) / 2.0
            points_by_x.append((x, y_mid))
    if not points_by_x:
        return []
    points_by_x.sort(key=lambda p: p[0])
    if smooth_win >= 3 and len(points_by_x) >= smooth_win:
        xs = np.array([p[0] for p in points_by_x])
        ys = np.array([p[1] for p in points_by_x])
        kernel = np.ones(smooth_win) / smooth_win
        ys_smooth = np.convolve(ys, kernel, mode="same")
        points_by_x = list(zip(xs.tolist(), ys_smooth.tolist()))
    segments = []
    seg = [points_by_x[0]]
    for i in range(1, len(points_by_x)):
        if points_by_x[i][0] - points_by_x[i - 1][0] > gap_threshold:
            if len(seg) >= 2:
                segments.append(seg)
            seg = [points_by_x[i]]
        else:
            seg.append(points_by_x[i])
    if len(seg) >= 2:
        segments.append(seg)
    return segments


def tv_denoise_1d(y, weight):
    """TV-like denoising (L1 on total variation)."""
    if y is None:
        return y
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) < 3:
        return y
    return denoise_tv_chambolle(y, weight=weight)



def smooth_centerline_segment(seg_points, prev_seg_points=None):
    """Làm trơn centerline: TV-like theo không gian + EMA theo thời gian."""
    if not seg_points or len(seg_points) < 2:
        return seg_points

    xs = np.array([p[0] for p in seg_points], dtype=np.float64)
    ys = np.array([p[1] for p in seg_points], dtype=np.float64)

    ys_tv = tv_denoise_1d(ys, weight=_cfg().TV_WEIGHT)

    if prev_seg_points is not None and len(prev_seg_points) >= 2:
        prev_xs = np.array([p[0] for p in prev_seg_points], dtype=np.float64)
        prev_ys = np.array([p[1] for p in prev_seg_points], dtype=np.float64)
        # Interpolate previous line to current x-grid
        prev_ys_interp = np.interp(xs, prev_xs, prev_ys)

        # Clamp sudden y jumps to keep temporal consistency
        dy = ys_tv - prev_ys_interp
        dy = np.clip(dy, -_cfg().MAX_CENTERLINE_JUMP_PX, _cfg().MAX_CENTERLINE_JUMP_PX)
        ys_tv = prev_ys_interp + dy

        # EMA blend
        ys_tv = (1.0 - _cfg().TEMPORAL_ALPHA) * prev_ys_interp + _cfg().TEMPORAL_ALPHA * ys_tv

    return list(zip(xs.tolist(), ys_tv.tolist()))



def point_inside_mask(mask_2d, x, y):
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if yi < 0 or xi < 0:
        return False
    if yi >= mask_2d.shape[0] or xi >= mask_2d.shape[1]:
        return False
    return mask_2d[yi, xi] > 0



def project_along_dir_to_inside(mask_2d, center_point, dir_unit, target_offset_px):
    """Project a point back inside the mask along dir_unit."""
    cx, cy = float(center_point[0]), float(center_point[1])
    dx, dy = float(dir_unit[0]), float(dir_unit[1])

    max_d = int(round(float(target_offset_px)))
    if max_d < 0:
        return None

    # If already inside at target offset, keep it.
    x_t = cx + target_offset_px * dx
    y_t = cy + target_offset_px * dy
    if point_inside_mask(mask_2d, x_t, y_t):
        return (x_t, y_t)

    for d in range(max_d, -1, -1):
        x = cx + d * dx
        y = cy + d * dy
        if point_inside_mask(mask_2d, x, y):
            return (x, y)
    return None



def compute_parallel_lines(seg_points, pericardium_mask, offset_px):
    """Tạo 2 polyline song song (2 phía pháp tuyến), đảm bảo nằm trong pericardium mask."""
    if not seg_points or len(seg_points) < 2:
        return [], []

    pts = np.array(seg_points, dtype=np.float64)  # (N,2): x,y
    n = len(pts)

    # Tangents via finite differences => normals
    tangents = np.zeros_like(pts)
    for i in range(n):
        if i == 0:
            t = pts[i + 1] - pts[i]
        elif i == n - 1:
            t = pts[i] - pts[i - 1]
        else:
            t = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(t)
        if norm < 1e-8:
            tangents[i] = np.array([1.0, 0.0], dtype=np.float64)
        else:
            tangents[i] = t / norm

    # Normal in image coordinates: (-ty, tx)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    pos_points = []
    neg_points = []

    for i in range(n):
        cx, cy = pts[i]
        nx, ny = normals[i]

        # Positive side
        p = project_along_dir_to_inside(
            pericardium_mask,
            (cx, cy),
            (nx, ny),
            offset_px,
        )
        if p is not None:
            pos_points.append(p)

        # Negative side
        p = project_along_dir_to_inside(
            pericardium_mask,
            (cx, cy),
            (-nx, -ny),
            offset_px,
        )
        if p is not None:
            neg_points.append(p)

    return pos_points, neg_points



def compute_parallel_band_polygon(seg_points, pericardium_mask, halfwidth_px):
    """Tạo polygon vùng band giữa 2 biên song song (±halfwidth_px) nằm trong mask."""
    if not seg_points or len(seg_points) < 2:
        return None

    pts = np.array(seg_points, dtype=np.float64)  # (N,2): x,y
    n = len(pts)

    # Tangents via finite differences => normals
    tangents = np.zeros_like(pts)
    for i in range(n):
        if i == 0:
            t = pts[i + 1] - pts[i]
        elif i == n - 1:
            t = pts[i] - pts[i - 1]
        else:
            t = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(t)
        if norm < 1e-8:
            tangents[i] = np.array([1.0, 0.0], dtype=np.float64)
        else:
            tangents[i] = t / norm

    # Normal in image coordinates: (-ty, tx)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    pos_pts = []
    neg_pts = []

    for i in range(n):
        cx, cy = pts[i]
        nx, ny = normals[i]

        p_pos = project_along_dir_to_inside(
            pericardium_mask,
            (cx, cy),
            (nx, ny),
            halfwidth_px,
        )
        p_neg = project_along_dir_to_inside(
            pericardium_mask,
            (cx, cy),
            (-nx, -ny),
            halfwidth_px,
        )

        if p_pos is None or p_neg is None:
            continue

        pos_pts.append(p_pos)
        neg_pts.append(p_neg)

    if len(pos_pts) < 2 or len(neg_pts) < 2:
        return None

    # Polygon: pos theo thứ tự, neg theo thứ tự đảo để đóng
    poly = pos_pts + neg_pts[::-1]
    return poly



def fill_polygon_alpha(overlay_img, poly_points, color_bgr, alpha):
    """Fill polygon lên overlay với alpha (trộn màu nhẹ)."""
    if poly_points is None or len(poly_points) < 3:
        return

    h, w = overlay_img.shape[:2]
    mask_poly = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([(int(round(x)), int(round(y))) for x, y in poly_points], dtype=np.int32)
    pts = pts.reshape(-1, 1, 2)

    cv2.fillPoly(mask_poly, [pts], 255)

    color = np.array(color_bgr, dtype=np.float32)
    overlay_sel = overlay_img[mask_poly == 255].astype(np.float32)
    overlay_img[mask_poly == 255] = ((1.0 - alpha) * overlay_sel + alpha * color).astype(np.uint8)


def draw_solid_polyline(img, points, color, thickness):
    if not points or len(points) < 2:
        return
    pts = np.array([(int(round(x)), int(round(y))) for x, y in points], dtype=np.int32)
    pts = pts.reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)



def draw_dashed_polyline(img, points, color, thickness, dash_len, gap_len, step_px=2):
    """Vẽ polyline (đường mở) dạng nét đứt."""
    if not points or len(points) < 2:
        return
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    samples = []
    d_total = 0.0
    for i in range(n - 1):
        p1, p2 = pts[i], pts[i + 1]
        seg_len = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        if seg_len < 1e-6:
            continue
        num_steps = max(1, int(seg_len / step_px))
        for k in range(num_steps + 1):
            t = k / num_steps if num_steps > 0 else 1
            px = p1[0] + t * (p2[0] - p1[0])
            py = p1[1] + t * (p2[1] - p1[1])
            samples.append(((px, py), d_total))
            if k < num_steps:
                d_total += seg_len / num_steps
    if len(samples) < 2:
        return
    step = dash_len + gap_len
    in_dash = [((s[1] % step) < dash_len) for s in samples]
    for j in range(len(samples) - 1):
        if in_dash[j] and in_dash[j + 1]:
            p0 = (int(samples[j][0][0]), int(samples[j][0][1]))
            p1 = (int(samples[j + 1][0][0]), int(samples[j + 1][0][1]))
            cv2.line(img, p0, p1, color, thickness)


def draw_dashed_contour(img, contour, color, thickness, dash_len, gap_len, step_px=2):
    if contour is None or len(contour) < 2:
        return
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    samples = []
    d_total = 0.0
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        seg_len = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        if seg_len < 1e-6:
            continue
        num_steps = max(1, int(seg_len / step_px))
        for k in range(num_steps + 1):
            t = k / num_steps if num_steps > 0 else 1
            px = p1[0] + t * (p2[0] - p1[0])
            py = p1[1] + t * (p2[1] - p1[1])
            samples.append(((px, py), d_total))
            if k < num_steps:
                d_total += seg_len / num_steps
    if len(samples) < 2:
        return
    step = dash_len + gap_len
    in_dash = [((s[1] % step) < dash_len) for s in samples]
    for j in range(len(samples) - 1):
        if in_dash[j] and in_dash[j + 1]:
            p0 = (int(samples[j][0][0]), int(samples[j][0][1]))
            p1 = (int(samples[j + 1][0][0]), int(samples[j + 1][0][1]))
            cv2.line(img, p0, p1, color, thickness)


def compute_mask_centroid(bin_mask):
    m = cv2.moments(bin_mask.astype(np.uint8), binaryImage=True)
    if abs(m.get("m00", 0.0)) < 1e-6:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (float(cx), float(cy))



def draw_danger_triangle(img, anchor_xy, size=14):
    x, y = int(round(anchor_xy[0])), int(round(anchor_xy[1]))
    pts = np.array(
        [
            (x, y - size),
            (x - int(size * 0.9), y + size),
            (x + int(size * 0.9), y + size),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(img, [pts], _cfg().DANGER_TRIANGLE_COLOR)
    cv2.polylines(img, [pts], isClosed=True, color=_cfg().DANGER_TRIANGLE_OUTLINE, thickness=1)
    cv2.putText(img, "!", (x - 4, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



def draw_callout(img, from_xy, to_xy, text, color_arrow, color_text, bg_color, blink_on=True, prefix_danger=False):
    if from_xy is None:
        return
    p0 = (int(round(from_xy[0])), int(round(from_xy[1])))
    p1 = (int(round(to_xy[0])), int(round(to_xy[1])))

    # arrow
    cv2.arrowedLine(img, p0, p1, color_arrow, 1, tipLength=0.18)

    if not blink_on:
        return

    # text box
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, _cfg().CALLOUT_FONT_SCALE, _cfg().CALLOUT_THICKNESS)

    extra_left = 0
    if prefix_danger:
        extra_left = 22

    x0 = p1[0]
    y0 = p1[1]
    rect_pt1 = (x0 - _cfg().CALLOUT_PADDING, y0 - th - baseline - _cfg().CALLOUT_PADDING)
    rect_pt2 = (x0 + tw + _cfg().CALLOUT_PADDING + extra_left, y0 + _cfg().CALLOUT_PADDING)

    cv2.rectangle(img, rect_pt1, rect_pt2, bg_color, -1)
    cv2.rectangle(img, rect_pt1, rect_pt2, (200, 200, 200), 1)

    if prefix_danger:
        tri_anchor = (x0 + 12, y0 - th // 2)
        draw_danger_triangle(img, tri_anchor, size=10)
        text_org = (x0 + extra_left, y0)
    else:
        text_org = (x0, y0)

    cv2.putText(img, text, text_org, font, _cfg().CALLOUT_FONT_SCALE, color_text, _cfg().CALLOUT_THICKNESS, cv2.LINE_AA)



def draw_class_callouts(overlay, mask_resized, anchor_bin, frame_idx):
    """Arrow + text for CALLOUT_CLASS_IDS; blink triangle only for BLINK_WARNING_CLASS_ID."""
    H, W = overlay.shape[:2]
    callout_offset_px = float(_cfg().CALLOUT_OFFSET_CM) * float(_cfg().PIXELS_PER_CM)

    def clamp_xy(xy):
        x, y = float(xy[0]), float(xy[1])
        x = max(10.0, min(W - 10.0, x))
        y = max(20.0, min(H - 20.0, y))
        return (x, y)

    blink_on = True
    if _cfg().BLINK_WARNING_PERIOD_FRAMES is not None and _cfg().BLINK_WARNING_PERIOD_FRAMES > 0:
        blink_on = ((frame_idx // _cfg().BLINK_WARNING_PERIOD_FRAMES) % 2) == 0

    ids = list(_cfg().CALLOUT_CLASS_IDS)
    n_div = max(len(ids), 1)
    for j, c in enumerate(ids):
        if c == _cfg().GUIDELINE_ANCHOR_CLASS_ID:
            bin_mask = anchor_bin.astype(np.uint8)
        else:
            bin_mask = (mask_resized == c).astype(np.uint8)
        if int(bin_mask.sum()) < _cfg().CALLOUT_MIN_MASK_PIXELS:
            continue
        centroid = compute_mask_centroid(bin_mask)
        if centroid is None:
            continue
        cx, cy = float(centroid[0]), float(centroid[1])
        theta = (j * 2.0 * np.pi / n_div) - (np.pi / 2.0)
        r = callout_offset_px * (1.0 + 0.2 * j)
        label_pos = clamp_xy((cx + float(np.cos(theta)) * r, cy + float(np.sin(theta)) * r))
        name = _cfg().DISPLAY_CLASS_NAMES.get(c, _cfg().CLASS_NAMES[c] if 0 <= c < len(_cfg().CLASS_NAMES) else str(c))
        danger = _cfg().BLINK_WARNING_CLASS_ID is not None and c == _cfg().BLINK_WARNING_CLASS_ID
        if danger:
            draw_callout(
                overlay,
                centroid,
                label_pos,
                name,
                _cfg().CALLOUT_ARROW_COLOR,
                _cfg().CALLOUT_TEXT_COLOR,
                _cfg().CALLOUT_BG_COLOR,
                blink_on=blink_on,
                prefix_danger=True,
            )
        else:
            draw_callout(
                overlay,
                centroid,
                label_pos,
                name,
                _cfg().CALLOUT_ARROW_COLOR,
                _cfg().CALLOUT_TEXT_COLOR,
                _cfg().CALLOUT_BG_COLOR,
                blink_on=True,
                prefix_danger=False,
            )


def overlay_guideline(frame, mask, h, w, prev_centerline_seg=None, frame_idx=0):
    """Semi-transparent fill for OVERLAY_ALPHA_CLASS_IDS; geometry from GUIDELINE_ANCHOR_CLASS_ID."""
    overlay = frame.copy()
    mask_resized = cv2.resize(
        mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    for c in _cfg().OVERLAY_ALPHA_CLASS_IDS:
        if c < 0 or c >= len(_cfg().CLASS_COLORS_BGR):
            continue
        color = _cfg().CLASS_COLORS_BGR[c]
        sel = mask_resized == c
        overlay[sel] = (
            (1 - _cfg().ALPHA) * overlay[sel].astype(np.float32) + _cfg().ALPHA * np.array(color, dtype=np.float32)
        ).astype(np.uint8)

    anchor_bin = (mask_resized == _cfg().GUIDELINE_ANCHOR_CLASS_ID).astype(np.uint8)

    centerline_segments = get_pericardium_centerline(anchor_bin, gap_threshold=5, smooth_win=5)
    if not centerline_segments:
        geom_empty = {
            "main_centerline_segment_index": -1,
            "centerline_segments": [],
            "parallel_lines_offset_cm": {"positive_normal_side": [], "negative_normal_side": []},
            "band_polygons_between_parallel_bounds": [],
        }
        band_mask_empty = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        draw_class_callouts(overlay, mask_resized, anchor_bin, frame_idx)
        return overlay.astype(np.uint8), prev_centerline_seg, geom_empty, mask_resized, band_mask_empty

    main_idx = int(np.argmax([len(s) for s in centerline_segments]))
    new_prev_centerline_seg = None

    for idx, seg in enumerate(centerline_segments):
        if idx == main_idx:
            centerline_segments[idx] = smooth_centerline_segment(seg, prev_seg_points=prev_centerline_seg)
            new_prev_centerline_seg = centerline_segments[idx]
        else:
            centerline_segments[idx] = smooth_centerline_segment(seg, prev_seg_points=None)

    offset_px = float(_cfg().OFFSET_CM) * float(_cfg().PIXELS_PER_CM)
    band_halfwidth_px = offset_px * (_cfg().BAND_WIDTH_SCALE / 2.0)
    band_mask = np.zeros((overlay.shape[0], overlay.shape[1]), dtype=np.uint8)
    for seg in centerline_segments:
        band_poly = compute_parallel_band_polygon(seg, anchor_bin, band_halfwidth_px)
        fill_polygon_alpha(overlay, band_poly, _cfg().BAND_COLOR_BGR, _cfg().BAND_ALPHA)
        if band_poly is not None and len(band_poly) >= 3:
            pts = np.array(
                [(int(round(x)), int(round(y))) for x, y in band_poly], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.fillPoly(band_mask, [pts], 255)

    for seg in centerline_segments:
        draw_dashed_polyline(overlay, seg, _cfg().GUIDELINE_COLOR_BGR, _cfg().LINE_THICKNESS, _cfg().DASH_LEN, _cfg().GAP_LEN)

    draw_class_callouts(overlay, mask_resized, anchor_bin, frame_idx)

    export_segments = [[[float(x), float(y)] for x, y in seg] for seg in centerline_segments]
    export_band_polys = []
    for seg in centerline_segments:
        poly = compute_parallel_band_polygon(seg, anchor_bin, band_halfwidth_px)
        if poly is not None:
            export_band_polys.append([[float(x), float(y)] for x, y in poly])
        else:
            export_band_polys.append([])

    if len(centerline_segments[main_idx]) >= 2:
        pos_pts, neg_pts = compute_parallel_lines(
            centerline_segments[main_idx], anchor_bin, offset_px
        )
        pos_pl = [[float(x), float(y)] for x, y in pos_pts]
        neg_pl = [[float(x), float(y)] for x, y in neg_pts]
    else:
        pos_pl, neg_pl = [], []

    geom = {
        "main_centerline_segment_index": int(main_idx),
        "centerline_segments": export_segments,
        "parallel_lines_offset_cm": {
            "positive_normal_side": pos_pl,
            "negative_normal_side": neg_pl,
        },
        "band_polygons_between_parallel_bounds": export_band_polys,
    }
    return overlay.astype(np.uint8), new_prev_centerline_seg, geom, mask_resized, band_mask