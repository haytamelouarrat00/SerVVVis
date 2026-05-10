import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def save_side_by_side(img1, img2, path):
    out = np.concatenate([img1, img2], axis=1)
    out_uint8 = (out * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(out_uint8).save(path)


def save_match_visualization(
    img1,
    img2,
    kpts1_removed,
    kpts2_removed,
    matches_removed,
    kpts1_kept,
    kpts2_kept,
    matches_kept,
    path,
    draw_removed=True,
):
    W1 = img1.shape[1]
    canvas = np.concatenate([img1, img2], axis=1)
    canvas = (canvas * 255.0).clip(0, 255).astype(np.uint8)
    canvas = np.ascontiguousarray(canvas)

    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    GREEN_ALPHA = 0.5
    RED_ALPHA = 0.25

    def endpoints(src_kpts1, src_kpts2, i, j):
        p1 = (int(round(float(src_kpts1[i, 0]))), int(round(float(src_kpts1[i, 1]))))
        p2 = (
            int(round(float(src_kpts2[j, 0]))) + W1,
            int(round(float(src_kpts2[j, 1]))),
        )
        return p1, p2

    if draw_removed:
        red_overlay = canvas.copy()
        for i, j in matches_removed:
            p1, p2 = endpoints(kpts1_removed, kpts2_removed, i, j)
            cv2.line(red_overlay, p1, p2, RED, 1, lineType=cv2.LINE_AA)
        cv2.addWeighted(red_overlay, RED_ALPHA, canvas, 1.0 - RED_ALPHA, 0, canvas)

    green_overlay = canvas.copy()
    for i, j in matches_kept:
        p1, p2 = endpoints(kpts1_kept, kpts2_kept, i, j)
        cv2.line(green_overlay, p1, p2, GREEN, 1, lineType=cv2.LINE_AA)

    for i, j in matches_kept:
        p1, p2 = endpoints(kpts1_kept, kpts2_kept, i, j)
        cv2.circle(green_overlay, p1, 3, GREEN, 1, lineType=cv2.LINE_AA)
        cv2.circle(green_overlay, p2, 3, GREEN, 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(green_overlay, GREEN_ALPHA, canvas, 1.0 - GREEN_ALPHA, 0, canvas)

    Image.fromarray(canvas).save(path)


def save_error_evolution(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width = 1100
    panel_height = 210
    margin_left = 95
    margin_right = 35
    margin_top = 42
    margin_bottom = 38
    gap = 18

    metrics = [
        (
            "feature error norm",
            lambda item: item.get("controller_info", {}).get("residual_norm"),
            (30, 110, 230),
        ),
        (
            "pose distance (m)",
            lambda item: item.get("translation_error_m"),
            (35, 150, 85),
        ),
        (
            "rotation distance (deg)",
            lambda item: item.get("rotation_error_deg"),
            (210, 95, 35),
        ),
    ]

    height = margin_top + margin_bottom + len(metrics) * panel_height + (len(metrics) - 1) * gap
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    title = "Servo error evolution"
    cv2.putText(
        canvas,
        title,
        (margin_left, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        lineType=cv2.LINE_AA,
    )

    def metric_points(get_value):
        points = []
        for item in history:
            value = get_value(item)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            points.append((int(item["iteration"]), value))
        return points

    plot_left = margin_left
    plot_right = width - margin_right
    plot_width = plot_right - plot_left

    for metric_index, (label, get_value, color) in enumerate(metrics):
        y0 = margin_top + metric_index * (panel_height + gap)
        y1 = y0 + panel_height
        plot_top = y0 + 28
        plot_bottom = y1 - 34
        plot_height = plot_bottom - plot_top

        cv2.putText(
            canvas,
            label,
            (plot_left, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 35, 35),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (plot_left, plot_top),
            (plot_right, plot_bottom),
            (210, 210, 210),
            1,
            lineType=cv2.LINE_AA,
        )

        points = metric_points(get_value)
        if not points:
            cv2.putText(
                canvas,
                "no data",
                (plot_left + 12, plot_top + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (120, 120, 120),
                1,
                lineType=cv2.LINE_AA,
            )
            continue

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = min(0.0, float(ys.min()))
        y_max = float(ys.max())
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        y_pad = 0.05 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad

        cv2.putText(
            canvas,
            f"{y_max:.4g}",
            (8, plot_top + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{y_min:.4g}",
            (8, plot_bottom + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"iter {int(x_min)}",
            (plot_left, plot_bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"iter {int(x_max)}",
            (plot_right - 75, plot_bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            lineType=cv2.LINE_AA,
        )

        pixel_points = []
        for x, y in points:
            px = plot_left + int(round(((x - x_min) / (x_max - x_min)) * plot_width))
            py = plot_bottom - int(round(((y - y_min) / (y_max - y_min)) * plot_height))
            pixel_points.append([px, py])
        pixel_points = np.asarray(pixel_points, dtype=np.int32).reshape(-1, 1, 2)

        if len(pixel_points) >= 2:
            cv2.polylines(
                canvas,
                [pixel_points],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        for point in pixel_points.reshape(-1, 2):
            cv2.circle(canvas, tuple(point), 3, color, -1, lineType=cv2.LINE_AA)

        first = points[0][1]
        last = points[-1][1]
        cv2.putText(
            canvas,
            f"{first:.5g} -> {last:.5g}",
            (plot_right - 170, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    Image.fromarray(canvas).save(path)
