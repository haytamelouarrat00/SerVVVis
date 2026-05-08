import numpy as np
from PIL import Image
import cv2


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
