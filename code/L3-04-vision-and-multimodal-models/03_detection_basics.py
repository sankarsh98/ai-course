"""Detection bookkeeping from scratch: IoU, non-maximum suppression and average precision.

A simulated detector fires several jittered boxes around each true object (duplicates) and
some boxes on background, each with a confidence score. We then measure how the NMS
threshold and the IoU threshold used for matching change average precision (AP).
Boxes are (x1, y1, x2, y2).
"""
import numpy as np

rng = np.random.default_rng(0)


def iou(a, b):
    """IoU between every box in a (n, 4) and every box in b (m, 4)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def nms(boxes, scores, thresh):
    """Greedy NMS: keep the best box, drop boxes overlapping it by more than thresh, repeat."""
    order, keep = np.argsort(-scores), []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][iou(boxes[[i]], boxes[order[1:]])[0] <= thresh]
    return np.array(keep, dtype=int)


def average_precision(dets, gts, iou_thr):
    """dets: list over images of (boxes, scores); gts: list of true boxes. One class."""
    records, n_gt = [], sum(len(g) for g in gts)
    for (boxes, scores), g in zip(dets, gts):
        matched = np.zeros(len(g), bool)
        for j in np.argsort(-scores):                  # greedy, highest score first
            ious = iou(boxes[[j]], g)[0] if len(g) else np.array([])
            k = int(ious.argmax()) if len(ious) else -1
            tp = k >= 0 and ious[k] >= iou_thr and not matched[k]
            if tp:
                matched[k] = True                      # a second hit on the same object is a FP
            records.append((scores[j], tp))
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records]); fp = np.cumsum([not r[1] for r in records])
    recall, precision = tp / n_gt, tp / (tp + fp)
    # Area under the precision-recall curve, with precision made monotone (as in VOC/COCO).
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    r = np.concatenate([[0], recall])
    return float(np.sum((r[1:] - r[:-1]) * precision))


def simulate(n_images=200):
    gts, dets = [], []
    for _ in range(n_images):
        n_obj = rng.integers(1, 5)
        xy = rng.uniform(0, 80, (n_obj, 2)); wh = rng.uniform(10, 30, (n_obj, 2))
        g = np.hstack([xy, xy + wh])
        boxes, scores = [], []
        for b in g:
            if rng.random() < 0.1:                      # the detector misses 10% of objects
                continue
            for _ in range(rng.integers(2, 6)):        # several overlapping firings per object
                jitter = rng.normal(0, 0.08, 4) * np.r_[b[2:] - b[:2], b[2:] - b[:2]]
                boxes.append(b + jitter); scores.append(rng.uniform(0.5, 1.0))
        for _ in range(rng.integers(0, 3)):            # background false positives
            xy0 = rng.uniform(0, 80, 2)
            boxes.append(np.r_[xy0, xy0 + rng.uniform(10, 30, 2)]); scores.append(rng.uniform(0, 0.8))
        gts.append(g); dets.append((np.array(boxes).reshape(-1, 4), np.array(scores)))
    return gts, dets


a = np.array([[0, 0, 10, 10.]]); b = np.array([[5, 0, 15, 10.], [1, 1, 11, 11.], [20, 20, 30, 30.]])
print("IoU of a 10x10 box with: shifted half its width, shifted by 1 px diagonally, disjoint:",
      iou(a, b)[0].round(3))

gts, raw = simulate()
print(f"\n{len(gts)} images, {sum(len(g) for g in gts)} objects, {sum(len(d[0]) for d in raw)} raw detections")
print(f"{'NMS threshold':>14} {'boxes kept':>10} {'AP@0.5':>7} {'AP@0.75':>8} {'AP@[.5:.95]':>12}")
for thr in (None, 0.3, 0.5, 0.7, 0.9):
    dets = raw if thr is None else [(bx[nms(bx, sc, thr)], sc[nms(bx, sc, thr)]) for bx, sc in raw]
    aps = [average_precision(dets, gts, t) for t in np.arange(0.5, 0.96, 0.05)]
    print(f"{'none' if thr is None else thr:>14} {sum(len(d[0]) for d in dets):>10} {aps[0]:>7.3f}"
          f" {aps[5]:>8.3f} {np.mean(aps):>12.3f}")
