"""Real-image dataset for Modules A and B.

Turns a folder of photos / drawings into the same dictionary that datasets.shapes.make_dataset returns,
so b_train.py and a_train.py can use it unchanged:

    {"x": (N,1,H,W) float32 in [0,1], "contour": (N,1,H,W), "y": (N,) int64, "meta": [...], "classes": [...]}

Folder layout (class = sub-folder name):
    data_dir/
        cat/ img1.jpg img2.png ...
        dog/ ...
  or, if you want to choose the validation images yourself:
    data_dir/train/<class>/...   and   data_dir/val/<class>/...

Pipeline per image:  load -> isolate object (alpha channel, else GrabCut, else Otsu) -> clean mask
                     (largest blob, holes filled) -> optional rotation -> crop to object -> rescale so the
                     object spans scale * field_deg degrees -> place on a black 256x256 canvas with a random shift.
The object is white (1.0) on black (0.0), exactly like the synthetic shapes. Colour and background are dropped.

Train / validation split is made per SOURCE image, so augmented copies of one photo never land in both sets.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_WORK = 256          # segmentation is done at this size (longest side) for speed


# ----------------------------------------------------------------------------------- segmentation
def _clean_mask(m: np.ndarray) -> np.ndarray:
    """Binary uint8 mask -> largest connected blob, small gaps closed, holes filled."""
    m = (m > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)           # fills interior holes
    return filled


def _mask_ok(m: np.ndarray, lo: float = 0.01, hi: float = 0.90) -> bool:
    frac = float(m.mean())
    return lo <= frac <= hi


def _otsu_mask(gray: np.ndarray) -> np.ndarray:
    """Fallback: threshold, choosing the polarity so that the border is background."""
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = (th > 0).astype(np.uint8)
    border = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]])
    return (1 - th) if border.mean() > 0.5 else th


def _flat_bg_mask(img: np.ndarray) -> Optional[np.ndarray]:
    """Fast path for clean images (drawings, synthetic shapes, product shots): if the image border is one
    uniform colour, the object is everything that differs from that colour. ~100x faster than GrabCut and exact
    for flat backgrounds. Returns None when the border is not uniform (object touches the edge, photo, ...)."""
    h, w = img.shape[:2]
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
    med = np.median(border, axis=0)
    if (np.abs(border - med).max(axis=1) <= 12).mean() < 0.98:
        return None
    diff = np.abs(cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32) - med).max(axis=2)
    return _clean_mask((diff > 30).astype(np.uint8))


def to_silhouette(img: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """img: BGR / BGRA / grey uint8 array. Returns a float32 {0,1} silhouette (same size as the working
    resolution, longest side <= 256) or None if no usable object was found.
    `mask` (optional): a ready-made object mask for this image (any size, >0 = object); skips segmentation."""
    if img is None:
        return None
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]; img = img[:, :, :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = _WORK / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    m = None
    if mask is not None:
        m = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    elif alpha is not None and alpha.min() < 250:                    # transparent background (PNG cut-out)
        m = cv2.resize((alpha > 127).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if m is not None:
        m = _clean_mask(m)
        return m.astype(np.float32) if _mask_ok(m, 0.001, 0.98) else None

    fm = _flat_bg_mask(img)                                          # 0) uniform background: exact and fast
    if fm is not None and _mask_ok(fm):
        return fm.astype(np.float32)

    # 1) GrabCut, initialised with a rectangle that leaves a thin border as sure background
    cand = []
    try:
        gm = np.zeros((h, w), np.uint8)
        bx, by = max(2, w // 20), max(2, h // 20)
        rect = (bx, by, w - 2 * bx, h - 2 * by)
        cv2.grabCut(img, gm, rect, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
        cand.append(_clean_mask(np.where((gm == 1) | (gm == 3), 1, 0)))
    except cv2.error:
        pass
    # 2) Otsu threshold fallback
    cand.append(_clean_mask(_otsu_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
    for c in cand:
        if _mask_ok(c):
            return c.astype(np.float32)
    return None


# ----------------------------------------------------------------------------------- placement
def place_object(sil: np.ndarray, spec, rng: np.random.Generator, scale: Optional[float] = None,
                 rotate: bool = True, angle: Optional[float] = None, shift: bool = True) -> Tuple[np.ndarray, dict]:
    """Rotate, crop, rescale and position a silhouette on a (H, W) canvas. Returns (canvas, meta)."""
    H, W = spec.image_shape
    if angle is None:
        angle = float(rng.uniform(0, 360)) if rotate else 0.0
    if angle:
        d = int(math.ceil(math.hypot(*sil.shape)))
        pad = np.zeros((d, d), np.float32)
        r0, c0 = (d - sil.shape[0]) // 2, (d - sil.shape[1]) // 2
        pad[r0:r0 + sil.shape[0], c0:c0 + sil.shape[1]] = sil
        M = cv2.getRotationMatrix2D((d / 2, d / 2), angle, 1.0)
        sil = (cv2.warpAffine(pad, M, (d, d), flags=cv2.INTER_LINEAR) > 0.5).astype(np.float32)
    ys, xs = np.where(sil > 0.5)
    crop = sil[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if scale is None:
        scale = float(rng.uniform(*spec.scale_range))
    target = scale * spec.field_deg * spec.deg2pix                   # longest side of the object, pixels
    target = min(target, min(H, W) - 6)
    f = target / max(crop.shape)
    nh, nw = max(1, int(round(crop.shape[0] * f))), max(1, int(round(crop.shape[1] * f)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    crop = np.clip(crop, 0.0, 1.0)
    j = spec.max_shift_deg * spec.deg2pix if shift else 0.0
    r0 = (H - nh) // 2 + (int(round(rng.uniform(-j, j))) if j else 0)
    c0 = (W - nw) // 2 + (int(round(rng.uniform(-j, j))) if j else 0)
    r0 = int(np.clip(r0, 0, H - nh)); c0 = int(np.clip(c0, 0, W - nw))
    canvas = np.zeros((H, W), np.float32)
    canvas[r0:r0 + nh, c0:c0 + nw] = crop
    return canvas, {"scale": scale, "angle": angle, "centre_rc": [r0 + nh / 2, c0 + nw / 2]}


def _outline(canvas: np.ndarray) -> np.ndarray:
    m = (canvas > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, thickness=1)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------------- dataset
def _list_classes(root: Path) -> List[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(".")


def _images_in(folder: Path) -> List[Path]:
    """All image files under `folder`, searching sub-folders too (so extra nesting does not hide images)."""
    return sorted(p for p in folder.rglob("*") if _is_img(p))


def _describe(root: Path, n: int = 8) -> str:
    """Short text description of what is inside `root`, for error messages."""
    files = [p for p in root.rglob("*") if p.is_file()]
    ext = Counter(p.suffix.lower() or "<no extension>" for p in files)
    subs = _list_classes(root)
    lines = [f"  looked in: {root.resolve()}",
             f"  sub-folders: {subs[:n]}{' ...' if len(subs) > n else ''}",
             f"  files found (any type): {len(files)}; by extension: {dict(ext.most_common(8))}",
             f"  supported image extensions: {sorted(EXTS)}"]
    return "\n".join(lines)


def _flat_class(stem: str) -> str:
    """'circle_012' / 'Circle-3' / 'circle (4)' / 'circle12' -> 'circle'."""
    c = re.sub(r"[\s_\-]*\(?\d+\)?$", "", stem).strip(" _-").lower()
    return c or stem.lower()


def _resolve_root(root: Path, log=print) -> Path:
    """Step through 'wrapper' folders: a folder with no images of its own and exactly one sub-folder
    (e.g. archive/ -> 2D_Geometric_Shapes_Dataset/) is not a class, it only contains the dataset."""
    while True:
        direct = [f for f in root.iterdir() if _is_img(f)]
        subs = _list_classes(root)
        if direct or len(subs) != 1:
            return root
        log(f"  (stepping into wrapper folder '{subs[0]}')")
        root = root / subs[0]


def _collect(root: Path, log=print) -> Dict[str, List[Path]]:
    """class name -> list of image files. Handles (a) one sub-folder per class (images may be nested deeper),
    (b) a flat folder where the class is the file-name prefix ('circle_001.png')."""
    root = _resolve_root(root, log)
    direct = [f for f in sorted(root.iterdir()) if _is_img(f)]
    subs = _list_classes(root)
    out: Dict[str, List[Path]] = {}
    if direct and not subs:                                         # flat folder
        for f in direct:
            out.setdefault(_flat_class(f.stem), []).append(f)
        log(f"  flat folder: classes inferred from file names -> " + ", ".join(f"{c} ({len(v)})" for c, v in sorted(out.items())))
        if len(out) < 2 or len(out) > max(50, len(direct) // 2):
            raise RuntimeError("Could not infer sensible classes from the file names in a flat folder.\n"
                               "Put each class in its own sub-folder (data_dir/<class>/*.png).\n" + _describe(root))
        return out
    for c in subs:
        files = _images_in(root / c)
        if files:
            out[c] = files
        else:
            log(f"  class '{c}': no image files found inside (ignored)")
    if direct and subs:
        log(f"  note: {len(direct)} image(s) sit directly in {root.name}/ next to class folders; they are ignored.")
    if not out:
        raise RuntimeError("No image files found in any class folder.\n" + _describe(root))
    return out


def _load_silhouettes(files: Dict[str, List[Path]], log=print, every: int = 200) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    out: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    total = sum(len(v) for v in files.values())
    done, t0 = 0, time.time()
    for c, paths in files.items():
        items, skipped, unreadable = [], 0, 0
        for f in paths:
            raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if raw is None:
                unreadable += 1
            else:
                sil = to_silhouette(raw)
                if sil is None:
                    skipped += 1
                else:
                    items.append((f.name, sil))
            done += 1
            if done % every == 0:
                el = time.time() - t0
                log(f"    ... {done}/{total} images segmented, {el:.0f}s elapsed, ~{el / done * (total - done):.0f}s left")
        out[c] = items
        extra = (f", {skipped} skipped (no clear object found)" if skipped else "") + \
                (f", {unreadable} unreadable by OpenCV" if unreadable else "")
        log(f"  class '{c}': {len(items)} usable of {len(paths)} images" + extra)
    return out


def _cap(files: Dict[str, List[Path]], n: Optional[int], rng: np.random.Generator, log=print) -> Dict[str, List[Path]]:
    """Randomly keep at most n images per class (reproducible; keeps the order of the files otherwise)."""
    if not n:
        return files
    out = {}
    for c, v in files.items():
        if len(v) > n:
            log(f"  class '{c}': using a random {n} of {len(v)} images (--max-per-class)")
            keep = sorted(rng.choice(len(v), size=n, replace=False))
            v = [v[i] for i in keep]
        out[c] = v
    return out


def _build(sils: Dict[str, List[Tuple[str, np.ndarray]]], classes: Sequence[str], spec, n_aug: int, seed: int,
           rotate: bool, keep_x: bool = True, keep_contour: bool = True) -> Dict:
    """Memory-lean: arrays are allocated once at their final size and filled in shuffled order (the old version
    built lists, stacked them and then copied again to shuffle: ~3x the final size at peak). keep_x / keep_contour=False
    skips an array nobody will read (the entry is then None)."""
    rng = np.random.default_rng(seed)
    H, W = spec.image_shape
    items = [(ci, c, name, sil) for ci, c in enumerate(classes) for name, sil in sils[c]]
    n = len(items) * n_aug
    if n == 0:
        raise RuntimeError("No usable images found: files were found but object segmentation failed for all of them "
                           "(or the split left a set empty). Run scripts/check_dataset.py --dir <your data dir> --save-debug debug/ "
                           "to see the masks.")
    perm = np.random.default_rng([seed, 7]).permutation(n)          # final position of the i-th generated image
    x = np.zeros((n, 1, H, W), np.float32) if keep_x else None
    cn = np.zeros((n, 1, H, W), np.float32) if keep_contour else None
    y = np.zeros(n, np.int64); meta: list = [None] * n
    i = 0
    for ci, c, name, sil in items:
        for a in range(n_aug):
            canvas, m = place_object(sil, spec, rng, rotate=rotate)
            q = int(perm[i]); i += 1
            if keep_x:
                x[q, 0] = canvas
            if keep_contour:
                cn[q, 0] = _outline(canvas)
            y[q] = ci
            meta[q] = {"class": c, "source": name, "aug": a, **m}
    return {"x": x, "contour": cn, "y": y, "meta": meta, "classes": list(classes)}


def make_real_datasets(data_dir: str | Path, spec, val_frac: float = 0.2, n_aug_train: int = 5, n_aug_val: int = 1,
                       seed: int = 0, rotate: bool = True, max_per_class: Optional[int] = None, log=print,
                       keep_x: bool = True, keep_contour: bool = True) -> Tuple[Dict, Dict]:
    """Returns (train, val) dictionaries. See the module docstring for the folder layout.
    keep_x=False / keep_contour=False leave that array out (None) to save memory when a script never reads it."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"data dir not found: {root}")
    rng = np.random.default_rng(seed)
    root = _resolve_root(root, log)
    if (root / "train").is_dir() and (root / "val").is_dir():
        log("Segmenting train images ...");  f_tr = _cap(_collect(root / "train", log), max_per_class, rng, log)
        log("Segmenting val images ...");    f_va = _cap(_collect(root / "val", log), max_per_class, rng, log)
        classes = sorted(f_tr)
        s_tr = _load_silhouettes(f_tr, log); s_va = _load_silhouettes({c: f_va.get(c, []) for c in classes}, log)
    else:
        log("Segmenting images ...")
        files = _cap(_collect(root, log), max_per_class, rng, log)
        classes = sorted(files)
        allsil = _load_silhouettes(files, log)
        s_tr, s_va = {}, {}
        for c in classes:                                            # split by source image, per class
            items = list(allsil[c]); order = rng.permutation(len(items))
            n_val = max(1, int(round(val_frac * len(items)))) if len(items) > 1 else 0
            s_va[c] = [items[i] for i in order[:n_val]]
            s_tr[c] = [items[i] for i in order[n_val:]]
    empty = [c for c in classes if not s_tr.get(c)]
    if empty:
        log(f"WARNING: no usable training images for classes {empty}; they will be missing from training.")
    tr = _build(s_tr, classes, spec, n_aug_train, seed + 1, rotate, keep_x, keep_contour)
    va = _build(s_va, classes, spec, n_aug_val, seed + 2, rotate, keep_x, keep_contour)
    mb = sum(d[k].nbytes for d in (tr, va) for k in ("x", "contour") if d[k] is not None) / 1e6
    log(f"Train: {len(tr['y'])} images, val: {len(va['y'])} images, classes: {classes}  (~{mb:.0f} MB in memory)")
    return tr, va


def make_real_test(data_dir: str | Path, spec, classes: Sequence[str], n_aug: int = 1, seed: int = 0,
                   rotate: bool = True, max_per_class: Optional[int] = None, log=print) -> Dict:
    """Held-out TEST set from data_dir/test/<class>/... with exactly the pipeline (and class order) of the training set.
    Create the folder with scripts/split_dataset.py. Never used for training or model selection."""
    root = _resolve_root(Path(data_dir), log)
    if not (root / "test").is_dir():
        raise FileNotFoundError(f"{root}/test not found. Run scripts/split_dataset.py to create train/ val/ test/.")
    rng = np.random.default_rng(seed)
    log("Segmenting test images ...")
    files = _cap(_collect(root / "test", log), max_per_class, rng, log)
    extra = sorted(set(files) - set(classes))
    if extra:
        log(f"WARNING: test classes not in the training classes are ignored: {extra}")
    sils = _load_silhouettes({c: files.get(c, []) for c in classes}, log)
    te = _build(sils, list(classes), spec, n_aug, seed + 3, rotate)
    log(f"Test: {len(te['y'])} images, classes: {list(classes)}")
    return te
