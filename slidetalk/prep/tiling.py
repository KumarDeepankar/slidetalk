"""Tissue segmentation + tiling, following the Prov-GigaPath preprocessing recipe:
Otsu on a low-res thumbnail, resample to 0.5 µm/px (20x), 256x256 tiles, drop tiles with <10% tissue."""
from __future__ import annotations

import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET_MPP = 0.5
TILE_PX = 256  # at target mpp


def slide_mpp(slide: openslide.OpenSlide) -> float:
    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        return float(mpp)
    mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    return 10.0 / float(mag) if mag else 0.25  # 40x ~ 0.25 µm/px


def tissue_mask(slide: openslide.OpenSlide, thumb_size: int = 2048):
    """Otsu threshold on HSV saturation of a thumbnail (as in HIPT/CLAM/Prov-GigaPath)."""
    thumb = np.asarray(slide.get_thumbnail((thumb_size, thumb_size)).convert("RGB"))
    sat = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[..., 1]
    sat = cv2.medianBlur(sat, 7)
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # drop pen marks / very dark artefacts
    dark = thumb.mean(-1) < 40
    mask[dark] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    scale = slide.dimensions[0] / thumb.shape[1]
    return mask > 0, scale


def tile_coords(slide: openslide.OpenSlide, min_tissue: float = 0.1):
    """Returns level-0 (x, y) of each tissue tile, the level-0 tile size, and the tile grid indices."""
    mpp = slide_mpp(slide)
    tile0 = int(round(TILE_PX * TARGET_MPP / mpp))  # tile side in level-0 pixels
    mask, scale = tissue_mask(slide)
    W, H = slide.dimensions
    coords, grid = [], []
    for gy, y in enumerate(range(0, H - tile0 + 1, tile0)):
        for gx, x in enumerate(range(0, W - tile0 + 1, tile0)):
            x0, y0 = int(x / scale), int(y / scale)
            x1, y1 = max(int((x + tile0) / scale), x0 + 1), max(int((y + tile0) / scale), y0 + 1)
            if mask[y0:y1, x0:x1].mean() >= min_tissue:
                coords.append((x, y))
                grid.append((gx, gy))
    return np.array(coords, dtype=np.int64).reshape(-1, 2), tile0, np.array(grid, dtype=np.int64).reshape(-1, 2)


class TileDataset(Dataset):
    """Reads tiles lazily; each DataLoader worker opens its own OpenSlide handle."""

    def __init__(self, path, coords, tile0, transform):
        self.path, self.coords, self.tile0, self.transform = str(path), coords, tile0, transform
        self._slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.path)
        s = self._slide
        level = s.get_best_level_for_downsample(self.tile0 / TILE_PX + 1e-3)
        ds = s.level_downsamples[level]
        size = int(round(self.tile0 / ds))
        x, y = self.coords[i]
        img = s.read_region((int(x), int(y)), level, (size, size)).convert("RGB")
        if size != TILE_PX:
            img = img.resize((TILE_PX, TILE_PX), Image.BILINEAR)
        return self.transform(img)
