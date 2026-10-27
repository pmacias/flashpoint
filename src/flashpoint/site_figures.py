"""Helpers for the website/portfolio figures (notebooks/07_site_figures.ipynb).

VISUALIZATION ONLY -- NOT FOR FEATURES OR MODELS. Everything here reads the
FULL fire trajectory (index BUFFER_DAYS through n_days) to show what each
fire eventually did. That is exactly the information the project's early-
window rule forbids model inputs from seeing, so nothing in `features.py`,
`rasters.py`, `cnn.py`, or any modeling notebook may import from this
module.

Day numbering here is 1-indexed over the fire's actual days: "day 1" is
stored index BUFFER_DAYS, matching how the README and notebooks talk about
the day-1/day-2 cutoff window.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from flashpoint.data_access import (
    ACTIVE_FIRE_CHANNEL_IDX, BUFFER_DAYS, CHANNEL_NAMES, PIXEL_SIDE_M, HDF5Event,
    read_channel_all_days,
)
from flashpoint.rasters import early_fire_center, extract_crop

CUTOFF_DAY = 2

# Site palette (steel-blue page background, coral highlight).
SITE_BG = "#284a72"
SITE_TEXT = "#f3f6fa"
SITE_TEXT_MUTED = "#c9d6e6"
SITE_TICK = "#8ea6c3"
SITE_GRID = "#3d6290"
ACCENT = "#ff9273"
SITE_RCPARAMS = {
    "figure.facecolor": SITE_BG, "axes.facecolor": SITE_BG, "savefig.facecolor": SITE_BG,
    "text.color": SITE_TEXT, "axes.labelcolor": SITE_TEXT_MUTED,
    "xtick.color": SITE_TICK, "ytick.color": SITE_TICK,
    "axes.edgecolor": SITE_TICK, "grid.color": SITE_GRID,
}

# Arrival-time ramp: pale yellow -> pale pink -> magenta -> violet. It
# routes through pink rather than orange/red so no step comes near ACCENT
# (coral), which is reserved for the model's days 1-2; checked in notebook 07
# with an OKLab distance. The darkest stop stays clear of #284a72.
ARRIVAL_RAMP = ["#fff3b0", "#fcd9e0", "#f08bc4", "#c9459f", "#a23aa8"]

# Fuel-dryness ramp for the driver map: muted slate (moist) -> pale sand
# (dry). Single hue family, monotone in lightness, no reds, so neither the
# coral fire pixels nor the arrival ramp can be confused with it.
DRYNESS_RAMP = ["#4f6b84", "#8f9a8f", "#c9b98d", "#f2e2b0"]

# Muted-blue terrain ramp for the hillshade (shadow -> lit), centered on the
# page background so the terrain sits behind the fire rather than beside it.
HILLSHADE_RAMP = ["#1a3556", "#284a72", "#4b6e98"]


def event_from_row(row) -> HDF5Event:
    """Build an HDF5Event from an `events` table row (namedtuple or Series)."""
    return HDF5Event(
        event_id=row.event_id, year=int(row.year), hdf5_path=Path(row.hdf5_path),
        n_days=int(row.n_days), img_dates=[], lnglat=(row.centroid_lon, row.centroid_lat),
    )


def read_fire_days(event: HDF5Event) -> np.ndarray:
    """Binary active-fire masks for the fire's actual days -> (n_fire_days, H, W).

    Full-trajectory read (index BUFFER_DAYS onward); see module docstring.
    """
    # Single-channel read, so binarize directly (same `> 0` rule as
    # labels.active_fire_mask, which expects a full 23-channel day raster).
    hours = read_channel_all_days(event, ACTIVE_FIRE_CHANNEL_IDX)[BUFFER_DAYS:]
    return hours > 0


def read_elevation(event: HDF5Event) -> np.ndarray:
    """Elevation plane (m) from day 1. Static channel, so one day suffices."""
    with h5py.File(event.hdf5_path, "r") as f:
        return f["data"][BUFFER_DAYS, CHANNEL_NAMES.index("elevation")].astype(np.float32)


def arrival_day_map(fire_days: np.ndarray) -> np.ndarray:
    """1-indexed day each pixel was first detected; 0 where never detected."""
    ever = fire_days.any(axis=0)
    first = fire_days.argmax(axis=0) + 1
    return np.where(ever, first, 0).astype(np.int32)


def hillshade(elev: np.ndarray, cell_m: float = PIXEL_SIDE_M,
              azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """Standard (Horn-style gradient) hillshade in [0, 1]; NaN-tolerant.

    Rows run north -> south, so d/dy is negated to get a north-positive slope.
    """
    filled = np.where(np.isfinite(elev), elev, np.nanmean(elev))
    dz_dy, dz_dx = np.gradient(filled, cell_m)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    zenith = np.deg2rad(90.0 - altitude_deg)
    az = np.deg2rad(360.0 - azimuth_deg + 90.0)
    shade = (np.cos(zenith) * np.cos(slope)
             + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0.0, 1.0)


def scan_events(events: pd.DataFrame, crop_size: int = 128) -> pd.DataFrame:
    """Per-event early/final fire stats for picking showcase examples.

    One active-fire-channel read per event. Columns: early_d1_px, early_d2_px,
    early_px (day 1-2 union), footprint_px (union over all fire days),
    fallback_center, early_crop_clipped (the CNN's 128px crop hits a raster
    edge), footprint_touches_edge (final footprint reaches the raster border,
    so the true extent may be cut off), detected_days / gap_days (fire days
    with zero detections between the first and last detection), bbox_h/bbox_w.
    """
    rows = []
    for row in tqdm(events.itertuples(), total=len(events)):
        event = event_from_row(row)
        fire = read_fire_days(event)
        if fire.shape[0] < CUTOFF_DAY:
            continue
        h, w = fire.shape[1:]
        footprint = fire.any(axis=0)
        early = fire[:CUTOFF_DAY].any(axis=0)

        # early_fire_center wants a (T, 23, H, W) stack; only the fire channel
        # matters to it, so hand it a stack with just that channel populated.
        stub = np.zeros((CUTOFF_DAY, len(CHANNEL_NAMES), h, w), dtype=np.uint8)
        stub[:, ACTIVE_FIRE_CHANNEL_IDX] = fire[:CUTOFF_DAY]
        cy, cx, fallback = early_fire_center(stub)
        _, clipped = extract_crop(np.zeros((1, h, w), np.float32), cy, cx, crop_size)

        daily = fire.reshape(fire.shape[0], -1).sum(axis=1)
        on = np.nonzero(daily)[0]
        gap_days = int(((daily[on[0]:on[-1] + 1]) == 0).sum()) if len(on) else 0

        if footprint.any():
            ys, xs = np.nonzero(footprint)
            touches = ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
            bbox_h, bbox_w = int(np.ptp(ys) + 1), int(np.ptp(xs) + 1)
        else:
            touches, bbox_h, bbox_w = False, 0, 0

        rows.append(dict(
            event_id=row.event_id, early_d1_px=int(daily[0]), early_d2_px=int(daily[1]),
            early_px=int(early.sum()), footprint_px=int(footprint.sum()),
            fallback_center=bool(fallback), early_crop_clipped=bool(clipped),
            footprint_touches_edge=bool(touches), detected_days=int(len(on)),
            gap_days=gap_days, bbox_h=bbox_h, bbox_w=bbox_w,
        ))
    return pd.DataFrame(rows)


def rank_pairs(cands: pd.DataFrame, min_early_px: int = 6, max_early_ratio: float = 1.25,
               max_bbox_px: int = 160, max_gap_days: int = 0) -> pd.DataFrame:
    """Rank (escalated, contained) pairs that look alike after two days.

    `cands` is `scan_events` output merged with escalates/max_area_ha/n_days.
    Eligibility (both fires): fire on day 1 AND day 2, at least `min_early_px`
    early pixels, not a fallback-center event, not edge-clipped, no zero-
    detection gap days (`max_gap_days`). The escalated fire's footprint bbox is capped at
    `max_bbox_px` so its day-1/2 pixels stay visible in a footprint-sized
    window. Score = how different the outcomes are (log footprint ratio)
    minus how unalike the fires were on day 1 and on day 2 separately (log
    per-day pixel-count ratios), so a pair that only matches on the 2-day
    union but diverges day by day ranks lower. No model predictions are used.
    """
    ok = (
        (cands["early_d1_px"] > 0) & (cands["early_d2_px"] > 0)
        & (cands["early_px"] >= min_early_px)
        & ~cands["fallback_center"] & ~cands["early_crop_clipped"]
        & ~cands["footprint_touches_edge"] & (cands["gap_days"] <= max_gap_days)
    )
    pool = cands[ok]
    esc = pool[(pool["escalates"] == 1) & (pool[["bbox_h", "bbox_w"]].max(axis=1) <= max_bbox_px)]
    con = pool[pool["escalates"] == 0]

    pairs = esc.merge(con, how="cross", suffixes=("_esc", "_con"))
    early_ratio = np.exp(np.abs(np.log(pairs["early_px_esc"] / pairs["early_px_con"])))
    pairs = pairs[early_ratio <= max_early_ratio].copy()
    pairs["early_ratio"] = early_ratio[early_ratio <= max_early_ratio]
    pairs["growth_ratio"] = pairs["footprint_px_esc"] / pairs["footprint_px_con"]
    day_mismatch = sum(
        np.abs(np.log(pairs[f"early_d{d}_px_esc"] / pairs[f"early_d{d}_px_con"]))
        for d in (1, 2)
    )
    pairs["score"] = (np.log(pairs["growth_ratio"])
                      - 4 * np.log(pairs["early_ratio"]) - 2 * day_mismatch)
    return pairs.sort_values("score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Static figure
# --------------------------------------------------------------------------

def footprint_window(arrival: np.ndarray, side: int | None = None,
                     margin_frac: float = 0.12, min_margin: int = 4) -> tuple[int, int, int]:
    """Square window (row0, col0, side) around the final footprint's bbox.

    `side` forces a size (e.g. to put two fires at the same scale); the
    window is still centered on this fire's footprint. Row/col may run off
    the raster -- `crop_window` NaN-pads.
    """
    ys, xs = np.nonzero(arrival)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    if side is None:
        extent = max(np.ptp(ys), np.ptp(xs)) + 1
        side = int(extent + 2 * max(min_margin, round(margin_frac * extent)))
    return int(round(cy - side / 2)), int(round(cx - side / 2)), side


def crop_window(plane: np.ndarray, row0: int, col0: int, side: int, fill=np.nan) -> np.ndarray:
    """Crop a 2-D plane to a square window, padding outside the raster."""
    h, w = plane.shape
    out = np.full((side, side), fill, dtype=np.result_type(plane.dtype, type(fill)))
    r0, c0 = max(row0, 0), max(col0, 0)
    r1, c1 = min(row0 + side, h), min(col0 + side, w)
    if r1 > r0 and c1 > c0:
        out[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = plane[r0:r1, c0:c1]
    return out


def mask_outline(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Pixel-edge segments tracing a boolean mask's boundary (imshow coords)."""
    padded = np.pad(mask, 1)
    segs = []
    vert = padded[1:-1, 1:] != padded[1:-1, :-1]   # edges between columns
    for y, x in zip(*np.nonzero(vert)):
        segs.append([(x - 0.5, y - 0.5), (x - 0.5, y + 0.5)])
    horiz = padded[1:, 1:-1] != padded[:-1, 1:-1]  # edges between rows
    for y, x in zip(*np.nonzero(horiz)):
        segs.append([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5)])
    return segs


def load_fire_panels(event: HDF5Event) -> dict:
    """Everything one figure row needs: arrival map, early masks, hillshade.

    `detected_span_days` is first-to-last day with any detection -- the
    honest "how long it burned" number. The DB's duration_days equals the
    stored n_days, which includes the 4 pre-ignition buffer days.
    """
    fire = read_fire_days(event)
    on = np.nonzero(fire.reshape(fire.shape[0], -1).any(axis=1))[0]
    return dict(
        arrival=arrival_day_map(fire),
        day_masks=[fire[d] for d in range(CUTOFF_DAY)],
        shade=hillshade(read_elevation(event)),
        n_fire_days=fire.shape[0],
        detected_span_days=int(on[-1] - on[0] + 1),
        footprint_px=int(fire.any(axis=0).sum()),
    )


def arrival_cmap(max_day: int):
    """Discrete day colormap: ACCENT for the model's days, ramp after."""
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

    ramp = LinearSegmentedColormap.from_list("arrival", ARRIVAL_RAMP)
    later = max_day - CUTOFF_DAY
    colors = [ACCENT] * CUTOFF_DAY + [ramp(i / max(later - 1, 1)) for i in range(later)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, max_day + 1.5), cmap.N)
    return cmap, norm


def full_box_window(shape: tuple[int, int], side: int) -> tuple[int, int, int]:
    """Square window of `side` px centered on the whole raster box."""
    h, w = shape
    return (h - side) // 2, (w - side) // 2, side


def _nice_km(side_px: int, frac: float = 0.25) -> int:
    """Largest 1/2/5 x 10^k km bar no longer than `frac` of the panel."""
    target = side_px * PIXEL_SIDE_M / 1000 * frac
    return int(max(k for k in (1, 2, 5, 10, 20, 50, 100) if k <= max(target, 1)))


def _scale_bar(ax, side: int, km: int, color: str) -> None:
    bar_px = km * 1000 / PIXEL_SIDE_M
    x0, yb = 0.06 * side, 0.92 * side
    ax.plot([x0 - 0.5, x0 - 0.5 + bar_px], [yb, yb], color=color, lw=2.5, solid_capstyle="butt")
    ax.text(x0 - 0.5 + bar_px / 2, yb - 0.035 * side, f"{km} km", ha="center", va="bottom",
            fontsize=10, color=color)


def plot_driver_figure(rows: list[dict], driver_title: str, driver_unit: str,
                       driver_cmap_colors: list[str], title: str,
                       zoom_margin: int = 4, min_zoom_side: int = 20,
                       map_side: int | None = None):
    """2 x (Days 1-2 zoom | driver map | divider | final footprint) figure.

    Two map scales, each shared across rows: the Days 1-2 panel is a zoom
    on each fire's early pixels (same zoom for both rows, own scale bar);
    the driver and footprint panels share one window per row (the whole
    raster box, or `map_side`) at a common scale, with a thin rectangle
    marking the zoom window.

    `rows`: dicts with label, sublines (list of str), p_escalate, the
    `load_fire_panels` output, plus driver_plane / driver_scalar from
    `driver_plane` (display units). Driver color range is shared: 2nd-98th
    percentile of both rows' whole-box pixels pooled. The driver annotation
    is always the whole-box scalar the EBM used, even when the map is
    cropped.

    `map_side` (px) switches the driver and footprint panels from the whole
    box to a shared window of that size centered on each fire's early
    pixels; detections falling outside it are counted in the panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Rectangle

    plt.rcParams.update(SITE_RCPARAMS)
    shade_cmap = LinearSegmentedColormap.from_list("shade", HILLSHADE_RAMP)
    driver_cmap = LinearSegmentedColormap.from_list("driver", driver_cmap_colors)
    driver_cmap.set_bad(alpha=0)

    # Zoom windows: shared side, each centered on its own day-1/2 union.
    early_unions = [(r["arrival"] > 0) & (r["arrival"] <= CUTOFF_DAY) for r in rows]
    zoom_side = max(min_zoom_side, max(
        int(max(np.ptp(np.nonzero(u)[0]), np.ptp(np.nonzero(u)[1]))) + 1 + 2 * zoom_margin
        for u in early_unions))
    zooms = [footprint_window(u.astype(np.int32), side=zoom_side) for u in early_unions]

    # Full-box side: the larger raster's longer edge.
    box_side = max(max(r["arrival"].shape) for r in rows)

    pooled = np.concatenate([r["driver_plane"][np.isfinite(r["driver_plane"])] for r in rows])
    vmin, vmax = np.percentile(pooled, [2, 98])
    max_day = max(int(r["arrival"].max()) for r in rows)
    arr_cmap, arr_norm = arrival_cmap(max_day)
    day2_tint = "#ffc9b8"  # lighter accent for pixels new on day 2

    W, H = 8.0, 6.0
    fig = plt.figure(figsize=(W, H))
    panel, gap, divider = 1.85, 0.08, 0.42
    x_left = 1.8
    xs = [x_left, x_left + panel + gap, x_left + 2 * (panel + gap) + divider]
    ys = [3.1, 1.13]

    if map_side is not None:
        map_windows = [footprint_window(u.astype(np.int32), side=map_side)
                       for u in early_unions]
    else:
        map_windows = [full_box_window(r["arrival"].shape, box_side) for r in rows]

    def add_ax(x, y):
        ax = fig.add_axes([x / W, y / H, panel / W, panel / H])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    axes = []
    for r, early, (mr, mc, ms), (zr, zc, zs), y in zip(
            rows, early_unions, map_windows, zooms, ys):
        row_axes = []

        # Days 1-2 zoom.
        ax = add_ax(xs[0], y)
        ax.imshow(crop_window(r["shade"], zr, zc, zs), cmap=shade_cmap, vmin=0, vmax=1,
                  interpolation="bilinear")
        d1 = crop_window(r["day_masks"][0].astype(np.uint8), zr, zc, zs, fill=0).astype(bool)
        d2 = crop_window(r["day_masks"][1].astype(np.uint8), zr, zc, zs, fill=0).astype(bool)
        layer = np.zeros(d1.shape, np.uint8)
        layer[d2 & ~d1] = 1
        layer[d1] = 2
        ax.imshow(np.ma.masked_equal(layer, 0), cmap=ListedColormap([day2_tint, ACCENT]),
                  vmin=1, vmax=2, interpolation="nearest")
        ax.add_collection(LineCollection(mask_outline(d2 & ~d1), colors=ACCENT, linewidths=0.8))
        ax.set_xlim(-0.5, zs - 0.5); ax.set_ylim(zs - 0.5, -0.5)
        row_axes.append(ax)

        # Driver map + early fire.
        ax = add_ax(xs[1], y)
        ax.imshow(crop_window(r["driver_plane"], mr, mc, ms), cmap=driver_cmap,
                  vmin=vmin, vmax=vmax, interpolation="nearest")
        early_map = crop_window(early.astype(np.uint8), mr, mc, ms, fill=0)
        ax.imshow(np.ma.masked_equal(early_map, 0), cmap=ListedColormap([ACCENT]),
                  interpolation="nearest")
        row_axes.append(ax)

        # Final footprint, same window as the driver map.
        ax = add_ax(xs[2], y)
        ax.imshow(crop_window(r["shade"], mr, mc, ms), cmap=shade_cmap, vmin=0, vmax=1,
                  interpolation="bilinear")
        arrival = crop_window(r["arrival"], mr, mc, ms, fill=0)
        ax.imshow(np.ma.masked_equal(arrival, 0), cmap=arr_cmap, norm=arr_norm,
                  interpolation="nearest")
        n_outside = int((r["arrival"] > 0).sum() - (arrival > 0).sum())
        if n_outside:
            ax.text(0.96 * ms, 0.04 * ms, f"+{n_outside} detections\noutside view",
                    ha="right", va="top", fontsize=10, color=SITE_TEXT_MUTED, linespacing=1.2)
        row_axes.append(ax)

        for ax in row_axes[1:]:
            if ms > zs:
                ax.add_patch(Rectangle((zc - mc - 0.5, zr - mr - 0.5), zs, zs, fill=False,
                                       edgecolor=SITE_TEXT, lw=0.8))
            ax.set_xlim(-0.5, ms - 0.5); ax.set_ylim(ms - 0.5, -0.5)

        row_axes[1].text(0.04 * ms, 0.04 * ms,
                         f"mean over\nmapped area: {r['driver_scalar']:.0f}",
                         ha="left", va="top", fontsize=10, color=SITE_TEXT, linespacing=1.2,
                         bbox=dict(boxstyle="round,pad=0.25", fc=SITE_BG, ec="none", alpha=0.85))
        axes.append(row_axes)

        cy = (y + panel / 2) / H
        fig.text((x_left - 0.12) / W, cy + 0.06, r["label"], ha="right", va="bottom",
                 fontsize=14, fontweight="bold", color=SITE_TEXT)
        fig.text((x_left - 0.12) / W, cy + 0.045, "\n".join(r["sublines"]), ha="right", va="top",
                 fontsize=10, color=SITE_TEXT_MUTED, linespacing=1.45)

    # Headers + divider between driver and footprint columns.
    head_y = (ys[0] + panel + 0.12) / H
    for x, text in zip(xs, ["Days 1–2 (zoomed)", driver_title, "Final footprint"]):
        fig.text((x + panel / 2) / W, head_y, text, ha="center", va="bottom",
                 fontsize=12, color=SITE_TEXT)
    div_x = (xs[1] + panel + gap / 2 + divider / 2) / W
    top = (ys[0] + panel + 0.5) / H
    fig.add_artist(plt.Line2D([div_x, div_x], [ys[1] / H, top], color=SITE_TICK, lw=1,
                              ls=(0, (3, 3))))
    fig.text(div_x - 0.012, top, "model sees  ←", ha="right", va="center", fontsize=11,
             color=ACCENT, fontweight="bold")
    fig.text(div_x + 0.012, top, "→  what happened", ha="left", va="center", fontsize=11,
             color=SITE_TEXT, fontweight="bold")
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=15, fontweight="bold")

    # One scale bar per scale, bottom row.
    _scale_bar(axes[-1][0], zooms[-1][2], _nice_km(zooms[-1][2]), SITE_TEXT)
    map_px = map_windows[-1][2]
    _scale_bar(axes[-1][2], map_px, _nice_km(map_px), SITE_TEXT)

    # Bottom keys: day 1/2 swatches | driver colorbar | arrival colorbar.
    key_y = 0.745 / H
    for i, (color, text) in enumerate([(ACCENT, "day 1"), (day2_tint, "new on day 2")]):
        x = xs[0] + i * 0.75
        fig.add_artist(Rectangle((x / W, 0.68 / H), 0.13 / W, 0.13 / H, color=color,
                                 transform=fig.transFigure))
        fig.text((x + 0.2) / W, key_y, text, fontsize=10, va="center", color=SITE_TEXT_MUTED)

    def hbar(x, mappable, label, **kw):
        cax = fig.add_axes([x / W, 0.68 / H, panel / W, 0.13 / H])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", **kw)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=10, length=0, pad=2)
        cb.set_label(label, fontsize=10, color=SITE_TEXT_MUTED, labelpad=3)
        return cb

    hbar(xs[1], plt.cm.ScalarMappable(cmap=driver_cmap, norm=plt.Normalize(vmin, vmax)),
         driver_unit)
    ticks = [1.5] + [d for d in (5, 10, 15, 20, 25, 30, 40, 50, 60) if d <= max_day]
    cb = hbar(xs[2], plt.cm.ScalarMappable(cmap=arr_cmap, norm=arr_norm), "day first detected",
              ticks=ticks)
    cb.set_ticklabels(["1–2"] + [str(d) for d in ticks[1:]])
    return fig


# --------------------------------------------------------------------------
# Driver selection: which EBM term separates a look-alike pair?
# --------------------------------------------------------------------------

# Notebook 03 Step 9's binary EBM feature set (base + topo).
EBM_BASE_COLS = [
    "fire_extent_ha", "wind_speed_mean", "wind_speed_max",
    "wind_direction_sin_mean", "wind_direction_cos_mean",
    "max_temp_max", "min_temp_min", "humidity_min", "pdsi_mean", "erc_mean",
]
EBM_TOPO_COLS = ["slope_mean", "slope_max", "aspect_sin_mean", "aspect_cos_mean", "elevation_mean"]
EBM_COLS = EBM_BASE_COLS + EBM_TOPO_COLS

# Terms with a plain-language map: feature -> (channel, per-pixel reduction
# over the early days, column title, unit label, value transform for display).
# The reduction mirrors features.early_window_stats, so the box-wide
# nanmean/nanmax/nanmin of the mapped plane IS the scalar the EBM saw (before
# the display transform). Aspect, elevation and wind-direction components
# are deliberately absent (not slope/ERC/weather, or no plain-language map).
MAPPABLE_TERMS = {
    "slope_mean": ("slope", "mean", "Slope", "°", lambda v: v),
    "slope_max": ("slope", "max", "Slope (steepest)", "°", lambda v: v),
    "erc_mean": ("energy_release_component", "mean", "Fuel dryness (ERC)", "index", lambda v: v),
    "wind_speed_mean": ("wind_speed", "mean", "Wind speed", "m/s", lambda v: v),
    "wind_speed_max": ("wind_speed", "max", "Wind speed (peak)", "m/s", lambda v: v),
    "max_temp_max": ("max_temp", "max", "Daytime high", "°C", lambda v: v - 273.15),
    "min_temp_min": ("min_temp", "min", "Overnight low", "°C", lambda v: v - 273.15),
    "humidity_min": ("specific_humidity", "min", "Air moisture (driest)", "g/kg", lambda v: v * 1000),
    "pdsi_mean": ("pdsi", "mean", "Drought index (PDSI)", "index", lambda v: v),
}
_SCALAR_REDUCE = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}


def fit_oof_ebms(train_df: pd.DataFrame, cols: list[str] = EBM_COLS):
    """Step 9's binary EBM, refit per LOYO fold; per-event OOF terms.

    Returns (models_by_test_year, oof) where `oof` is indexed like
    `train_df` with p_escalate plus one column per term holding that
    event's out-of-fold logit contribution (`eval_terms`, the same numbers
    `explain_local` reports).
    """
    from interpret.glassbox import ExplainableBoostingClassifier

    models, frames = {}, []
    for year in sorted(train_df["year"].unique()):
        tr, te = train_df[train_df["year"] != year], train_df[train_df["year"] == year]
        m = ExplainableBoostingClassifier(interactions=0, random_state=0)
        m.fit(tr[cols], tr["escalates"])
        models[year] = m
        contrib = pd.DataFrame(m.eval_terms(te[cols]), index=te.index, columns=m.term_names_)
        contrib.insert(0, "p_escalate", m.predict_proba(te[cols])[:, 1])
        frames.append(contrib)
    return models, pd.concat(frames).loc[train_df.index]


def shape_segment_ok(model, term: str, v_a: float, v_b: float,
                     max_reversal_frac: float = 0.10, max_step_frac: float = 0.50) -> dict:
    """Is a fitted EBM's shape function well-behaved between two values?

    Walks the term's bin scores from v_a to v_b. Monotonic = total movement
    against the overall direction is <= `max_reversal_frac` of the net
    change. No cliff = no single bin-to-bin step exceeds `max_step_frac` of
    the net change.
    """
    i = model.term_names_.index(term)
    edges = np.asarray(model.bins_[i][0])
    scores = np.asarray(model.term_scores_[i])[1:-1]  # drop missing/unknown bins
    lo, hi = sorted((v_a, v_b))
    b_lo, b_hi = np.searchsorted(edges, lo, side="left"), np.searchsorted(edges, hi, side="left")
    seg = scores[b_lo:b_hi + 1]
    steps = np.diff(seg)
    net = seg[-1] - seg[0] if len(seg) > 1 else 0.0
    if len(steps) == 0 or net == 0:
        return dict(net=float(net), reversal_frac=0.0, max_step_frac=1.0, ok=False)
    reversal = np.abs(steps[np.sign(steps) != np.sign(net)]).sum() / abs(net)
    step_frac = np.abs(steps).max() / abs(net)
    return dict(net=float(net), reversal_frac=float(reversal), max_step_frac=float(step_frac),
                ok=bool(reversal <= max_reversal_frac and step_frac <= max_step_frac))


def tail_support(train_values: pd.Series, v: float) -> int:
    """Training fires as extreme as v or more, toward v's nearer tail."""
    return int(min((train_values <= v).sum(), (train_values >= v).sum()))


def driver_plane(event: HDF5Event, term: str) -> tuple[np.ndarray, float]:
    """Full-box map for a mappable term plus the exact scalar the EBM used.

    Reads only the early window, same slice as features.py. Returns the
    plane and scalar in display units (MAPPABLE_TERMS transform).
    """
    from flashpoint.data_access import read_event_window

    channel, reduce, _, _, to_display = MAPPABLE_TERMS[term]
    early = read_event_window(event, BUFFER_DAYS, BUFFER_DAYS + CUTOFF_DAY)
    stack = early[:, CHANNEL_NAMES.index(channel)].astype(np.float32)
    plane = _SCALAR_REDUCE[reduce](stack, axis=0)
    scalar = float(_SCALAR_REDUCE[reduce](stack))
    return to_display(plane), to_display(scalar)


def driver_funnel(pairs: pd.DataFrame, oof: pd.DataFrame, models: dict, train_df: pd.DataFrame,
                  global_rank: pd.Series, events: pd.DataFrame, compact_px: int | None = 64,
                  min_margin: float = 0.1, max_global_rank: int = 5, min_support: int = 20,
                  min_visible_frac: float = 0.25) -> tuple[pd.DataFrame, dict]:
    """Apply the pre-registered driver rules to `rank_pairs` output.

    Rules, in order (counts of surviving pairs returned per step):
    both footprint boxes <= `compact_px` (None skips); both fires called
    correctly out of fold with >= `min_margin` from 0.5; the largest-gap
    non-size term is in MAPPABLE_TERMS and favors the escalated fire; that
    term ranks <= `max_global_rank` in global importance; its shape function
    is monotonic with no single-bin cliff between the two values in both
    fires' fold models; each value has >= `min_support` training fires as
    extreme or more; and the two box means differ by >= `min_visible_frac`
    of the shared (2nd-98th pct pooled) color range.
    """
    counts = {"look-alike pairs (stage-1 rules)": len(pairs)}
    p = pairs.copy()
    if compact_px is not None:
        p = p[(p[["bbox_h_esc", "bbox_w_esc"]].max(axis=1) <= compact_px)
              & (p[["bbox_h_con", "bbox_w_con"]].max(axis=1) <= compact_px)]
        counts[f"both footprint boxes <= {compact_px} px"] = len(p)
    p["p_esc"] = oof["p_escalate"].reindex(p["event_id_esc"]).to_numpy()
    p["p_con"] = oof["p_escalate"].reindex(p["event_id_con"]).to_numpy()
    p = p[(p["p_esc"] >= 0.5 + min_margin) & (p["p_con"] <= 0.5 - min_margin)]
    counts[f"both OOF-correct by >= {min_margin}"] = len(p)

    terms = [t for t in EBM_COLS if t != "fire_extent_ha"]
    gaps = (oof[terms].reindex(p["event_id_esc"]).to_numpy()
            - oof[terms].reindex(p["event_id_con"]).to_numpy())
    top = np.abs(gaps).argmax(axis=1) if len(p) else np.array([], int)
    p["driver"] = [terms[i] for i in top]
    p["gap"] = gaps[np.arange(len(p)), top]
    p = p[p["driver"].isin(MAPPABLE_TERMS.keys())]
    counts["top non-size term is mappable"] = len(p)
    p = p[p["gap"] > 0]
    counts["...and favors the escalated fire"] = len(p)
    p["global_rank"] = global_rank.reindex(p["driver"]).to_numpy()
    p = p[p["global_rank"] <= max_global_rank]
    counts[f"driver in global top {max_global_rank}"] = len(p)

    feats = train_df.set_index("event_id")
    checks = []
    for r in p.itertuples():
        t = r.driver
        ids = (r.event_id_esc, r.event_id_con)
        vals = [feats.loc[e, t] for e in ids]
        shapes = [shape_segment_ok(models[feats.loc[e, "year"]], t, *vals) for e in ids]
        support = [tail_support(train_df.loc[train_df["year"] != feats.loc[e, "year"], t], v)
                   for e, v in zip(ids, vals)]
        checks.append(dict(
            val_esc=vals[0], val_con=vals[1],
            contrib_esc=oof.loc[ids[0], t], contrib_con=oof.loc[ids[1], t],
            shape_ok=all(s["ok"] for s in shapes),
            reversal_frac=max(s["reversal_frac"] for s in shapes),
            max_step_frac=max(s["max_step_frac"] for s in shapes),
            min_support=min(support),
        ))
    p = pd.concat([p.reset_index(drop=True), pd.DataFrame(checks)], axis=1) if checks else p
    if len(p):
        p = p[p["shape_ok"]]
    counts["shape monotonic, no cliff"] = len(p)
    if len(p):
        p = p[p["min_support"] >= min_support]
    counts[f">= {min_support} training fires as extreme"] = len(p)

    ev = events.set_index("event_id", drop=False)
    vis = []
    for r in p.itertuples():
        planes = [driver_plane(event_from_row(ev.loc[e]), r.driver)[0]
                  for e in (r.event_id_esc, r.event_id_con)]
        lo, hi = np.percentile(np.concatenate([x[np.isfinite(x)] for x in planes]), [2, 98])
        vis.append(abs(np.nanmean(planes[0]) - np.nanmean(planes[1])) / (hi - lo))
    if len(p):
        p = p.assign(visible_frac=vis)
        p = p[p["visible_frac"] >= min_visible_frac]
    counts[f"box means differ >= {min_visible_frac:.0%} of color range"] = len(p)
    return p.reset_index(drop=True), counts
