#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["matplotlib", "numpy", "pillow"]
# ///
"""Annotate the Blender render with labels for the manuscript figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _style import COL_DOUBLE, set_style  # noqa: E402

set_style()


def main():
    img_path = Path(__file__).parent / "fig_setup3d.png"
    img = Image.open(img_path).convert("RGBA")

    # Composite on white background
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white, img)
    arr = np.array(composited.convert("RGB"))

    fig, ax = plt.subplots(figsize=(COL_DOUBLE, COL_DOUBLE * 0.38))
    ax.imshow(arr, aspect="equal")
    ax.set_axis_off()

    h, w = arr.shape[:2]

    # Labels — positioned relative to image dimensions
    label_style = dict(fontsize=8, ha="center", va="center")

    # B_0 arrow label
    ax.annotate(r"$B_0$", xy=(0.91 * w, 0.38 * h),
                color="darkred", fontsize=9, ha="center", va="center")

    # xi* crossing marker
    ax.annotate(r"$\xi^*$", xy=(0.44 * w, 0.80 * h),
                color="0.35", **label_style)

    # Solenoid label
    ax.annotate("solenoid", xy=(0.58 * w, 0.10 * h),
                color="0.45", fontsize=7, fontstyle="italic",
                ha="center", va="center")

    # Capillary label with arrow pointing to the tube
    ax.annotate("capillary", xy=(0.33 * w, 0.55 * h),
                xytext=(0.20 * w, 0.88 * h),
                color="#56B4E9", fontsize=7, fontstyle="italic",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color="#56B4E9",
                                lw=0.5))

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    out = Path(__file__).parent / "fig_setup3d_final.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
