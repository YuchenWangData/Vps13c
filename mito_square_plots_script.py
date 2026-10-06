#!/usr/bin/env python3
import gzip
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------
# configuration: data root, output directory, thresholds
# ----------------------------------------------------------------------
DATA_DIR = os.environ.get("MITO_EQTL_DATA_DIR", "data_root")
OUT_DIR = os.environ.get("MITO_EQTL_OUT_DIR", "output_square")

cutoff = 1e-3          # significance threshold for cis pairs
trans_cutoff = 4.1e-6  # significance threshold for trans pairs
CIS_CUT, TRANS_CUT = cutoff, trans_cutoff

os.makedirs(OUT_DIR, exist_ok=True)
for fp in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Noto Serif CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# step 1: chromosome offsets (mm10) for the concatenated 1..X axis
# ----------------------------------------------------------------------
CHROM_MB = {"1": 194.3, "2": 182.1, "3": 159.6, "4": 156.5, "5": 151.8,
            "6": 149.5, "7": 144.9, "8": 130.1, "9": 124.4, "10": 130.5,
            "11": 121.9, "12": 117.5, "13": 93.8, "14": 124.9, "15": 109.4,
            "16": 98.1, "17": 94.9, "18": 90.6, "19": 61.3, "X": 169.0}
order = [str(i) for i in range(1, 20)] + ["X"]
offsets = {}
_acc = 0.0
for _c in order:
    offsets[_c] = _acc
    _acc += CHROM_MB[_c]
ACC_MB = _acc

# chromosome midpoints for the tick labels
TICKS = [offsets[c] + CHROM_MB[c] / 2 for c in order]


def add_chromosome_bands(ax):
    """alternating grey bands over odd chromosomes, on both axes."""
    for i, c in enumerate(order):
        if i % 2 == 1:
            lo, hi = offsets[c], offsets[c] + CHROM_MB[c]
            ax.axvspan(lo, hi, color="0.55", alpha=0.15, lw=0)
            ax.axhspan(lo, hi, color="0.55", alpha=0.15, lw=0)


# SNP id -> (chr, pos mm10) from the genotype .bim file
BIM = {}
with open(f"{DATA_DIR}/genotype/all_strains.bim") as f:
    for line in f:
        p = line.split()
        BIM[p[1]] = (p[0], int(p[3]))

VPS13C_MB = 67.8   # red triangle marker on chr9


def x_of(c, pos):
    """absolute mm10 coordinate of a position on chromosome c."""
    c = "X" if c == "20" else c
    return offsets[c] + pos / 1e6 if c in offsets else None


# ----------------------------------------------------------------------
# step 2: gene annotation
# ----------------------------------------------------------------------
def panel_annotation(path, header=True):
    """(gene_id, symbol, chr, start, end) per row of an expression matrix."""
    ann = []
    with open(path) as f:
        if header:
            f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            ann.append((p[0], p[1], p[2], int(float(p[3])), int(float(p[4]))))
    return ann


# one shared annotation table for all cohorts
ANN = panel_annotation(f"{DATA_DIR}/expression/reference_annotation_expr.tsv",
                       header=True)
ANN_BY_ID = {a[0]: a for a in ANN}


def panel_gene_ids(path):
    """gene ids in expression-file row order (hit tables store these row
    indices); the first line is skipped only when it is a header."""
    ids, seen = [], set()
    with open(path) as f:
        first = f.readline()
        if not first.startswith("gene_id"):
            p = first.rstrip("\n").split("\t")
            if p[0]:
                ids.append(p[0])
                seen.add(p[0])
        for line in f:
            gid = line.rstrip("\n").split("\t")[0]
            if gid and gid != "gene_id" and gid not in seen:
                seen.add(gid)
                ids.append(gid)
    return ids


# ----------------------------------------------------------------------
# step 3: keep significant cis/trans pairs
# ----------------------------------------------------------------------
def square_points(hits_gz, ids):
    """significant pairs -> (x, y, -log10p) with x = SNP, y = gene."""
    pts = []
    op = gzip.open if hits_gz.endswith(".gz") else open
    with op(hits_gz, "rt") as fh:
        for line in fh:
            gi, sid, pv = line.rstrip("\n").split("\t")
            # a hit row stores either an Ensembl gene id or a 0-based row
            # index into the cohort's expression matrix
            if gi.startswith("ENSMUSG"):
                rec = ANN_BY_ID.get(gi)
            elif gi.isdigit() and int(gi) < len(ids):
                rec = ANN_BY_ID.get(ids[int(gi)])
            else:
                rec = None
            if rec is None:
                continue
            gid, sym, gchr, gstart, gend = rec
            sc, sp = BIM.get(sid, (None, None))
            if sc is None:
                continue
            p = float(pv)
            cis = sc == gchr and gstart - 1e6 <= sp <= gend + 1e6
            if not p < (CIS_CUT if cis else TRANS_CUT):
                continue
            x = x_of(sc, sp)
            y = x_of(gchr, (gstart + gend) / 2)
            if x is None or y is None:
                continue
            pts.append((x, y, min(-math.log10(p), 21.0) if p > 0 else 21.0))
    arr = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)
    return arr[np.argsort(arr[:, 2])]   # small points first, big on top


# ----------------------------------------------------------------------
# step 4: the square plot
# ----------------------------------------------------------------------
def draw_square(path, pts, title):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    gx, gy, lp = pts[:, 0], pts[:, 1], pts[:, 2]
    size = 0.5686 * (np.clip(lp, 3, 21) ** 1.6542)
    add_chromosome_bands(ax)
    ax.scatter(gx, gy, s=size, c=lp, cmap="viridis", vmin=3, vmax=21,
               alpha=0.45, linewidths=0.25, edgecolors="black")
    ax.set_xticks(TICKS)
    ax.set_xticklabels(order, fontsize=7)
    ax.set_yticks(TICKS)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("SNP position", fontsize=8)
    ax.set_ylabel("Gene position", fontsize=8)
    ax.set_title(f"{title} (n={len(pts):,} pairs, cis p<1e-3 / trans p<4.1e-6)",
                 fontsize=10, pad=14)
    # red triangle: chr9 Vps13c locus, drawn just above the frame
    ax.plot([offsets["9"] + VPS13C_MB], [ACC_MB * 1.03], marker="v",
            color="red", markersize=7, clip_on=False, linestyle="none")
    ax.set_ylim(0, ACC_MB * 1.005)
    ax.margins(x=0.01)
    # size legend (-log10 p), outside the axes on the right
    axl = fig.add_axes([0.86, 0.12, 0.10, 0.50])
    axl.set_axis_off()
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate(range(21, 2, -2)):
        axl.scatter([0.5], [i], s=0.5686 * (v ** 1.6542),
                    c=[cmap((v - 3) / 18)], edgecolors="black", linewidths=0.25)
        axl.text(1.5, i, str(v), va="center", fontsize=8)
    axl.text(0.9, 9.7, "-log10(p-value)", fontsize=8.5, ha="center")
    axl.set_xlim(0, 2.4)
    axl.set_ylim(-0.1, 9.6)
    fig.subplots_adjust(right=0.84)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("done", os.path.basename(path), f"({len(pts):,} pairs)", flush=True)


# ----------------------------------------------------------------------
# cohorts: (cohort key, plot title)
#   hit table      = DATA_DIR/hits/<key>.hits.tsv.gz
#   expression file= DATA_DIR/expression/<key>_expr.tsv
# ----------------------------------------------------------------------
COHORTS = [
    ("hfpef_merged93",
     "Merged batch1+2 male (93 strains, batch = seq batch) - FaST-LMM+batch+LOCO"),
    ("hfpef_batch1_male32",
     "Batch 1 male (32 strains, seq batch 1) - FaST-LMM-LOCO"),
    ("hfpef_batch1_female28",
     "Batch 1 female (28 strains, seq batch 1) - FaST-LMM-LOCO"),
    ("hfpef_batch2_bxd64",
     "Batch 2 male (64 strains, seq batch 2, no BXD9/20) - FaST-LMM-LOCO"),
    ("hfhs_male96",
     "HF/HS heart male (96 strains) - FaST-LMM-LOCO"),
    ("hfhs_female88",
     "HF/HS heart female (88 strains) - FaST-LMM-LOCO"),
]

if __name__ == "__main__":
    for key, title in COHORTS:
        hits = f"{DATA_DIR}/hits/{key}.hits.tsv.gz"
        expr = f"{DATA_DIR}/expression/{key}_expr.tsv"
        if not os.path.exists(hits):
            print("hit table missing, skipped:", hits)
            continue
        ids = panel_gene_ids(expr)           # row order of the hit indices
        pts = square_points(hits, ids)
        draw_square(f"{OUT_DIR}/FigureB_{key}.png", pts, title)
    print("SQUARE_PLOTS_DONE")
