#!/usr/bin/env python3
"""Sweep matmul shapes against the DRAMSim3 backend; emit a table and a
multi-point roofline SVG.

Builds one kernel ELF per (N, K) shape via assembler --defsym, runs the
bench at the requested (M, N, K), and collects per-shape stats.
"""
import json
import math
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO_ROOT, "build/tests/matmul/matmul_bench")
KERNEL_S = os.path.join(REPO_ROOT, "tests/matmul/matmul_kernel.S")
LINK_LD = os.path.join(REPO_ROOT, "tests/matmul/link.ld")
DRAMSIM3_INI = os.path.join(REPO_ROOT, "sim/configs/dram/DDR3_4Gb_x16_800.ini")
OUT_SVG = os.path.join(REPO_ROOT, "tools/roofline_matmul_shapes.svg")
KERNEL_DIR = "/tmp/matmul_kernels"

PEAK_COMPUTE = 64.0
PEAK_DRAM_OPT = 21.33
PEAK_DRAM_SUST = 12.4
CACHE_LINE = 128

SHAPES = [
    # (M,   N,   K)
    (128, 128, 128),  # original
    (128,  64, 128),  # narrower output
    (128, 256, 128),  # wider output
    (128, 128,  64),  # shorter reduction
    (128, 128, 256),  # longer reduction
    ( 64,  64,  64),  # all small
    (256, 128, 128),  # tall
    ( 64, 256, 128),  # narrow rows, wide output
]


def build_kernel(N, K):
    cc = shutil.which("riscv64-unknown-elf-gcc") or \
         shutil.which("riscv32-unknown-elf-gcc")
    if not cc:
        sys.exit("RISC-V cross-compiler not found")
    os.makedirs(KERNEL_DIR, exist_ok=True)
    elf = os.path.join(KERNEL_DIR, f"k_{N}_{K}.elf")
    if not os.path.exists(elf):
        subprocess.check_call([
            cc, "-march=rv32im_zicsr", "-mabi=ilp32",
            "-nostdlib", "-nostartfiles",
            f"-Wa,--defsym=MATMUL_N={N},--defsym=MATMUL_K={K}",
            "-T", LINK_LD,
            "-o", elf, KERNEL_S,
        ])
    return elf


def run_shape(M, N, K):
    elf = build_kernel(N, K)
    out = subprocess.check_output(
        [BENCH, "--memory-backend=dramsim3",
         f"--dramsim3-config-path={DRAMSIM3_INI}",
         f"--m={M}", f"--n={N}", f"--k={K}",
         f"--kernel-elf={elf}",
         "--max-cycles=20000000",
         "--json"],
        text=True,
    )
    d = json.loads(out)
    cycles = d["total_cycles"]
    macs = float(d["macs_per_cycle"]) * cycles
    bytes_ = (d["external_memory_reads"] + d["external_memory_writes"]) * CACHE_LINE
    return {
        "M": M, "N": N, "K": K,
        "cycles": cycles,
        "macs_per_cycle": macs / cycles,
        "macs": macs,
        "bytes": bytes_,
        "ai": macs / bytes_ if bytes_ else float("inf"),
        "bytes_per_cycle": bytes_ / cycles,
        "external_reads": d["external_memory_reads"],
        "external_writes": d["external_memory_writes"],
        "cache_hits": d["cache_hits"],
        "cache_misses": d["cache_misses"],
        "avg_read_latency": (d["external_read_latency_total"] /
                             d["external_read_latency_count"])
            if d["external_read_latency_count"] else float("nan"),
    }


def render_svg(points):
    ai_min, ai_max = 0.1, 32.0
    perf_min, perf_max = 0.5, 128.0
    W, H = 820, 560
    L, R, T, B = 80, 220, 40, 70
    pw, ph = W - L - R, H - T - B

    def x(ai):
        return L + (math.log10(ai) - math.log10(ai_min)) / (
            math.log10(ai_max) - math.log10(ai_min)) * pw

    def y(p):
        return T + (1.0 - (math.log10(p) - math.log10(perf_min)) / (
            math.log10(perf_max) - math.log10(perf_min))) * ph

    def roof(bw):
        ridge = PEAK_COMPUTE / bw
        pts = [(ai_min, bw * ai_min)]
        if ai_min < ridge < ai_max:
            pts.append((ridge, PEAK_COMPUTE))
        pts.append((ai_max, PEAK_COMPUTE if ai_max >= ridge else bw * ai_max))
        return " ".join(f"{x(a):.1f},{y(p):.1f}" for a, p in pts)

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
               f'height="{H}" viewBox="0 0 {W} {H}" '
               f'font-family="Helvetica, Arial, sans-serif" font-size="12">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{(L+L+pw)/2}" y="22" text-anchor="middle" '
               f'font-size="16" font-weight="600">'
               f'Roofline — matmul shape sweep (DDR3-800)</text>')
    svg.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" '
               f'fill="none" stroke="#333"/>')

    def decades(lo, hi):
        out = []
        d = math.floor(math.log10(lo))
        while 10 ** d <= hi * 1.001:
            for m in (1, 2, 5):
                v = m * 10 ** d
                if lo <= v <= hi:
                    out.append((v, m == 1))
            d += 1
        return out

    for ai, major in decades(ai_min, ai_max):
        xc = x(ai)
        svg.append(f'<line x1="{xc:.1f}" y1="{T}" x2="{xc:.1f}" y2="{T+ph}" '
                   f'stroke="{"#bbb" if major else "#eee"}" '
                   f'stroke-dasharray="{"" if major else "2,2"}"/>')
        if major:
            svg.append(f'<text x="{xc:.1f}" y="{T+ph+15}" '
                       f'text-anchor="middle">{ai:g}</text>')
    for p, major in decades(perf_min, perf_max):
        yc = y(p)
        svg.append(f'<line x1="{L}" y1="{yc:.1f}" x2="{L+pw}" y2="{yc:.1f}" '
                   f'stroke="{"#bbb" if major else "#eee"}" '
                   f'stroke-dasharray="{"" if major else "2,2"}"/>')
        if major:
            svg.append(f'<text x="{L-6}" y="{yc+4:.1f}" '
                       f'text-anchor="end">{p:g}</text>')

    svg.append(f'<text x="{L+pw/2}" y="{H-25}" text-anchor="middle" '
               f'font-size="13">Arithmetic intensity (MAC / DRAM byte)</text>')
    svg.append(f'<text transform="translate(20,{T+ph/2}) rotate(-90)" '
               f'text-anchor="middle" font-size="13">'
               f'Performance (MAC / fabric cycle)</text>')

    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.2" '
               f'points="{roof(PEAK_DRAM_OPT)}"/>')
    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" '
               f'stroke-dasharray="6,4" points="{roof(PEAK_DRAM_SUST)}"/>')
    yc = y(PEAK_COMPUTE)
    svg.append(f'<text x="{L+pw-6}" y="{yc-6:.1f}" text-anchor="end" '
               f'fill="#1f77b4" font-weight="600">'
               f'Compute roof: 64 MAC/cycle</text>')

    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    legend_x = L + pw + 18
    svg.append(f'<text x="{legend_x}" y="{T+8}" font-weight="600">'
               f'M x N x K  (AI, MAC/cycle)</text>')
    for i, p in enumerate(points):
        color = palette[i % len(palette)]
        px, py = x(p["ai"]), y(p["macs_per_cycle"])
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" '
                   f'fill="{color}" stroke="white" stroke-width="1.2"/>')
        ly = T + 28 + i * 18
        svg.append(f'<circle cx="{legend_x+6}" cy="{ly-4}" r="5" '
                   f'fill="{color}"/>')
        svg.append(f'<text x="{legend_x+18}" y="{ly}" font-size="11">'
                   f'{p["M"]}×{p["N"]}×{p["K"]}  '
                   f'({p["ai"]:.2f}, {p["macs_per_cycle"]:.1f})</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    if not os.path.exists(BENCH):
        sys.exit(f"matmul_bench not found at {BENCH}; build first")
    points = []
    print(f"{'shape':14s} {'cycles':>9s} {'MAC/cyc':>8s} {'AI':>6s} "
          f"{'B/cyc':>6s} {'GB/s':>6s} {'cmp%':>5s} {'mem%sust':>9s} "
          f"{'avgLat':>7s}  hits/miss")
    for (M, N, K) in SHAPES:
        p = run_shape(M, N, K)
        gbs = p["bytes_per_cycle"] * 150 / 1000
        mem_roof = min(PEAK_COMPUTE, p["ai"] * PEAK_DRAM_SUST)
        cmp_pct = 100 * p["macs_per_cycle"] / PEAK_COMPUTE
        mem_pct = 100 * p["macs_per_cycle"] / mem_roof
        print(f"{M:>3d}x{N:>3d}x{K:>3d}  "
              f"{p['cycles']:>9d} {p['macs_per_cycle']:>8.2f} "
              f"{p['ai']:>6.2f} {p['bytes_per_cycle']:>6.2f} {gbs:>6.2f} "
              f"{cmp_pct:>5.1f} {mem_pct:>9.1f} "
              f"{p['avg_read_latency']:>7.2f}  "
              f"{p['cache_hits']}/{p['cache_misses']}")
        points.append(p)
    svg = render_svg(points)
    with open(OUT_SVG, "w") as f:
        f.write(svg)
    print(f"\nWrote {OUT_SVG}")


if __name__ == "__main__":
    main()
