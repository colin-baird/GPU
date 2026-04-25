#!/usr/bin/env python3
"""Generate a roofline plot (SVG) for the matmul benchmark.

Runs `matmul_bench --json` against the DRAMSim3 backend, derives the
arithmetic-intensity / achieved-performance point, and overlays it on the
architectural compute and memory bandwidth roofs. No third-party deps.

Roofs (in fabric-clock units, 150 MHz):
  * Compute: VDOT8 = 4 INT8 MACs/lane * 32 lanes = 128 MACs/instruction;
    operand-collector is the binding constraint at 1 VDOT8 / 2 cycles -->
    64 MAC/cycle peak.
  * DRAM (optimistic, data-bus floor): DDR3-800, BL8 = 32-byte burst, tCCD_S
    = 4 tCK at 400 MHz DRAM = 10 ns/burst = 3.2 GB/s. Converted to fabric
    cycles: 21.33 bytes/cycle.
  * DRAM (sustained empirical, with CAS/ACT overhead): ~27.5 tCK per
    128-byte line under DE-10 Nano timings --> 12.4 bytes/cycle.

Arithmetic intensity is computed as (total MACs) / (DRAM bytes moved),
counting reads and write-throughs.
"""
import json
import math
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO_ROOT, "build/tests/matmul/matmul_bench")
DRAMSIM3_INI = os.path.join(REPO_ROOT, "sim/configs/dram/DDR3_4Gb_x16_800.ini")
OUT_SVG = os.path.join(REPO_ROOT, "tools/roofline_matmul.svg")

# Architectural roofs
PEAK_COMPUTE_MAC_PER_CYCLE = 64.0
PEAK_DRAM_BYTES_PER_CYCLE_OPT = 21.33   # data-bus floor
PEAK_DRAM_BYTES_PER_CYCLE_SUST = 12.4   # empirical sustained
CACHE_LINE_BYTES = 128


def run_matmul_json():
    out = subprocess.check_output(
        [BENCH, "--memory-backend=dramsim3",
         f"--dramsim3-config-path={DRAMSIM3_INI}", "--json"],
        text=True,
    )
    return json.loads(out)


def compute_point(stats):
    cycles = stats["total_cycles"]
    macs_per_cycle = stats["macs_per_cycle"]
    total_macs = macs_per_cycle * cycles
    bytes_moved = (stats["external_memory_reads"] +
                   stats["external_memory_writes"]) * CACHE_LINE_BYTES
    ai = total_macs / bytes_moved
    return {
        "cycles": cycles,
        "macs": total_macs,
        "bytes": bytes_moved,
        "macs_per_cycle": macs_per_cycle,
        "ai": ai,
    }


def render_svg(point):
    # Log-scale axes.
    ai_min, ai_max = 0.1, 32.0          # MAC / byte
    perf_min, perf_max = 0.5, 128.0      # MAC / cycle

    W, H = 760, 520
    L, R, T, B = 80, 40, 40, 70  # margins
    pw, ph = W - L - R, H - T - B

    def x(ai):
        return L + (math.log10(ai) - math.log10(ai_min)) / (
            math.log10(ai_max) - math.log10(ai_min)) * pw

    def y(p):
        return T + (1.0 - (math.log10(p) - math.log10(perf_min)) / (
            math.log10(perf_max) - math.log10(perf_min))) * ph

    def roof_path(bw):
        # perf = min(peak_compute, bw * ai)
        ridge_ai = PEAK_COMPUTE_MAC_PER_CYCLE / bw
        pts = [(ai_min, bw * ai_min)]
        if ai_min < ridge_ai < ai_max:
            pts.append((ridge_ai, PEAK_COMPUTE_MAC_PER_CYCLE))
        pts.append((ai_max, PEAK_COMPUTE_MAC_PER_CYCLE if ai_max >= ridge_ai
                    else bw * ai_max))
        return " ".join(f"{x(a):.1f},{y(p):.1f}" for a, p in pts)

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
               f'height="{H}" viewBox="0 0 {W} {H}" '
               f'font-family="Helvetica, Arial, sans-serif" font-size="12">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')

    # Title
    svg.append(f'<text x="{W/2}" y="22" text-anchor="middle" '
               f'font-size="16" font-weight="600">'
               f'Roofline — matmul (128x128x128 INT8, DDR3-800)</text>')

    # Axes box
    svg.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" '
               f'fill="none" stroke="#333"/>')

    # Gridlines + labels (log decades)
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
        if major or ai in (0.2, 0.5, 2, 5, 20):
            label = f"{ai:g}"
            svg.append(f'<text x="{xc:.1f}" y="{T+ph+15}" '
                       f'text-anchor="middle">{label}</text>')

    for p, major in decades(perf_min, perf_max):
        yc = y(p)
        svg.append(f'<line x1="{L}" y1="{yc:.1f}" x2="{L+pw}" y2="{yc:.1f}" '
                   f'stroke="{"#bbb" if major else "#eee"}" '
                   f'stroke-dasharray="{"" if major else "2,2"}"/>')
        if major or p in (2, 5, 20, 50):
            label = f"{p:g}"
            svg.append(f'<text x="{L-6}" y="{yc+4:.1f}" '
                       f'text-anchor="end">{label}</text>')

    # Axis titles
    svg.append(f'<text x="{L+pw/2}" y="{H-25}" text-anchor="middle" '
               f'font-size="13">Arithmetic intensity (MAC / DRAM byte)</text>')
    svg.append(f'<text transform="translate(20,{T+ph/2}) rotate(-90)" '
               f'text-anchor="middle" font-size="13">'
               f'Performance (MAC / fabric cycle)</text>')

    # Memory roofs
    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.2" '
               f'points="{roof_path(PEAK_DRAM_BYTES_PER_CYCLE_OPT)}"/>')
    svg.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" '
               f'stroke-dasharray="6,4" '
               f'points="{roof_path(PEAK_DRAM_BYTES_PER_CYCLE_SUST)}"/>')

    # Compute roof annotation (horizontal segment label)
    yc = y(PEAK_COMPUTE_MAC_PER_CYCLE)
    svg.append(f'<text x="{L+pw-6}" y="{yc-6:.1f}" text-anchor="end" '
               f'fill="#1f77b4" font-weight="600">'
               f'Compute roof: 64 MAC/cycle (VDOT8, op-collector limited)</text>')

    # Slope labels along memory roofs (placed near low-AI side)
    ai_label = 0.18
    svg.append(
        f'<text x="{x(ai_label)+6:.1f}" '
        f'y="{y(PEAK_DRAM_BYTES_PER_CYCLE_OPT*ai_label)-6:.1f}" '
        f'fill="#1f77b4" font-weight="600">'
        f'DRAM peak (data-bus floor): 21.3 B/cycle</text>')
    svg.append(
        f'<text x="{x(ai_label)+6:.1f}" '
        f'y="{y(PEAK_DRAM_BYTES_PER_CYCLE_SUST*ai_label)-6:.1f}" '
        f'fill="#1f77b4">'
        f'DRAM sustained: 12.4 B/cycle</text>')

    # Achieved point
    px, py = x(point["ai"]), y(point["macs_per_cycle"])
    svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" '
               f'fill="#d62728" stroke="white" stroke-width="1.5"/>')
    label = (f'matmul: AI={point["ai"]:.2f} MAC/B, '
             f'{point["macs_per_cycle"]:.2f} MAC/cycle')
    svg.append(f'<text x="{px+10:.1f}" y="{py-8:.1f}" '
               f'fill="#d62728" font-weight="600">{label}</text>')

    # Vertical drop line to AI axis to make ridge comparison obvious
    svg.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" '
               f'y2="{T+ph}" stroke="#d62728" stroke-dasharray="3,3" '
               f'stroke-width="1"/>')

    # Footer with utilization breakdown
    util_compute = 100 * point["macs_per_cycle"] / PEAK_COMPUTE_MAC_PER_CYCLE
    mem_roof_opt = min(PEAK_COMPUTE_MAC_PER_CYCLE,
                       point["ai"] * PEAK_DRAM_BYTES_PER_CYCLE_OPT)
    mem_roof_sust = min(PEAK_COMPUTE_MAC_PER_CYCLE,
                        point["ai"] * PEAK_DRAM_BYTES_PER_CYCLE_SUST)
    util_mem_opt = 100 * point["macs_per_cycle"] / mem_roof_opt
    util_mem_sust = 100 * point["macs_per_cycle"] / mem_roof_sust
    svg.append(
        f'<text x="{L}" y="{H-8}" font-size="11" fill="#444">'
        f'{point["cycles"]:,} cycles  |  '
        f'{int(point["macs"]):,} MACs  |  '
        f'{int(point["bytes"]):,} DRAM bytes  |  '
        f'compute util {util_compute:.1f}%  |  '
        f'mem-roof util {util_mem_sust:.1f}% (sustained) / '
        f'{util_mem_opt:.1f}% (peak)'
        f'</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def main():
    if not os.path.exists(BENCH):
        sys.exit(f"matmul_bench not found at {BENCH}; build first")
    stats = run_matmul_json()
    point = compute_point(stats)
    svg = render_svg(point)
    with open(OUT_SVG, "w") as f:
        f.write(svg)
    print(f"Wrote {OUT_SVG}")
    print(f"  cycles            = {point['cycles']:,}")
    print(f"  total MACs        = {int(point['macs']):,}")
    print(f"  DRAM bytes moved  = {int(point['bytes']):,}")
    print(f"  arithmetic intensity = {point['ai']:.3f} MAC/byte")
    print(f"  achieved          = {point['macs_per_cycle']:.3f} MAC/cycle")
    print(f"  compute roof      = {PEAK_COMPUTE_MAC_PER_CYCLE} MAC/cycle "
          f"({100*point['macs_per_cycle']/PEAK_COMPUTE_MAC_PER_CYCLE:.1f}% util)")
    mem_sust = min(PEAK_COMPUTE_MAC_PER_CYCLE,
                   point["ai"] * PEAK_DRAM_BYTES_PER_CYCLE_SUST)
    mem_opt = min(PEAK_COMPUTE_MAC_PER_CYCLE,
                  point["ai"] * PEAK_DRAM_BYTES_PER_CYCLE_OPT)
    print(f"  memory roof (sustained) = {mem_sust:.2f} MAC/cycle "
          f"({100*point['macs_per_cycle']/mem_sust:.1f}% util)")
    print(f"  memory roof (peak)      = {mem_opt:.2f} MAC/cycle "
          f"({100*point['macs_per_cycle']/mem_opt:.1f}% util)")


if __name__ == "__main__":
    main()
