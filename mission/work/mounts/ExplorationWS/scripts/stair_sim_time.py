#!/usr/bin/env python3
"""专项实验墙钟 → 仿真时间换算工具。

专项的 transition trace 只有墙钟 t（time.monotonic 累计）；
仿真耗时 = 墙钟 × RTF。RTF 用同目录 map_growth_timeseries.jsonl
的 ros_time/wall_time 线性斜率估计。

用法:
    python3 scripts/stair_sim_time.py <experiment_dir> [--ascent|--descent]
    python3 scripts/stair_sim_time.py /root/.ros/results_stair_optim_20260820/U1a

按 climb 场景 (second_to_third_floor_stair_transition.json) 或 descent
场景 (third_to_second / second_to_first ...) 分别换算爬梯执行段。
"""

import argparse
import glob
import json
import os
import statistics
import sys


def rtf_of(run_dir):
    """整轮 RTF = Δros_time / Δwall_time。"""
    path = os.path.join(run_dir, "map_growth_timeseries.jsonl")
    rows = []
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("ros_time") and d.get("wall_time"):
                rows.append((d["ros_time"], d["wall_time"]))
    except OSError:
        return None
    if len(rows) < 3:
        return None
    dr = rows[-1][0] - rows[0][0]
    dw = rows[-1][1] - rows[0][1]
    return dr / dw if dw > 0 else None


def trace_span(run_dir, json_name, start_phase):
    """trace 内从 start_phase 首次出现到 trace 末的 t 跨度（墙钟）。"""
    path = os.path.join(run_dir, "logs", json_name)
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tr = d.get("trace") or []
    idx = [i for i, p in enumerate(tr) if p.get("phase") == start_phase]
    if not idx:
        return None
    return tr[-1]["t"] - tr[idx[0]]["t"]


def fmt(v):
    return "%.1f" % v if v is not None else "-"


def analyze(run_dir, kind):
    if kind == "ascent":
        span = trace_span(run_dir, "second_to_third_floor_stair_transition.json",
                          "STAIR_ASCENT")
        label = "上行爬梯执行段"
    else:
        # 下行取两段之和（若只有一段存在也统计）
        s1 = trace_span(run_dir, "third_to_second_floor_stair_transition.json",
                        "STAIR_DESCENT_PRE_ALIGN")
        s2 = trace_span(run_dir, "second_to_first_floor_stair_transition.json",
                        "STAIR_DESCENT_PRE_ALIGN")
        if s1 is None and s2 is None:
            span = None
        else:
            span = (s1 or 0.0) + (s2 or 0.0)
        label = "下行爬梯执行段(两段和)"
    rtf = rtf_of(run_dir)
    sim = span * rtf if (span is not None and rtf) else None
    return label, span, rtf, sim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", help="实验目录（含 f2_spawn_stair_run*/ 等子目录）")
    parser.add_argument("--ascent", action="store_true", help="按上行爬梯口径换算")
    parser.add_argument("--descent", action="store_true", help="按下行爬梯口径换算")
    args = parser.parse_args(argv)

    patterns = [os.path.join(args.experiment_dir, "f2_spawn_stair_run*"),
                os.path.join(args.experiment_dir, "f3_spawn_stair_descent_run*"),
                os.path.join(args.experiment_dir, "f2_spawn_stair_descent_run*")]
    kind = "ascent" if args.ascent else ("descent" if args.descent else "ascent")
    if kind == "descent":
        patterns = [os.path.join(args.experiment_dir, "f3_spawn_stair_descent_run*"),
                    os.path.join(args.experiment_dir, "f2_spawn_stair_descent_run*")]

    rows = []
    for pat in patterns:
        for rd in sorted(glob.glob(pat)):
            label, span, rtf, sim = analyze(rd, kind)
            if span is None:
                continue
            rows.append((os.path.basename(rd), span, rtf, sim, label))
            print(f"{os.path.basename(rd)}: {label} 墙钟 {fmt(span)}s "
                  f"x RTF {fmt(rtf)} = 仿真 {fmt(sim)}s")
    if rows:
        walls = [r[1] for r in rows if r[1] is not None]
        sims = [r[3] for r in rows if r[3] is not None]
        print(f"\n{rows[0][4]}: 墙钟中位 {fmt(statistics.median(walls))}s | "
              f"仿真中位 {fmt(statistics.median(sims))}s (n={len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
