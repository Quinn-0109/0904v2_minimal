#!/usr/bin/env python3
"""聚合多轮三层全流程测试结果:分段耗时 + 房间探索率 + 危险源识别。

用法:
    python3 scripts/aggregate_full_flow_stats.py <run_dir_1> [<run_dir_2> ...]
    python3 scripts/aggregate_full_flow_stats.py /root/.ros/results_fuel_full_20260820/fuel_rooffix_full*
    python3 scripts/aggregate_full_flow_stats.py --output results/full_flow_stats.md <run_dirs...>

每轮目录读取(与 run_fuel_three_floor_full.sh 产物布局一致):
    分段耗时  -> 复用 src/simenv_exploration/scripts/statistics_timing_report.analyze()
    房间     -> baseline_summary.json / second_floor|third_floor/baseline_summary.json
    危险源   -> detected_danger.json vs 同目录 danger_truth.json(阈值 1.0m)
               复用 src/building_obstacles/scripts/evaulate_danger 的匹配与评分
    mission  -> logs/stair_descent.json phase=FIRST_FLOOR_RETURNED + first_floor_returned.json

输出 Markdown 报告到 stdout;--output 指定落盘文件。纯标准库 + 上述两个复用模块。
"""

import argparse
import json
import os
import statistics
import sys

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for rel in ("src/simenv_exploration/scripts", "src/building_obstacles/scripts"):
    p = os.path.join(WS, rel)
    if p not in sys.path:
        sys.path.insert(0, p)

import statistics_timing_report as strt
import sim_time_calibration as stc

# 危险源评分依赖原仓库 src/building_obstacles/scripts/evaulate_danger.py
# (探测部分已从本交付移除)。可选:无该模块时跳过危险源统计与评分。
try:
    import evaulate_danger as evd
except ImportError:
    evd = None

DANGER_THRESHOLD_M = 1.0  # 比赛固定阈值

# 与 run_fuel_three_floor_full.sh 的 F3_OK_TERMS 保持一致:
# F3 探索被认可(可触发下梯)的终止原因集合
F3_OK_TERMS = (
    "STAIR_CORRIDOR_EXIT_HANDOFF",
    "STAIR_CORRIDOR_EXIT_HANDOFF_TRUTH_GATE",
    "STAIR_LOBBY_HANDOFF_LOCALIZATION_FALLBACK",
    "STAIR_WAIT_ZONE_REACHED",
)


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def fmt(value, digits=1):
    if value is None:
        return "-"
    try:
        return "{:.{d}f}".format(float(value), d=digits)
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value):
    return "-" if value is None else str(value)


def sim_cell(r, key):
    """仿真时间单元格;段RTF偏离整轮RTF>25% 时尾标 *。"""
    if key == "residual_sec":
        v = r.get("sim_residual_sec")
        return "-" if v is None else fmt(v)
    v = r.get("sim", {}).get(key)
    if v is None:
        return "-"
    ftf = (r.get("rtfs", {}).get(key) or {}).get("rtf")
    run_rtf = r.get("run_rtf")
    s = fmt(v)
    if (ftf and run_rtf and ftf > 0 and
            abs(ftf - run_rtf) / run_rtf > 0.25):
        s += "*"
    return s


def room_stats(floor_summary):
    """从一层 baseline_summary.json 提取房间/覆盖指标 dict。"""
    if not floor_summary:
        return None
    rcs = floor_summary.get("region_coverage_status") or {}
    return {
        "complete": floor_summary.get("complete_room_count"),
        "exited": floor_summary.get("exited_room_count"),
        "recognized": floor_summary.get("recognized_room_count"),
        "target": floor_summary.get("room_target_count"),
        "coverage_ratio": rcs.get("coverage_ratio"),
        "coverage_complete": rcs.get("coverage_complete"),
        "termination": floor_summary.get("termination_reason"),
    }


def hazard_stats(run_dir):
    """危险源:正确/漏报/虚警 + 37 分制得分。exploration_time 用总探索时长。"""
    if evd is None:
        return None
    truth = load_json(os.path.join(run_dir, "danger_truth.json"))
    detected = load_json(os.path.join(run_dir, "detected_danger.json"))
    if not truth:
        return None
    truth_pos = evd.load_positions_from_data(truth, "danger_sources", "position")
    n_truth = len(truth_pos)
    if not detected:
        return {"n_truth": n_truth, "correct": 0, "missed": n_truth,
                "false_alarms": 0, "n_detected": 0, "scores": None}
    det_pos = evd.load_positions_from_data(detected, "detected_danger_sources")
    correct, missed, false_alarms = evd.evaluate_detection(
        truth_pos, det_pos, DANGER_THRESHOLD_M)
    out = {"n_truth": n_truth, "correct": correct, "missed": missed,
           "false_alarms": false_alarms, "n_detected": len(det_pos),
           "scores": None}
    return out


def scores_for(hstat, exploration_time):
    """37 分制客观分:时间 15 + 识别 14 + 虚警 8。exploration_time 缺省时跳过。"""
    if evd is None or hstat is None or exploration_time is None:
        return None
    t, p, f = evd.compute_scores(exploration_time, hstat["correct"],
                                 hstat["n_truth"], hstat["false_alarms"],
                                 hstat["n_detected"])
    return {"time": t, "prob": p, "far": f, "total": t + p + f}


def analyze_run(run_dir, sim_time=False, cal_source="trajectory"):
    """单轮全量指标。sim_time=True 时附加仿真时间换算结果。"""
    result = {"dir": run_dir}

    metrics, notes = strt.analyze(run_dir)
    result["metrics"] = metrics
    result["notes"] = list(notes)
    if sim_time:
        cal = stc.convert_run(run_dir, metrics, cal_source)
        result["sim"] = cal["sim"]
        result["sim_residual_sec"] = cal["sim_residual_sec"]
        result["rtfs"] = cal["rtfs"]
        result["run_rtf"] = cal["run_rtf"]
        result["calibration_source"] = cal["calibration_source"]
        result["notes"] += cal["notes"]

    # mission 成功 = 回到一楼
    descent = load_json(os.path.join(run_dir, "logs", "stair_descent.json"))
    returned = load_json(os.path.join(run_dir, "first_floor_returned.json"))
    result["mission_ok"] = bool(
        returned and returned.get("phase") == "FIRST_FLOOR_RETURNED"
        and descent and descent.get("phase") == "FIRST_FLOOR_RETURNED")

    result["rooms"] = {
        "F1": room_stats(load_json(os.path.join(run_dir, "baseline_summary.json"))),
        "F2": room_stats(load_json(os.path.join(run_dir, "second_floor",
                                                "baseline_summary.json"))),
        "F3": room_stats(load_json(os.path.join(run_dir, "third_floor",
                                                "baseline_summary.json"))),
    }

    # 失败定位:按推进顺序取第一个卡点
    t12 = load_json(os.path.join(run_dir, "logs", "stair_transition.json"))
    t23 = load_json(os.path.join(run_dir, "logs",
                                  "second_to_third_floor_stair_transition.json"))
    result["fail_point"] = None
    if not result["mission_ok"]:
        if (t12 or {}).get("phase") not in (None, "SECOND_FLOOR_HANDOFF_COMPLETE"):
            result["fail_point"] = f"F1→F2: {(t12 or {}).get('phase')}"
        elif (t23 or {}).get("phase") not in (None, "THIRD_FLOOR_HANDOFF_COMPLETE"):
            result["fail_point"] = f"F2→F3: {(t23 or {}).get('phase')}"
        else:
            f3_term = (result["rooms"].get("F3") or {}).get("termination")
            if f3_term and f3_term not in F3_OK_TERMS:
                result["fail_point"] = f"F3探索: {f3_term}"
            elif descent and descent.get("phase") != "FIRST_FLOOR_RETURNED":
                result["fail_point"] = f"返程: {descent.get('phase')}"

    hstat = hazard_stats(run_dir)
    result["hazard"] = hstat
    result["scores"] = scores_for(hstat, metrics.get("total_sec"))
    return result


def aggregate(rows):
    """对数值段做 mean±std(墙钟,含 sim 时另存仿真值);空列表返回 None。"""
    keys = ("f1_explore_sec", "stair_guide_sec", "stair_ascent_total_sec",
            "f2_corridor_guide_sec", "f2_explore_sec", "f2_to_f3_stair_sec",
            "f3_explore_sec", "descent_segment1_sec", "descent_segment2_sec",
            "descent_total_sec", "total_sec", "residual_sec")
    out = {}
    for key in keys:
        vals = [r["metrics"].get(key) for r in rows
                if r["metrics"].get(key) is not None]
        if vals:
            out[key] = {"mean": statistics.fmean(vals),
                        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals)}
        svals = [r["sim"].get(key) for r in rows
                 if r.get("sim", {}).get(key) is not None]
        if svals:
            out["sim_" + key] = {"mean": statistics.fmean(svals),
                                 "std": statistics.stdev(svals)
                                 if len(svals) > 1 else 0.0,
                                 "n": len(svals)}
    sres = [r["sim_residual_sec"] for r in rows
            if r.get("sim_residual_sec") is not None]
    if sres:
        out["sim_residual_sec"] = {"mean": statistics.fmean(sres),
                                   "std": statistics.stdev(sres)
                                   if len(sres) > 1 else 0.0,
                                   "n": len(sres)}
    return out


SEG_ORDER = [
    ("f1_explore_sec", "F1探索"),
    ("stair_guide_sec", "楼梯引导"),
    ("stair_ascent_total_sec", "F1→F2爬梯"),
    ("f2_corridor_guide_sec", "F2走廊引导"),
    ("f2_explore_sec", "F2探索"),
    ("f2_to_f3_stair_sec", "F2→F3爬梯"),
    ("f3_explore_sec", "F3探索"),
    ("descent_segment1_sec", "返程段1(F3→F2)"),
    ("descent_segment2_sec", "返程段2(F2→F1)"),
    ("descent_total_sec", "返程总"),
    ("total_sec", "总时长"),
    ("residual_sec", "残差"),
]


def render_md(runs, agg, sim_time=False):
    lines = []
    a = lines.append
    a("# 三层全流程测试聚合统计")
    a("")
    ok = sum(1 for r in runs if r["mission_ok"])
    a(f"轮次: {len(runs)}  成功: {ok}  成功率: {ok}/{len(runs)}")
    a("")
    if sim_time:
        a("> 耗时=仿真时间: 每段端点在校准流(trajectory_timeseries.jsonl 5Hz,"
          "缺省回退 map_growth_timeseries.jsonl 0.5Hz)上线性插值,"
          "段RTF=Δros/Δwall; 残差在仿真空间重算。`*` 标段RTF偏离整轮RTF>25%。")
        a("")
    a("## 1. 逐轮总览(分段耗时 " + ("仿真s" if sim_time else "s") + ")")
    if sim_time:
        a("| run | mission | RTF | " + " | ".join(n for _, n in SEG_ORDER) + " |")
        a("|---|" + "---|" * (len(SEG_ORDER) + 2))
    else:
        a("| run | mission | " + " | ".join(n for _, n in SEG_ORDER) + " |")
        a("|---|" + "---|" * (len(SEG_ORDER) + 1))
    for r in runs:
        cells = ["✅" if r["mission_ok"] else "❌"]
        if sim_time:
            rtf = r.get("run_rtf")
            cells.append("-" if rtf is None else "%.3f" % rtf)
            for key, _ in SEG_ORDER:
                cells.append(sim_cell(r, key))
        else:
            m = r["metrics"]
            for key, _ in SEG_ORDER:
                cells.append(fmt(m.get(key)))
        a(f"| {os.path.basename(r['dir'])} | " + " | ".join(cells) + " |")
    a("")
    if sim_time:
        a("## 2. 成功轮分段耗时统计(仿真时间 mean±std, n, 墙钟均值参考)")
        a("| 环节 | 仿真mean | 仿真std | n | 仿真min | 仿真max | 墙钟mean |")
        a("|---|---|---|---|---|---|---|")
        for key, name in SEG_ORDER:
            st = agg.get("sim_" + key) if key != "residual_sec" else \
                agg.get("sim_residual_sec")
            if not st:
                a(f"| {name} | - | - | - | - | - | - |")
                continue
            vals = []
            for r in runs:
                v = r.get("sim_residual_sec") if key == "residual_sec" \
                    else r.get("sim", {}).get(key)
                if v is not None:
                    vals.append(v)
            wmean = (agg.get(key) or {}).get("mean")
            a(f"| {name} | {fmt(st['mean'])} | {fmt(st['std'])} | {st['n']} | "
              f"{fmt(min(vals))} | {fmt(max(vals))} | {fmt(wmean)} |")
    else:
        a("## 2. 成功轮分段耗时统计(mean±std, n)")
        a("| 环节 | mean | std | n | min | max |")
        a("|---|---|---|---|---|---|")
        for key, name in SEG_ORDER:
            st = agg.get(key)
            if not st:
                a(f"| {name} | - | - | - | - | - |")
                continue
            vals = []
            for r in runs:
                v = r["metrics"].get(key)
                if v is not None:
                    vals.append(v)
            a(f"| {name} | {fmt(st['mean'])} | {fmt(st['std'])} | {st['n']} | "
              f"{fmt(min(vals))} | {fmt(max(vals))} |")
    a("")
    a("## 3. 房间探索率")
    a("| run | F1 c/e | F2 c/e | F3 c/e | 合计 complete | 覆盖比 F1/F2/F3 |")
    a("|---|---|---|---|---|---|")
    for r in runs:
        rs = r["rooms"]
        cells = []
        total_comp = 0
        ratios = []
        for f in ("F1", "F2", "F3"):
            st = rs.get(f)
            if not st:
                cells.append("-")
                continue
            c, e = st["complete"], st["exited"]
            total_comp += c or 0
            cells.append(f"{fmt_int(c)}/{fmt_int(e)}")
            if st["coverage_ratio"] is not None:
                ratios.append(fmt(st["coverage_ratio"] * 100, 0))
        a(f"| {os.path.basename(r['dir'])} | " + " | ".join(
            cells + [f"{total_comp}/12", "/".join(ratios) if ratios else "-"]) + " |")
    a("")
    # 每层房间目标与均值
    a("### 每层房间完成数(目标 4/层,12 间)与覆盖率均值")
    a("| 楼层 | complete 均值 | 覆盖率均值 | 终止原因(逐轮) |")
    a("|---|---|---|---|")
    for f in ("F1", "F2", "F3"):
        comps = [r["rooms"][f]["complete"] for r in runs
                 if r["rooms"].get(f) and r["rooms"][f]["complete"] is not None]
        ratios = [r["rooms"][f]["coverage_ratio"] for r in runs
                  if r["rooms"].get(f) and r["rooms"][f]["coverage_ratio"] is not None]
        terms = [r["rooms"][f]["termination"] or "-" for r in runs
                 if r["rooms"].get(f)]
        comp_m = statistics.fmean(comps) if comps else None
        ratio_m = statistics.fmean(ratios) if ratios else None
        a(f"| {f} | {fmt(comp_m, 1)} | {fmt((ratio_m or 0) * 100, 1)}% | "
          f"{' / '.join(map(str, terms))} |")
    a("")
    a("## 4. 危险源识别(真值 18,阈值 1.0m,客观分 37 分制)")
    a("| run | 正确 | 漏报 | 虚警 | 检测数 | 时间分/15 | 识别分/14 | 虚警分/8 | 总分/37 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        h = r["hazard"]
        if not h:
            a(f"| {os.path.basename(r['dir'])} | - | - | - | - | - | - | - | - |")
            continue
        sc = r["scores"]
        sc_cells = (["-", "-", "-", "-"] if sc is None else
                    [fmt(sc["time"]), fmt(sc["prob"]), fmt(sc["far"]),
                     fmt(sc["total"])])
        a(f"| {os.path.basename(r['dir'])} | {h['correct']} | {h['missed']} | "
          f"{h['false_alarms']} | {h['n_detected']} | " + " | ".join(sc_cells) + " |")
    a("")
    # 危险源聚合
    okruns = [r for r in runs if r["mission_ok"]]
    if okruns:
        corrs = [r["hazard"]["correct"] for r in okruns if r["hazard"]]
        fas = [r["hazard"]["false_alarms"] for r in okruns if r["hazard"]]
        tots = [r["scores"]["total"] for r in okruns
                if r["scores"] is not None]
        if corrs:
            a(f"成功轮危险源正确识别: {statistics.fmean(corrs):.1f}±"
              f"{statistics.stdev(corrs) if len(corrs) > 1 else 0:.1f}"
              f" /18 (范围 {min(corrs)}-{max(corrs)})")
        if fas:
            a(f"成功轮平均虚警: {statistics.fmean(fas):.1f} "
              f"(范围 {min(fas)}-{max(fas)})")
        if tots:
            a(f"成功轮客观分: {statistics.fmean(tots):.1f}±"
              f"{statistics.stdev(tots) if len(tots) > 1 else 0:.1f}/37")
    if sim_time:
        a("")
        a("> 注: 危险源时间分按比赛规则(600s 阈值,每 60s 扣 1 分)用墙钟总时长"
          "计算;本报告其余耗时均为仿真时间。")
    a("")
    a("## 5. 失败轮原因")
    failed = [r for r in runs if not r["mission_ok"]]
    if failed:
        a("| run | 卡点 | 说明 |")
        a("|---|---|---|")
        for r in failed:
            a(f"| {os.path.basename(r['dir'])} | {r['fail_point'] or '-'} | "
              f"{'; '.join(r['notes']) if r['notes'] else '-'} |")
    else:
        a("无失败轮。")
    a("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="一轮结果目录(fuel_rooffix_fullN)")
    parser.add_argument("--output", default=None, help="报告落盘路径")
    parser.add_argument("--sim-time", action="store_true",
                        help="分段耗时换算为仿真时间(端点插值,默认 trajectory 校准)")
    parser.add_argument("--calibration-source",
                        choices=("trajectory", "map_growth"),
                        default="trajectory",
                        help="仿真时间校准源(默认 trajectory,数据不足自动回退 map_growth)")
    args = parser.parse_args(argv)

    runs = [analyze_run(d, sim_time=args.sim_time,
                        cal_source=args.calibration_source)
            for d in args.run_dirs]
    agg = aggregate(runs)
    md = render_md(runs, agg, sim_time=args.sim_time)
    print(md)
    if args.output:
        out = os.path.join(WS, args.output) if not os.path.isabs(args.output) \
            else args.output
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as stream:
            stream.write(md + "\n")
        print(f"\n报告已写入: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
