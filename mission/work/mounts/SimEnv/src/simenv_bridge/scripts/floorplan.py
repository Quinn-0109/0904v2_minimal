#!/usr/bin/env python3
"""
floorplan.py — 从 layout_metadata.json 的真值几何渲染场景楼层平面图 (占据栅格风格)。

机器人实测地图受深度相机 FOV + 运动稳定性限制难以完整; 这里直接从场景生成器的
真值 bounds 渲染正确的楼层平面: 房间/走廊/电梯/楼梯各自的矩形边界=墙(黑), 内部=空闲(白),
门=墙上的开口(白). 这就是场景的真实结构.

用法: python3 floorplan.py [--meta ...] [--out ...]
"""
import json, argparse, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_region(ax, bounds, color="black", door=None, door_gap=0.7, lw=2.0):
    """画一个矩形区域的墙轮廓. door=(x,y) 在某条边上 → 那条边留个缺口."""
    x0, x1, y0, y1 = bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"]
    edges = [
        ((x0, y0), (x1, y0)),  # 下边 y=y0
        ((x0, y1), (x1, y1)),  # 上边 y=y1
        ((x0, y0), (x0, y1)),  # 左边 x=x0
        ((x1, y0), (x1, y1)),  # 右边 x=x1
    ]
    for (p, q) in edges:
        if door is not None:
            dx, dy = door
            # 判断门是否在这条边上(中点距离)
            mx, my = (p[0]+q[0])/2, (p[1]+q[1])/2
            on_edge = (abs(p[1]-q[1]) < 1e-6 and abs(dy-p[1]) < 0.3 and min(p[0],q[0])-0.3 <= dx <= max(p[0],q[0])+0.3) or \
                      (abs(p[0]-q[0]) < 1e-6 and abs(dx-p[0]) < 0.3 and min(p[1],q[1])-0.3 <= dy <= max(p[1],q[1])+0.3)
            if on_edge:
                # 在边上以 door 为中心留缺口
                if abs(p[1]-q[1]) < 1e-6:  # 水平边
                    lo = max(p[0], q[0]); hi = min(p[0], q[0])
                    lo, hi = min(p[0],q[0]), max(p[0],q[0])
                    dlo = max(lo, dx-door_gap/2); dhi = min(hi, dx+door_gap/2)
                    ax.plot([lo, dlo], [p[1], p[1]], color=color, lw=lw)
                    ax.plot([dhi, hi], [p[1], p[1]], color=color, lw=lw)
                else:  # 竖直边
                    lo, hi = min(p[1],q[1]), max(p[1],q[1])
                    dlo = max(lo, dy-door_gap/2); dhi = min(hi, dy+door_gap/2)
                    ax.plot([p[0], p[0]], [lo, dlo], color=color, lw=lw)
                    ax.plot([p[0], p[0]], [dhi, hi], color=color, lw=lw)
                continue
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="/workspace/SimEnv/generated_building/layout_metadata.json")
    ap.add_argument("--out", default="/workspace/SimEnv/results/map/floorplan.png")
    a = ap.parse_args()
    m = json.load(open(a.meta))
    f = m["floors"][0]

    fig, ax = plt.subplots(figsize=(8, 14))
    # 收集所有区域 bounds, 计算建筑外墙(并集外包框)
    region_bs = []
    for key in ["corridor_bounds", "lobby_bounds", "elevator_bounds", "stair_bounds"]:
        b = f.get(key)
        if b and "x_min" in b: region_bs.append(b)
    for r in f["rooms"]:
        if "bounds" in r: region_bs.append(r["bounds"])
    fp = m.get("footprint", {})
    if region_bs:
        xs = [v for b in region_bs for v in (b["x_min"], b["x_max"])]
        ys = [v for b in region_bs for v in (b["y_min"], b["y_max"])]
        exterior = {"x_min": min(xs)-0.2, "x_max": max(xs)+0.2,
                    "y_min": min(ys)-0.2, "y_max": max(ys)+0.2}
        draw_region(ax, exterior, color="black", lw=3.0)

    # 走廊/电梯厅/电梯井/楼梯 (内部功能区, 画轮廓)
    for key in ["corridor_bounds", "lobby_bounds", "elevator_bounds", "stair_bounds"]:
        b = f.get(key)
        if b:
            draw_region(ax, b, color="black", lw=1.8)
            # 标注
            cx = (b["x_min"]+b["x_max"])/2; cy = (b["y_min"]+b["y_max"])/2
            ax.text(cx, cy, key.replace("_bounds",""), ha="center", va="center",
                    fontsize=7, color="#444444", rotation=90 if (b["x_max"]-b["x_min"]) < (b["y_max"]-b["y_min"]) else 0)

    # 房间 (画轮廓 + 门洞 + 编号)
    for i, r in enumerate(f["rooms"]):
        b = r["bounds"]
        door = tuple(r.get("door_pose", [None,None])[:2])
        if door[0] is None: door = None
        draw_region(ax, b, color="black", lw=1.8, door=door)
        cx = (b["x_min"]+b["x_max"])/2; cy = (b["y_min"]+b["y_max"])/2
        ax.text(cx, cy, "%s\n%s" % (r.get("id","room"), r.get("room_type","")),
                ha="center", va="center", fontsize=7, color="#0066cc")
        # 房间内部填淡色表示空闲
        ax.add_patch(Rectangle((b["x_min"], b["y_min"]), b["x_max"]-b["x_min"], b["y_max"]-b["y_min"],
                               facecolor="#f0f0f0", edgecolor="none", zorder=0))

    # 标题 + 范围
    xs = [v for b in region_bs for v in (b["x_min"], b["x_max"])]
    ys = [v for b in region_bs for v in (b["y_min"], b["y_max"])]
    ax.set_xlim(min(xs)-2, max(xs)+2)
    ax.set_ylim(min(ys)-2, max(ys)+2)
    ax.set_aspect("equal"); ax.grid(True, color="#dddddd", lw=0.4); ax.set_axisbelow(True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Floor plan (ground truth, 1 floor, %d rooms) — %s" % (len(f["rooms"]), m.get("model_name","")))
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
