#!/usr/bin/env python3
"""Crop a sweep PLY to the generated building layout bounds."""

import argparse
import json
import os
import struct


def read_ply_xyz(path):
    data = open(path, "rb").read()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    count = [int(line.split()[-1]) for line in header.splitlines() if line.startswith("element vertex ")][0]
    points = [struct.unpack_from("<fff", data, header_end + i * 12) for i in range(count)]
    return points


def write_ply_xyz(path, points):
    with open(path, "wb") as f:
        f.write(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            ).encode("ascii")
        )
        for point in points:
            f.write(struct.pack("<fff", *point))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=float, default=0.6)
    parser.add_argument("--z-min", type=float, default=-0.35)
    parser.add_argument("--z-extra", type=float, default=0.8)
    args = parser.parse_args()

    layout = json.load(open(args.layout, "r", encoding="utf-8"))
    floor = layout["floors"][0]
    regions = [("lobby", floor["lobby_bounds"]), ("corridor", floor["corridor_bounds"])]
    regions.extend((room["id"], room["bounds"]) for room in floor["rooms"])
    z_max = float(layout["wall_height"]) + args.z_extra

    kept = []
    region_counts = {name: 0 for name, _ in regions}
    for x, y, z in read_ply_xyz(args.input):
        if not (args.z_min <= z <= z_max):
            continue
        for name, bounds in regions:
            if (
                bounds["x_min"] - args.margin <= x <= bounds["x_max"] + args.margin
                and bounds["y_min"] - args.margin <= y <= bounds["y_max"] + args.margin
            ):
                kept.append((x, y, z))
                region_counts[name] += 1
                break

    if not kept:
        raise RuntimeError("crop removed all points")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_ply_xyz(args.output, kept)
    meta_path = os.path.splitext(args.output)[0] + ".txt"
    xs = [p[0] for p in kept]
    ys = [p[1] for p in kept]
    zs = [p[2] for p in kept]
    with open(meta_path, "w", encoding="ascii") as f:
        f.write(f"source={args.input}\n")
        f.write(f"points={len(kept)}\n")
        f.write("bbox_min=%.6f %.6f %.6f\n" % (min(xs), min(ys), min(zs)))
        f.write("bbox_max=%.6f %.6f %.6f\n" % (max(xs), max(ys), max(zs)))
        for name, count in region_counts.items():
            f.write(f"region_{name}_points={count}\n")
    print(f"wrote {len(kept)} points to {args.output}")


if __name__ == "__main__":
    main()
