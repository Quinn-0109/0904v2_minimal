#!/usr/bin/env bash
# Run the mission on N DIFFERENT official scenes.
#
# Why this is not just "change SEED": for an official scene the seed offset is
# inert.  randomize_three_floor_scene.py skips furniture and red-ball placement
# when preserve_official_source_positions is set, and viewpoints are derived
# from geometry alone, so two offsets on one scene produce byte-identical
# routes (verified: 0 waypoints moved, no field differs).  A different scene
# means a different official generator seed, i.e. a new scene directory.
#
#   bash run_multiseed.sh                       # 10 seeds, RTF 0.55
#   SEEDS="20 111 4242" TARGET_RTF=0.55 bash run_multiseed.sh
#   REGEN=1 bash run_multiseed.sh               # regenerate scenes that exist
set -Eeuo pipefail

# The multiseed parent process owns the optional Gazebo RTF-cap helper.  The
# mission child sources ROS independently, but the parent must source ROS too
# because its /usr/bin/python3 block imports rospy directly.
if [ -f /opt/ros/noetic/setup.bash ]; then
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
else
  echo "ERROR: ROS Noetic setup.bash is missing" >&2
  exit 2
fi

ROOT="${ROOT:-/home/quinn/0904}"
SEEDS="${SEEDS:-20 111 4242 20260 20260820 20260821 31337 51501 70707 90210}"
TARGET_RTF="${TARGET_RTF:-0.55}"     # 0.55 is the cap that first completed; "none" to skip
TAG="${TAG:-multiseed}"
REGEN="${REGEN:-0}"
# Solver settings baked into a generated scene.  "default" is what the generator
# itself defaults to, what auto.sh now defaults to, and what the base world of
# the 15/15-passing randomizer family carries; "coarse" is the 0.002 / 40 / 5.0
# every official scene in this workspace used to be built with.  Only scenes
# this script GENERATES are affected -- a reused scene directory keeps the
# solver block baked into its world file, so pass REGEN=1 to switch an existing
# set of scenes over.
PHYS="${PHYS:-default}"
case "$PHYS" in
  coarse)  PHYS_STEP=0.002; PHYS_RATE=325; PHYS_ITERS=40; PHYS_CVEL=5.0  ;;
  default) PHYS_STEP=0.001; PHYS_RATE=650; PHYS_ITERS=50; PHYS_CVEL=10.0 ;;
  *) echo "ERROR: PHYS must be coarse or default" >&2; exit 2 ;;
esac
RESULTS="$ROOT/results"
SCENES="${SCENES:-$ROOT/generated_multiseed}"

SIM="$ROOT/SimEnv-master"
GEN="$SIM/src/building_obstacles/scripts/generate_competition_scene.py"
POLICY="$ROOT/runtime_assets/SimEnv/src/unitree_guide/logs"

# Support both the historical 0904 layout and the merged minimal bundle.
# The mission implementation remains owned by the bundle selected here; the
# command-line interface and ROOT/SEEDS contract are unchanged.
LEGACY_MISSION_ROOT="$ROOT/threefloor_stress5_repro_20260827/work/mounts/SimEnv/src/simenv_bridge"
MINIMAL_MISSION_ROOT="$ROOT/mission/work/mounts/SimEnv/src/simenv_bridge"
if [ -f "$MINIMAL_MISSION_ROOT/config/three_floor_rl_mission.json" ]; then
  MISSION_ROOT="$MINIMAL_MISSION_ROOT"
elif [ -f "$LEGACY_MISSION_ROOT/config/three_floor_rl_mission.json" ]; then
  MISSION_ROOT="$LEGACY_MISSION_ROOT"
else
  echo "ERROR: no supported simenv_bridge mission tree under $ROOT" >&2
  exit 2
fi
CANON="$MISSION_ROOT/config/three_floor_rl_mission.json"
BRIDGE="$MISSION_ROOT/scripts"
RUNNER="$ROOT/run_native.sh"
if [ ! -f "$RUNNER" ] && [ -f "$ROOT/run.sh" ]; then
  RUNNER="$ROOT/run.sh"
fi
# Prefer the bundle-pinned LibTorch when this is the minimal bundle.  This
# keeps the public ROOT/SEEDS command unchanged and prevents the minimal
# runner from falling back to an incomplete /opt/libtorch installation.
if [ -z "${SIMENV_TORCH_ROOT:-}" ] && [ -d "$ROOT/third_party/libtorch" ]; then
  export SIMENV_TORCH_ROOT="$ROOT/third_party/libtorch"
fi

for f in "$GEN" "$CANON" "$RUNNER" \
         "$POLICY/policy_act_inference_plane.pt" "$POLICY/policy_act_inference_stair.pt"; do
  [ -s "$f" ] || { echo "ERROR: missing $f" >&2; exit 2; }
done

export SIMENV_VALIDATED_CONTROLLER="$ROOT/runtime_assets/SimEnv/runtime_bin/junior_ctrl_validated"
export SIMENV_SKIP_UNITREE_REBUILD=1
export CUDA_VISIBLE_DEVICES=""
export RECORD_CAMERA_VIDEO="${RECORD_CAMERA_VIDEO:-false}"
export MAX_WALL_SEC="${MAX_WALL_SEC:-3000}"
export PYTHONNOUSERSITE=1

echo "=== configuration under test ==="
python3 - "$CANON" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["scene_randomization"]
for k in ("room_door_crossing_speed", "room_door_maximum_yaw_rate", "scan_yaw_rate_rad_s"):
    print("  %-32s %s" % (k, s.get(k)))
PY
echo "  solver: $PHYS (step $PHYS_STEP, iters $PHYS_ITERS, correcting_vel $PHYS_CVEL)"
echo "  seeds: $SEEDS"
echo "  RTF cap: $TARGET_RTF   scenes: $SCENES"
echo

# ---- 1. make sure every seed has a scene -----------------------------------
mkdir -p "$SCENES"
declare -A SCENE_OF
for seed in $SEEDS; do
  # reuse an existing official scene for this seed if one is already unpacked
  existing="$(ls -d "$ROOT"/generated_official_s${seed}_* 2>/dev/null | head -1 || true)"
  dir="$SCENES/scene_s${seed}"
  if [ "$REGEN" = "0" ] && [ -s "$dir/competition_scene.world" ]; then
    echo "  seed $seed: reusing $dir"
  elif [ "$REGEN" = "0" ] && [ -n "$existing" ] && [ -s "$existing/competition_scene.world" ]; then
    # Existing official scenes were generated with different physics (some bake
    # real_time_update_rate 500 = RTF 1.0, the working one bakes 325 = 0.65).
    # Copy it, then normalise the throttle so every seed is compared on equal
    # terms; step size and solver iterations are left exactly as generated.
    echo "  seed $seed: copying $existing"
    rm -rf "$dir"; cp -r "$existing" "$dir"
  else
    echo "  seed $seed: generating"
    rm -rf "$dir"; mkdir -p "$dir"
    PYTHONPATH="$SIM/src/building_generator_core:$SIM/src/building_generator_classic${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$GEN" --output-dir "$dir" --seed "$seed" \
      --floor-count 3 --rooms-per-floor 4 --width 20 --length 36 \
      --danger-count 3:6 --distractor-count 4:8 \
      --referee-results-dir "$dir" --team-info-dir "$dir" \
      --physics-max-step-size "$PHYS_STEP" --physics-real-time-update-rate "$PHYS_RATE" \
      --physics-ode-iters "$PHYS_ITERS" --physics-contact-max-correcting-vel "$PHYS_CVEL" \
      > "$dir/generate.log" 2>&1 || {
        echo "  ERROR: generation failed for seed $seed; see $dir/generate.log" >&2; exit 2; }
  fi
  [ -s "$dir/danger_truth.json" ] || { echo "ERROR: $dir has no danger_truth.json" >&2; exit 2; }

  # Normalise the baked RTF CEILING, not the raw rate: ceiling = step x rate,
  # so a world generated at the finer 0.001 s step needs twice the rate for the
  # same pace.  Rewriting the rate to a fixed 325 regardless of step size would
  # silently halve the pace of a fine-solver world.  The solver block itself is
  # left exactly as generated -- it is the variable under test.
  python3 - "$dir/competition_scene.world" <<'PY'
import io, re, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
mr = re.search(r"<real_time_update_rate>([0-9.]+)</real_time_update_rate>", s)
ms = re.search(r"<max_step_size>([0-9.]+)</max_step_size>", s)
if mr and ms:
    step = float(ms.group(1)); want = int(round(0.65 / step))
    if int(float(mr.group(1))) != want:
        s = s.replace(mr.group(0), "<real_time_update_rate>%d</real_time_update_rate>" % want, 1)
        io.open(p, "w", encoding="utf-8").write(s)
        print("    world throttle %s -> %d (step %.4f s, RTF ceiling 0.65)" % (mr.group(1), want, step))
PY

  # assemble the mission config for this scene
  python3 - "$CANON" "$dir" "$seed" "$POLICY" <<'PY'
import json, os, sys
canon, dirpath, seed, policy = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = json.load(open(canon))
d["runtime"] = {
    "world_file": os.path.join(dirpath, "competition_scene.world"),
    "layout_metadata": os.path.join(dirpath, "layout_metadata.json"),
    "stair_model_sdf": os.path.join(dirpath, "model.sdf"),
    "plane_policy": os.path.join(policy, "policy_act_inference_plane.pt"),
    "stair_policy": os.path.join(policy, "policy_act_inference_stair.pt"),
}
d["scene_randomization"]["official_scene_seed"] = seed
d["scene_randomization"]["preserve_official_source_positions"] = True
out = os.path.join(dirpath, "three_floor_rl_mission.json")
json.dump(d, open(out, "w"), indent=2, sort_keys=True); open(out, "a").write("\n")
n = len(json.load(open(os.path.join(dirpath, "danger_truth.json")))["danger_sources"])
print("    mission config written; %d danger sources in this scene" % n)
PY
  SCENE_OF[$seed]="$dir"
done

# ---- 1b. plan every scene offline before spending a single run -------------
# A scene the planner cannot solve fails in seconds here instead of twenty
# minutes into a mission, and the shape it prints is what separates "this seed
# is slow" from "the robot is slow".
echo
echo "=== offline plan check (no simulation) ==="
PLANOK=()
for seed in $SEEDS; do
  dir="${SCENE_OF[$seed]}"; work="$dir/.plancheck"
  rm -rf "$work"; mkdir -p "$work"
  if ! python3 "$BRIDGE/prepare_three_floor_scene.py" --world "$dir/competition_scene.world"         --model "$dir/model.sdf" --layout "$dir/layout_metadata.json"         --output-dir "$work" > "$work/prepare.json" 2> "$work/prepare.err"; then
    echo "  seed $seed: PREPARE FAILED - $(tail -1 "$work/prepare.err")"; continue
  fi
  if ! python3 "$BRIDGE/randomize_three_floor_scene.py"         --world "$work/competition_scene_with_dangers.world" --model "$work/model.sdf"         --layout "$dir/layout_metadata.json" --mission-config "$dir/three_floor_rl_mission.json"         --output-dir "$work" --seed-offset "$seed" > "$work/randomize.json" 2> "$work/randomize.err"; then
    echo "  seed $seed: PLAN FAILED - $(tail -1 "$work/randomize.err")"; continue
  fi
  if ! python3 "$BRIDGE/check_three_floor_rl_mission.py" --config "$work/mission_config.json"         --validate-config > "$work/validate.log" 2>&1; then
    echo "  seed $seed: VALIDATE FAILED - $(tail -1 "$work/validate.log")"; continue
  fi
  python3 - "$seed" "$work" <<'PY'
import json, re, sys
seed, work = sys.argv[1], sys.argv[2]
mc = json.load(open(work + "/mission_config.json"))
md = json.load(open(work + "/layout_metadata.json"))["metadata"]
wps = [w for fl in mc["floors"] for w in fl["waypoints"]]
bends = sum(1 for w in wps if re.search(r"_g[34]_path_\d+$", w["note"]))
rt = md.get("room_type_counts", {})
print("  seed %-9s %3d wp  %2d bends  route %6.1f m  cov %.3f  %d balls  %s"
      % (seed, len(wps), bends, md["runtime_floor_route_m"],
         md["mean_room_scan_coverage_fraction"], md["red_ball_count"],
         " ".join("%s=%d" % (k[:5], v) for k, v in sorted(rt.items()) if v)))
PY
  PLANOK+=("$seed")
done
if [ "${#PLANOK[@]}" -ne "$(echo $SEEDS | wc -w)" ]; then
  echo
  echo "  only ${#PLANOK[@]} of $(echo $SEEDS | wc -w) scenes are runnable; the rest are skipped."
  SEEDS="${PLANOK[*]}"
fi

# ---- 2. run one mission per seed -------------------------------------------
wait_for_idle() {
  local deadline=$((SECONDS + 180))
  while [ $SECONDS -lt $deadline ]; do
    ps -eo stat=,args= | awk '$1 !~ /^Z/ && $0 ~ /[r]osmaster|[g]zserver|[r]oslaunch/ { found=1 }
                              END { exit !found }' || return 0
    sleep 5
  done
  echo "ERROR: a ROS/Gazebo session is still live after 180 s; stopping" >&2
  return 1
}

mkdir -p "$RESULTS"
STARTED="$(date +%Y%m%d_%H%M%S)"
NAMES=()
echo
for seed in $SEEDS; do
  RUN="${TAG}_s${seed}_${STARTED}"
  NAMES+=("$RUN")
  echo "[$(date +%H:%M:%S)] seed $seed  ->  $RUN"
  wait_for_idle || break
  export MISSION_CONFIG="${SCENE_OF[$seed]}/three_floor_rl_mission.json"
  set +e
  OLD_GZ="$(pgrep -x gzserver 2>/dev/null | tr '\n' ' ')"
  bash "$RUNNER" "$RUN" "$seed" > "$RESULTS/$RUN.console.log" 2>&1 &
  MISSION_PID=$!
  if [ "$TARGET_RTF" != "none" ]; then
    GZPID=""
    for _ in $(seq 1 300); do
      for pid in $(pgrep -x gzserver 2>/dev/null); do
        case " $OLD_GZ " in *" $pid "*) ;; *) GZPID="$pid" ;; esac
      done
      [ -n "$GZPID" ] && break
      kill -0 "$MISSION_PID" 2>/dev/null || break
      sleep 1
    done
    if [ -n "$GZPID" ]; then
      for NAME in ROS_MASTER_URI GAZEBO_MASTER_URI; do
        VALUE="$(tr '\0' '\n' < "/proc/$GZPID/environ" | sed -n "s/^${NAME}=//p" | head -1)"
        [ -n "$VALUE" ] && export "$NAME=$VALUE"
      done
      TARGET_RTF="$TARGET_RTF" /usr/bin/python3 - <<'PYRTF'
import os, time, rospy
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties, SetPhysicsPropertiesRequest
from rosgraph_msgs.msg import Clock
target = float(os.environ["TARGET_RTF"])
rospy.init_node("multiseed_rtf_cap", anonymous=True, disable_signals=True)
try:
    rospy.wait_for_service("/gazebo/get_physics_properties", timeout=180.0)
    rospy.wait_for_service("/gazebo/set_physics_properties", timeout=30.0)
    rospy.wait_for_message("/clock", Clock, timeout=30.0)
except Exception as e:
    raise SystemExit("           RTF_CAP_SKIPPED: %s" % e)
time.sleep(2.0)
cur = rospy.ServiceProxy("/gazebo/get_physics_properties", GetPhysicsProperties)()
req = SetPhysicsPropertiesRequest()
req.time_step = cur.time_step
req.max_update_rate = target / float(cur.time_step)
req.gravity = cur.gravity
req.ode_config = cur.ode_config
r = rospy.ServiceProxy("/gazebo/set_physics_properties", SetPhysicsProperties)(req)
print("           RTF capped at %.2f" % target if r.success else "           RTF_CAP_FAILED")
PYRTF
    else
      if kill -0 "$MISSION_PID" 2>/dev/null; then
        echo "           WARNING: no gzserver appeared within the cap wait window" >&2
      else
        echo "           runner exited before Gazebo; see $RESULTS/$RUN.console.log" >&2
      fi
    fi
  fi
  wait "$MISSION_PID"; rc=$?
  set -e
  python3 - "$RESULTS/$RUN/mission_stage_timing.json" "$rc" <<'PY'
import json, os, sys
try:
    with open(sys.argv[1]) as stream:
        d = json.load(stream)
except (OSError, ValueError) as exc:
    print("           rc=%s  timing unavailable: %s" % (sys.argv[2], exc))
    console = os.path.dirname(sys.argv[1]) + ".console.log"
    try:
        lines = open(console, encoding="utf-8", errors="replace").read().splitlines()
        error_lines = [line for line in lines if "[runtime FAIL]" in line or line.startswith("ERROR:")
                       or line.startswith("OSError:") or line.startswith("ModuleNotFoundError:")]
        for line in error_lines[:20]:
            print("           " + line)
    except OSError:
        pass
    raise SystemExit(0)
w = d.get("wall_duration_sec") or 0
sim = d.get("sim_duration_sec")
print("           %s  sim %s s  RTF %s  %s" % (
    d.get("status"), "%.1f" % sim if sim is not None else "n/a (exploration not timed)",
    "%.2f" % (sim / w) if sim is not None and w else "n/a",
    d.get("failure_reason") or ""))
if sim is None:
    print("           preparation %.1f s; inspect %s" % (
        d.get("algorithm_preparation_duration_sec") or 0,
        os.path.join(os.path.dirname(sys.argv[1]), "roslaunch.log")))
PY
done

# ---- 3. summary ------------------------------------------------------------
echo
echo "=== across ${#NAMES[@]} scenes, RTF cap $TARGET_RTF ==="
python3 - "$RESULTS" "$SCENES" "${NAMES[@]}" <<'PY'
import json, os, statistics as st, sys
results, scenes, names = sys.argv[1], sys.argv[2], sys.argv[3:]
TAIL = {"stair_f3_to_f1_descent": 56.5, "return_to_first_floor_lobby": 10.9}
rows, kinds = [], {}
for n in names:
    seed = n.rsplit("_s", 1)[1].split("_")[0]
    p = os.path.join(results, n, "mission_stage_timing.json")
    if not os.path.isfile(p):
        kind = "no timing file"
        try:
            console = open(os.path.join(results, n + ".console.log"), encoding="utf-8", errors="replace").read()
            if "[runtime FAIL]" in console:
                kind = "preflight failed"
        except OSError:
            pass
        kinds[kind] = kinds.get(kind, 0) + 1
        rows.append((seed, None, None, None, kind, "")); continue
    try:
        d = json.load(open(p))
    except (OSError, ValueError) as exc:
        kinds["invalid timing file"] = kinds.get("invalid timing file", 0) + 1
        rows.append((seed, None, None, None, "invalid timing file: " + str(exc), "")); continue
    stage = {s["name"]: s.get("sim_duration_sec") or 0.0 for s in d.get("stages", [])}
    expl = sum(v for k, v in stage.items() if "exploration" in k)
    fr = str(d.get("failure_reason") or "")
    completed = d.get("status") == "completed"
    # A failed baseline state is generic. It does not establish an attitude loss.
    # Look at the route's specific reason, and distinguish preparation failures.
    route_reason = ""
    try:
        route = json.load(open(os.path.join(results, n, "scanplanner_route_summary.json")))
        route_reason = str(route.get("failure_reason") or route.get("reason") or "")
    except (OSError, ValueError):
        pass
    evidence = fr + " " + route_reason
    startup_failed = (not completed and not stage and d.get("sim_duration_sec") is None
                      and d.get("exploration_started_wall_time") is None)
    kind = ("COMPLETED" if completed else
            "startup failed" if startup_failed else
            "ran out of time" if "deadline" in fr else
            "went down" if ("attitude_lost" in evidence or "lost_upright" in evidence) else
            "route failed" if ("ROUTE_FAILED" in fr or "EXPLORATION_FAILED" in fr or "no_progress" in evidence) else "other")
    kinds[kind] = kinds.get(kind, 0) + 1
    fixed = sum(v for k, v in stage.items() if "exploration" not in k and k not in TAIL)
    for k, med in TAIL.items():
        fixed += stage.get(k, 0.0) if (completed and stage.get(k)) else med
    acc = os.path.join(results, n, "three_floor_rl_acceptance.json")
    balls = ""
    if os.path.isfile(acc):
        c = json.load(open(acc)).get("checks", {})
        # The acceptance flags are "zero_..." predicates, so True is the GOOD
        # case.  Printing them as FN=/FP= inverted the reading for a whole
        # batch: FN=False looked like "no false negative" when it means balls
        # were missed.  Say what happened instead of naming the flag.
        missed = c.get("zero_false_negative_dangers")
        alarms = c.get("zero_false_positive_dangers")
        def verdict(ok):
            return "ok" if ok is True else ("BAD" if ok is False else "?")
        balls = "missed=%-3s falsealarm=%-3s %s" % (
            verdict(missed), verdict(alarms),
            "CLEAN" if (missed is True and alarms is True) else "")
    # scene shape, so a slow seed can be told apart from a slow robot
    shape = ""
    try:
        mcp = os.path.join(results, n, "mission_config.json")
        mc = json.load(open(mcp))
        wps = [w for fl in mc["floors"] for w in fl["waypoints"]]
        import re
        bends = sum(1 for w in wps if re.search(r"_g[34]_path_\d+$", w["note"]))
        shape = "%dwp/%dbend" % (len(wps), bends)
    except Exception:
        pass
    if startup_failed:
        rows.append((seed, None, None, None, kind + " (see roslaunch.log)", shape))
    else:
        rows.append((seed, expl, fixed, expl + fixed, "%s %s" % (kind, balls), shape))
print("%-10s %8s %8s %10s %-12s %s" % ("seed", "explor", "fixed", "projected", "scene", "outcome"))
for seed, e, f, t, k, shape in rows:
    if e is None: print("%-10s %8s %8s %10s %-12s %s" % (seed, "-", "-", "-", shape, k))
    else: print("%-10s %8.1f %8.1f %10.1f %-12s %s" % (seed, e, f, t, shape, k))
print("\noutcomes over %d scenes:" % len(rows))
for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print("  %-22s %2d  (%.0f%%)" % (k, v, 100 * v / len(rows)))
ok = [r for r in rows if r[1]]
if len(ok) >= 2:
    e = [r[1] for r in ok]; t = [r[3] for r in ok]
    print("\nexploration  min %.1f  median %.1f  max %.1f" % (min(e), st.median(e), max(e)))
    print("projected    min %.1f  median %.1f  max %.1f   (600 s limit)" % (min(t), st.median(t), max(t)))
    over = [r[0] for r in rows if r[3] and r[3] > 600]
    print("\nscenes projected over 600 s: %s" % (", ".join(over) if over else "none"))
PY
