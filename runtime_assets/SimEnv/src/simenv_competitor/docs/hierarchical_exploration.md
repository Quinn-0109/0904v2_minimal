# Official TARE + FAR autonomous exploration architecture

> 中文详细现状、节点/话题表、回退逻辑和运行诊断见
> [`current_tare_far_architecture_zh.md`](current_tare_far_architecture_zh.md)。

## 1. Runtime architecture

The active launch no longer runs the locally developed frontier graph or
region state machine.  Exploration and global navigation are delegated to the
official CMU planners:

```text
FAST-LIO
  │  /Odometry, /cloud_registered
  ▼
Voxel Mapper / OccupancyGrid
  │  /simenv/voxel_floor_projection
  ▼
tare_interface.py
  │  /tare/state_estimation_at_scan
  │  /tare/registered_scan
  │  /tare/terrain_map
  │  /tare/terrain_map_ext
  ▼
Official TARE Planner
  │  /tare/way_point
  ▼
far_interface.py ───────────────┐
  │ /far/goal_point             │ A* fallback only after FAR timeout/failure
  ▼                             │
Official FAR Planner            │
  │ /far/viz_path_topic         │
  ▼                             │
Marker → nav_msgs/Path adapter ◄┘
  │ /simenv/far_path
  ▼
SCAN-lite path/footprint check
  │ /exploration_goal
  ▼
Existing Goal Executor
  │ /cmd_vel
  ▼
Existing RL locomotion
```

Upstream FAST-LIO, the voxel mapper, SCAN-lite, Goal Executor and RL
locomotion are unchanged.

## 2. Official source workspaces

- TARE Planner: `third_party/tare_planner`
- FAR Planner: `third_party/far_planner`
- TARE upstream: <https://github.com/caochao39/tare_planner>
- FAR upstream: <https://github.com/MichaelFYang/far_planner>
- CMU environment: <https://github.com/HongbiaoZ/autonomous_exploration_development_environment>

Both third-party planners are built as independent catkin workspaces.  This
avoids copying their planning algorithms into `simenv_competitor`.

## 3. Adapter responsibilities

### `tare_interface.py`

This is a format adapter, not an exploration implementation.

- republishes the FAST-LIO odometry at each registered scan;
- republishes the registered cloud under the TARE input namespace;
- converts known OccupancyGrid cells to PointXYZI terrain clouds;
- encodes free cells with intensity `0.0`;
- encodes occupied cells with intensity `1.0`;
- publishes a local terrain crop and the full known terrain map.

Official TARE hard-codes its world frame to `map`.  FAST-LIO's initial world
coordinates are therefore exposed under the `map` header without changing
their numeric values.  FAR consumes the same normalized odometry and clouds,
so all components use one consistent coordinate convention.

### `far_interface.py`

This node does not implement FAR.

- submits each TARE exploration waypoint to official FAR on
  `/far/goal_point`;
- consumes FAR's real visibility-graph path marker;
- converts the `LINE_STRIP` points into `nav_msgs/Path`;
- normalizes path order using the current robot pose;
- passes the path through the existing SCAN-lite footprint checks;
- sends one safe look-ahead waypoint to Goal Executor;
- requests a fresh FAR path after each execution step.

Official FAR publishes its complete path as a visualization marker rather
than `nav_msgs/Path`.  `/simenv/far_path` is a lossless ROS type conversion of
that official output.

If no usable FAR path arrives within `far_plan_timeout`, and only then, the
interface runs the retained A* implementation.  Status messages state either
`far` or `astar_fallback`, so fallback use is observable.

## 4. Launch organization

`hierarchical_fastlio_exploration.launch` now only composes:

1. `fastlio_mapping.launch`
2. `tare_exploration.launch`
3. `far_navigation.launch`
4. `goal_executor.launch`

The following old nodes are not launched:

- `frontier_cluster_node.py`
- `exploration_graph_node.py`
- `hierarchical_explorer_node.py`
- corridor-side filter and branch scheduler logic
- fixed depth/breadth room budgets

Their files remain in the repository only for old-run reproducibility.  They
are deprecated and are not part of the active control path.

## 5. Important ROS topics

| Topic | Type | Producer | Consumer |
|---|---|---|---|
| `/tare/terrain_map` | `sensor_msgs/PointCloud2` | TARE adapter | TARE, FAR |
| `/tare/terrain_map_ext` | `sensor_msgs/PointCloud2` | TARE adapter | TARE, FAR |
| `/tare/state_estimation_at_scan` | `nav_msgs/Odometry` | TARE adapter | TARE, FAR |
| `/tare/registered_scan` | `sensor_msgs/PointCloud2` | TARE adapter | TARE, FAR |
| `/tare/way_point` | `geometry_msgs/PointStamped` | official TARE | FAR adapter |
| `/far/goal_point` | `geometry_msgs/PointStamped` | FAR adapter | official FAR |
| `/far/viz_path_topic` | `visualization_msgs/Marker` | official FAR | FAR adapter |
| `/simenv/far_path` | `nav_msgs/Path` | FAR adapter | diagnostics/SCAN-lite |
| `/exploration_goal` | `geometry_msgs/PoseStamped` | FAR adapter | Goal Executor |
| `/cmd_vel` | `geometry_msgs/Twist` | Goal Executor | RL locomotion |

## 6. Build

Build third-party workspaces in underlay order, then rebuild the main
workspace:

```bash
cd /root/autodl-tmp/code/restore/SimEnv
source /opt/ros/noetic/setup.bash
source devel/setup.bash

cd third_party/tare_planner
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -j2
source devel/setup.bash

cd ../far_planner
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -j2
source devel/setup.bash

cd ../..
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -j2
```

## 7. Run

The setup files must be sourced in dependency order:

```bash
cd /root/autodl-tmp/code/restore/SimEnv
source /opt/ros/noetic/setup.bash
source devel/setup.bash
source third_party/tare_planner/devel/setup.bash
source third_party/far_planner/devel/setup.bash

roslaunch simenv_competitor hierarchical_fastlio_exploration.launch \
  output_dir:=/root/autodl-tmp/code/restore/SimEnv/results/tare_far_700s_run1 \
  maximum_duration:=700.0
```

## 8. Runtime verification

The following commands distinguish real FAR planning from fallback:

```bash
rostopic echo /simenv/tare_far_status
rostopic hz /far/viz_path_topic
rostopic echo -n 1 /simenv/far_path
```

The run summary is written to
`<output_dir>/logs/tare_far_summary.json`.  Its architecture field is
`official_tare_official_far`.
