# 当前自主探索系统详细总结：官方 TARE + FAR 架构

## 1. 文档目的

本文档描述 `/root/autodl-tmp/code/restore/SimEnv` 当前已经落地的自主探索架构、
ROS 节点关系、数据流、路径规划回退、安全检查、终止逻辑、构建方式和运行诊断方法。

当前系统已经停止把自研 `exploration_graph` 和
`hierarchical_goal_selector` 继续扩展成 TARE 的替代实现，探索决策和全局可见图导航分别
交给官方 TARE Planner 与官方 FAR Planner。

## 2. 当前最终架构

```text
Gazebo / A1 sensors
        │
        ▼
FAST-LIO
  /Odometry
  /cloud_registered
        │
        ├──────────────────────────────┐
        ▼                              │
fastlio_voxel_mapper                   │
  Octomap / voxel map                 │
  2D OccupancyGrid                    │
  SCAN-lite collision service         │
        │                              │
        │ /simenv/voxel_floor_projection
        ▼                              │
tare_interface.py ◄───────────────────┘
  OccupancyGrid → PointXYZI terrain cloud
  FAST-LIO frame → map header normalization
        │
        ├─ /tare/state_estimation_at_scan
        ├─ /tare/registered_scan
        ├─ /tare/terrain_map
        └─ /tare/terrain_map_ext
        │
        ▼
官方 TARE Planner
  全局/局部覆盖规划
  未覆盖区域选择
  探索完成与返航决策
        │
        │ /tare/way_point
        ▼
far_interface.py
  将 TARE waypoint 提交给 FAR
        │
        │ /far/goal_point
        ▼
官方 FAR Planner
  动态可见图
  障碍物轮廓与连通关系
  全局路径搜索
        │
        ├─ /far/viz_path_topic
        └─ /far/way_point
        │
        ▼
far_interface.py
  FAR Marker → nav_msgs/Path
  路径方向归一化
  A* 失败回退
        │
        │ /simenv/far_path
        ▼
SCAN-lite
  双圆柱足迹碰撞检查
  局部路径修复
  未知空间策略
        │
        │ /exploration_goal
        ▼
Goal Executor
  航向控制
  速度控制
  目标完成/失败反馈
        │
        │ /cmd_vel
        ▼
RL locomotion
```

## 3. 保留、替换和停用的模块

### 3.1 保留且继续使用

- FAST-LIO：提供实时里程计和注册点云。
- Voxel Mapper / Octomap：维护三维占据地图和二维地面投影。
- SCAN-lite：对规划路径进行机器人足迹级安全检查。
- Goal Executor：把安全航点转换为速度控制。
- RL locomotion：执行 `/cmd_vel`。
- A*：仅作为 FAR 失败后的 fallback。
- 既有 telemetry：继续记录真实轨迹和实验数据。

### 3.2 已替换

| 原模块 | 当前替代 |
|---|---|
| FUEL-lite frontier 目标选择 | 官方 TARE Planner |
| 自研 frontier cluster → region graph | 官方 TARE 覆盖规划内部表示 |
| 自研 hierarchical goal selector | 官方 TARE 全局/局部探索决策 |
| A* 主规划器 | 官方 FAR Planner |
| 自研区域覆盖状态机 | TARE 覆盖状态与探索完成判断 |

### 3.3 已从活动运行链移除

以下节点不会再被当前主 launch 启动：

- `frontier_cluster_node.py`
- `exploration_graph_node.py`
- `hierarchical_explorer_node.py`
- `corridor_side_filter`
- `branch_scheduler`
- 固定 depth budget
- 固定 breadth budget
- 基于房门的探索状态机
- 基于房间中心线的人工调度规则

这些源码暂时保留，目的是可以复现旧实验和对比历史结果，不代表仍在当前控制链中运行。

## 4. 第三方官方规划器

### 4.1 TARE Planner

本地位置：

```text
third_party/tare_planner
```

上游项目：

```text
https://github.com/caochao39/tare_planner
```

当前使用：

- ROS 分支：`melodic-noetic`
- 节点：`tare_planner/tare_planner_node`
- 场景参数：官方 `indoor.yaml`
- 自动开始：`kAutoStart=true`
- 探索结束后返航：`kNoExplorationReturnHome=true`

TARE 负责：

- 根据注册点云更新环境覆盖情况；
- 根据未覆盖点和 frontier 决定探索方向；
- 维护 keypose graph；
- 进行局部覆盖规划和全局子空间规划；
- 发布下一探索 waypoint；
- 判断探索是否结束并规划返航。

### 4.2 FAR Planner

本地位置：

```text
third_party/far_planner
```

上游项目：

```text
https://github.com/MichaelFYang/far_planner
```

当前使用：

- ROS 分支：`melodic-noetic`
- 节点：`far_planner/far_planner`
- 参数文件：`config/far_simenv.yaml`
- 世界坐标系：`map`
- 环境模式：静态环境
- A1 机器人规划尺寸：`0.42 m`

FAR 负责：

- 从局部/扩展地形点云提取可通行点和障碍点；
- 构造障碍物轮廓；
- 建立并更新动态可见图；
- 将 TARE waypoint 作为目标；
- 在可见图上搜索全局路径；
- 连续发布局部 waypoint 和全局路径 Marker。

## 5. TARE 输入适配

实现文件：

```text
scripts/tare_interface.py
```

该节点只做 ROS 数据格式适配，不做 frontier 评分或探索决策。

### 5.1 输入

| 输入话题 | 类型 | 来源 |
|---|---|---|
| `/Odometry` | `nav_msgs/Odometry` | FAST-LIO |
| `/cloud_registered` | `sensor_msgs/PointCloud2` | FAST-LIO |
| `/simenv/voxel_floor_projection` | `nav_msgs/OccupancyGrid` | Voxel Mapper |

### 5.2 输出

| 输出话题 | 类型 | 使用者 |
|---|---|---|
| `/tare/state_estimation_at_scan` | `nav_msgs/Odometry` | TARE、FAR |
| `/tare/registered_scan` | `sensor_msgs/PointCloud2` | TARE、FAR |
| `/tare/terrain_map` | `sensor_msgs/PointCloud2` | TARE、FAR |
| `/tare/terrain_map_ext` | `sensor_msgs/PointCloud2` | TARE、FAR |

### 5.3 OccupancyGrid 到 PointXYZI

官方 FAR 按点云强度区分可通行点与障碍点：

```text
intensity < Util/terrain_free_Z  → free
intensity >= Util/terrain_free_Z → obstacle
```

当前参数：

```text
Util/terrain_free_Z = 0.2
```

适配器编码：

```text
free cell     → intensity = 0.0
occupied cell → intensity = 1.0
unknown cell  → 不发布
```

因此 free/obstacle 分类与官方 FAR 的读取逻辑一致。

`/tare/terrain_map` 是机器人附近约 15 米的局部裁剪；
`/tare/terrain_map_ext` 是当前全部已知二维占据区域。

### 5.4 点云采样

默认：

- 障碍栅格每格保留；
- free 栅格按 2 倍步长降采样；
- OccupancyGrid 数值 `>=50` 判为障碍；
- 未知栅格不伪造为 free。

这样可以限制地形点云规模，同时保留墙体和障碍边界。

## 6. 坐标系处理

官方 TARE 源码将世界坐标系固定为 `map`。

FAST-LIO 默认使用 `camera_init` 作为初始世界坐标系。当前适配方式是：

```text
FAST-LIO camera_init 数值坐标
             ↓
保持 x/y/z 数值不变
             ↓
统一发布为 map header
```

这是一种坐标系别名归一化，不是使用 Gazebo 真值进行坐标变换，也没有读取裁判里程计。

以下数据全部使用相同的数值坐标：

- TARE odometry；
- TARE registered scan；
- terrain map；
- FAR goal；
- FAR path；
- Goal Executor goal；
- SCAN-lite 查询位置。

FAR 订阅适配后的 `/tare/state_estimation_at_scan`，避免尝试查找不存在的
`map ↔ camera_init` TF。

## 7. TARE 到 FAR 的目标传递

官方 TARE 发布：

```text
/tare/way_point
geometry_msgs/PointStamped
```

`far_interface.py` 接收该目标并发布：

```text
/far/goal_point
geometry_msgs/PointStamped
```

连续位置接近的 TARE waypoint 会按距离阈值合并，避免相同目标不断重置 FAR。

如果 TARE 明显改变目标，当前 FAR 请求和 Goal Executor 航点允许被新目标替换。

## 8. FAR Path 转换

官方 FAR 不原生发布 `nav_msgs/Path`。

它发布：

- `/far/way_point`：当前局部导航 waypoint；
- `/far/viz_path_topic`：完整全局路径，类型为
  `visualization_msgs/Marker`，Marker 类型为 `LINE_STRIP`。

当前适配器读取真实 `/far/viz_path_topic`：

1. 提取 `Marker.points`；
2. 去除距离过近的重复点；
3. 根据机器人当前位置判断路径方向；
4. 必要时将路径反转为机器人到目标方向；
5. 转换为 `nav_msgs/Path`；
6. 发布到 `/simenv/far_path`。

所以 `/simenv/far_path` 是官方 FAR 真实可见图路径的 ROS 类型转换，不是用 A*
重新生成的替代路径。

## 9. FAR 与 A* 回退逻辑

路径选择流程：

```text
收到 TARE waypoint
        │
        ▼
发送 /far/goal_point
        │
        ▼
等待新的 FAR LINE_STRIP
        │
   ┌────┴────┐
   │         │
成功       超时/无有效路径
   │         │
FAR path     ▼
   │      A* fallback
   └────┬────┘
        ▼
SCAN-lite
```

默认 FAR 等待时间：

```text
far_plan_timeout = 5.0 s
```

只有满足以下情况才调用 A*：

- FAR 未在超时时间内返回请求之后生成的新路径；
- FAR 返回的 Marker 为空；
- FAR 路径点数少于有效路径要求。

运行状态中：

```text
backend = far
```

表示使用官方 FAR。

```text
backend = astar_fallback
```

表示 FAR 失败或超时后才启用了 A*。

## 10. SCAN-lite 安全检查

FAR 或 A* 输出的路径不会直接发给机器人。

当前流程：

1. 按 waypoint spacing 稀疏化；
2. 计算每个路径点的切向 yaw；
3. 调用 Voxel Mapper 的双圆柱足迹服务；
4. 检查机器人前后两个碰撞圆柱；
5. 检查路径点和路径段；
6. 如果局部碰撞，尝试在有限半径内寻找替代点；
7. 重新检查修复后的完整路径；
8. 只发布安全的前视 waypoint。

服务：

```text
/tare_far_voxel_mapper/check_twin_cylinder
simenv_competitor/CheckTwinCylinder
```

默认未知空间策略：

```text
scan_lite_unknown_policy = penalize
```

即未知区域不是一律拒绝，但在局部修复评分中会受到惩罚。

## 11. Goal Executor 执行方式

`far_interface.py` 不直接发布 `/cmd_vel`。

它从安全路径中选取约 1.2 米前视位置，然后发布：

```text
/exploration_goal
geometry_msgs/PoseStamped
```

Goal Executor 负责：

- 转向；
- 加速和减速；
- 目标容差；
- 速度限制；
- 卡住/超时判断；
- 发布执行结果。

执行反馈：

```text
/simenv/goal_execution_result
std_msgs/String(JSON)
```

执行一个前视 waypoint 后，系统会请求新的 FAR 路径，而不是一次性盲目执行完整旧路径。
这样可以利用 FAR 持续更新的动态可见图。

## 12. 探索完成、返航和时间限制

### 12.1 TARE 探索完成

TARE 判断没有需要继续覆盖的有效区域后发布探索完成状态，并切换到返航路径。

`far_interface.py` 收到探索完成消息后不会立即停止，而是继续执行 TARE 发布的返航目标。

机器人回到初始位置约 0.5 米范围内后：

- 发布 `/simenv/finalize_voxel_map=true`；
- 发布 `/simenv/mission_complete=true`；
- 保存最终地图；
- 关闭探索任务。

### 12.2 最大运行时间

`maximum_duration` 是从 FAR 接口节点启动开始计算的硬墙钟预算，当前默认值为
300 秒。它包含 FAST-LIO、地图、运动控制和 TARE 首目标的启动等待时间。

到达时间限制后会进入 `time_limit` 终止，触发地图保存。

独立的 0.25 秒监督定时器保证即使 FAR 或 SCAN-lite 服务调用变慢，任务仍会在
300 秒预算附近终止。

## 13. 当前 Launch 结构

主 launch：

```text
launch/hierarchical_fastlio_exploration.launch
```

当前只组合以下四个功能 launch：

```text
hierarchical_fastlio_exploration.launch
├── fastlio_mapping.launch
├── tare_exploration.launch
├── far_navigation.launch
└── goal_executor.launch
```

### 13.1 `fastlio_mapping.launch`

启动：

- Gazebo 场景；
- A1 模型；
- RL 控制器；
- FAST-LIO；
- `fastlio_voxel_mapper`；
- SCAN-lite 查询服务。

### 13.2 `tare_exploration.launch`

启动：

- `tare_interface.py`；
- 官方 `tare_planner_node`；
- 官方 indoor 参数；
- TARE 输入输出话题重定向。

### 13.3 `far_navigation.launch`

启动：

- 官方 `far_planner`；
- `far_interface.py`；
- TARE → FAR 目标桥；
- FAR Marker → Path；
- SCAN-lite；
- A* fallback。

### 13.4 `goal_executor.launch`

启动现有 Goal Executor 和 RL 模式初始化。

## 14. 关键节点列表

完整主 launch 静态展开后包含：

```text
/gazebo
/urdf_spawner
/a1_gazebo/controller_spawner
/robot_state_publisher
/junior_ctrl
/fastlio_pointcloud_adapter
/fast_lio_mapping
/map_to_fastlio_origin
/tare_far_voxel_mapper
/tare_interface
/sensor_coverage_planner/tare_planner_node
/far_planner
/far_interface
/goal_executor_rl_bootstrap
/goal_executor
/tare_far_telemetry
```

不会出现：

```text
/frontier_cluster_node
/exploration_graph_node
/hierarchical_explorer_node
```

## 15. 关键话题表

| 话题 | 类型 | 发布者 | 订阅者 |
|---|---|---|---|
| `/Odometry` | `nav_msgs/Odometry` | FAST-LIO | TARE adapter |
| `/cloud_registered` | `sensor_msgs/PointCloud2` | FAST-LIO | mapper、TARE adapter |
| `/simenv/voxel_floor_projection` | `nav_msgs/OccupancyGrid` | mapper | TARE adapter、fallback |
| `/tare/state_estimation_at_scan` | `nav_msgs/Odometry` | TARE adapter | TARE、FAR、FAR adapter |
| `/tare/registered_scan` | `sensor_msgs/PointCloud2` | TARE adapter | TARE、FAR |
| `/tare/terrain_map` | `sensor_msgs/PointCloud2` | TARE adapter | TARE、FAR |
| `/tare/terrain_map_ext` | `sensor_msgs/PointCloud2` | TARE adapter | TARE、FAR |
| `/tare/way_point` | `geometry_msgs/PointStamped` | TARE | FAR adapter |
| `/far/goal_point` | `geometry_msgs/PointStamped` | FAR adapter | FAR |
| `/far/way_point` | `geometry_msgs/PointStamped` | FAR | diagnostics |
| `/far/viz_path_topic` | `visualization_msgs/Marker` | FAR | FAR adapter |
| `/simenv/far_path` | `nav_msgs/Path` | FAR adapter | diagnostics |
| `/exploration_goal` | `geometry_msgs/PoseStamped` | FAR adapter | Goal Executor |
| `/simenv/goal_execution_result` | `std_msgs/String` | Goal Executor | FAR adapter |
| `/cmd_vel` | `geometry_msgs/Twist` | Goal Executor | RL locomotion |
| `/simenv/tare_far_status` | `std_msgs/String` | FAR adapter | diagnostics |
| `/simenv/finalize_voxel_map` | `std_msgs/Bool` | FAR adapter | mapper |

## 16. 参数文件

### FAR 参数

```text
config/far_simenv.yaml
```

主要参数：

```yaml
world_frame: map
robot_dim: 0.42
vehicle_height: 0.45
sensor_range: 12.0
terrain_range: 15.0
local_planner_range: 5.0
is_static_env: true
is_multi_layer: false
Util/terrain_free_Z: 0.2
Util/obs_inflate_size: 2
```

### TARE 参数

以官方 `indoor.yaml` 为基础，launch 覆盖：

```yaml
kAutoStart: true
kNoExplorationReturnHome: true
pub_waypoint_topic_: /tare/way_point
sub_state_estimation_topic_: /tare/state_estimation_at_scan
sub_registered_scan_topic_: /tare/registered_scan
sub_terrain_map_topic_: /tare/terrain_map
sub_terrain_map_ext_topic_: /tare/terrain_map_ext
```

## 17. 日志与输出

每次运行至少生成：

```text
<output_dir>/
├── goal_execution_history.json
├── voxel_map.bt
├── floor_projection.*
├── telemetry.*
└── logs/
    └── tare_far_summary.json
```

当前主流程会在任务结束、超时或正常 ROS shutdown 时启动独立离线后处理进程。该进程
不受 roslaunch 的 15 秒关闭期限影响，会等待地图与轨迹文件写稳后输出：

```text
<output_dir>/visualization/
├── 12_room_layout_and_trajectory.png
├── 13_room_truth_trajectory_lidar_goals.png
├── visualization_summary.json
└── baseline_visual_analysis.md
```

官方 TARE 不产生旧版 `room_recognition_history.json`。对于官方 TARE/FAR 运行，
图 13 使用同步 Gazebo 真值轨迹、最终 LiDAR/OctoMap 覆盖和离线真值布局生成，不把离线
布局提供给在线规划器。

`tare_far_summary.json` 包含：

```json
{
  "architecture": "official_tare_official_far",
  "far_path_source": "/far/viz_path_topic",
  "fallback": "astar_only_after_far_failure",
  "finished": false,
  "elapsed_sec": 0.0,
  "tare_goal_generation": 0
}
```

## 18. 已完成验证

已经完成：

1. 官方 TARE 在当前真实路径重新编译成功；
2. 官方 FAR 源码下载并编译成功；
3. 主 catkin 工作空间完整编译成功；
4. 新 Python 节点通过语法检查；
5. 四个 launch 文件通过 XML 检查；
6. 三个 ROS workspace 叠加后均可被 `rospack find`；
7. 主 launch 可以完整静态展开；
8. TARE、FAR 和两个接口完成 ROS 进程级同时启动测试；
9. 官方 FAR 与官方 TARE 均未出现动态库或消息类型启动崩溃；
10. 旧自研探索节点没有出现在当前主 launch 节点列表。

尚未完成：

- 新参数架构完整 300 秒实跑；
- 与旧 FUEL-lite 的同场景覆盖率对比；
- FAR 正常路径与 A* fallback 次数统计；
- 返航全过程验证；
- 多楼层参数验证。

因此当前结论是：接口集成和构建已经完成，但探索效果仍需通过完整仿真结果评估。

## 19. 构建命令

正常情况下当前已经编译完成，不需要每次运行前重新编译。

需要重编时：

```bash
cd /root/autodl-tmp/code/restore/SimEnv

source /opt/ros/noetic/setup.bash

catkin_make \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DTorch_DIR=/root/autodl-tmp/deps/libtorch/share/cmake/Torch \
  -j2

source devel/setup.bash

cd third_party/tare_planner
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -j2
source devel/setup.bash

cd ../far_planner
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -j2
```

## 20. 当前推荐运行命令

```bash
cd /root/autodl-tmp/code/restore/SimEnv

source /opt/ros/noetic/setup.bash
source devel/setup.bash
source third_party/tare_planner/devel/setup.bash
source third_party/far_planner/devel/setup.bash

roslaunch simenv_competitor hierarchical_fastlio_exploration.launch \
  output_dir:=/root/autodl-tmp/code/restore/SimEnv/results/tare_far_300s_run1 \
  maximum_duration:=700.0
```

## 21. 运行时检查命令

确认 TARE 正在给目标：

```bash
rostopic hz /tare/way_point
rostopic echo -n 1 /tare/way_point
```

确认 FAR 正在真实规划：

```bash
rostopic hz /far/viz_path_topic
rostopic echo -n 1 /far/viz_path_topic
```

确认 FAR Path 转换：

```bash
rostopic echo -n 1 /simenv/far_path
```

确认当前使用 FAR 还是 A*：

```bash
rostopic echo /simenv/tare_far_status
```

确认 Goal Executor：

```bash
rostopic echo /simenv/goal_execution_result
rostopic hz /cmd_vel
```

确认地形点云：

```bash
rostopic hz /tare/terrain_map
rostopic hz /tare/terrain_map_ext
rostopic hz /tare/registered_scan
```

## 22. 当前主要风险与下一步

### 22.1 OccupancyGrid 是二维地形近似

当前 TARE/FAR terrain cloud 来自二维地面投影，而不是完整三维 traversability map。
这适合当前第一层平面房间探索，但多楼层、楼梯和悬空障碍需要进一步提供真正的三维
terrain map。

### 22.2 FAR 启动阶段可能 fallback

FAR 需要先积累 terrain cloud 和 registered scan 才能初始化可见图。如果 TARE 很早发布
第一个目标，首次规划可能在 5 秒内没有有效 FAR path，因而使用 A* fallback。

应通过 `/simenv/tare_far_status` 统计，而不能只看机器人是否移动。

### 22.3 TARE 参数仍需场景调优

当前使用官方 indoor 参数。其传感器范围、viewpoint 网格、碰撞余量和 lookahead 距离最初
并非针对 A1 + Mid-360 + 当前建筑尺寸设计，完整实跑后需要根据真实覆盖率调整。

调整应集中在官方 TARE/FAR 参数，不再恢复 corridor/room 人工状态机。

### 22.4 下一轮实验建议记录

- TARE waypoint 数量；
- FAR 成功规划次数；
- A* fallback 次数；
- SCAN-lite 拒绝次数；
- Goal Executor 成功率；
- 总路径长度；
- 已知栅格/体素比例；
- 重复进入区域次数；
- 是否触发 TARE return home；
- 返航最终误差。

## 23. 当前结论

当前代码已经完成从“自研类 TARE 层次探索器”到“官方 TARE + 官方 FAR”的架构转换。

当前系统中：

- TARE 是探索决策主体；
- FAR 是主路径规划器；
- A* 仅是失败回退；
- SCAN-lite 仍负责机器人足迹安全；
- Goal Executor 和 RL locomotion 继续负责运动执行；
- 旧走廊、房门和房间预算规则不再影响当前探索。

下一阶段不应继续增加房门、走廊侧边或固定区域预算规则，而应基于完整 300 秒运行日志，
校准 TARE 覆盖参数、FAR 地形输入质量和两者之间的执行节奏。
