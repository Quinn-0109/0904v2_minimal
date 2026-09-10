# 0904 完整功能对齐检查（2026-09-09）

结论：在用户指定的 `run_multiseed.sh -> run_native.sh -> mission` 运行方式下，未发现 minimal 丢失 0904 的任务功能。运行资产、任务逻辑和离线结果已核对；尚未完成同步后版本的 Ubuntu 全程仿真，不能把本报告当作新的整场 PASS 证明。

比较对象是 Windows 上的 `Desktop/0904` 与 `Desktop/0904v2_minimal`。Ubuntu 用户补做了真实动态库加载检查，五个库均成功；Ubuntu 所有文件是否与 Windows 新补丁一致仍以安装器校验和本机实际运行为准。

## 文件与启动链

| 项目 | 结果 |
| --- | --- |
| runtime_assets/SimEnv | 1,561 个文件逐文件相同，无缺失；包含验证控制器、两份 RL 策略、机器人/传感器、世界、楼宇控制 |
| SCAN-Planner/src | 516 个文件逐文件相同，无缺失 |
| simenv_bridge | 所有当前任务源码、launch、配置、测试和 vendor 源码相同；运行脚本仅增加部署环境/依赖检查 |
| 该任务树缺少的 80 项 | 均为历史 `.before_*` 备份或 `__pycache__`，不是运行必需源码 |
| SimEnv-master/src | 没有缺少文件；差异包括缓存、两个 CMake 占位文件、两份 FixedStand 历史副本 |
| FixedStand | 启动脚本用相同 vendor 源覆盖目标源码；指定运行方式跳过 RL 重编译、执行相同验证二进制 |
| 新增 ExplorationWS/FUEL/FAST-LIO 框架 | 保留；指定 SCAN 主任务没有转去运行该入口 |

核实保留的功能：official 多 seed 场景生成；物理求解器参数；入口台阶；平地/楼梯策略切换；三层十二房双视点扫描；红球候选、确认、重扫、去重与反虚警；楼梯上行/下行；返回一楼大厅中心；任务计时；完整验收 JSON；轨迹与红球真值可视化。录像节点仍在，默认关闭录像编码；helper 现在尊重显式 `RECORD_CAMERA_VIDEO=true`。

## 实际离线运行

两棵树分别从 official generator 开始，执行场景生成、prepare、randomize、`--validate-config --require-runtime-files`。仅归一化输出目录/源码根目录后比较，其余字段未删减。

| seed | 航点 | 路线 m | 真球 | 两版校验 | 两版输出 |
| --- | ---: | ---: | ---: | --- | --- |
| 20 | 101 | 327.669 | 6 | 均通过 | 一致 |
| 111 | 111 | 340.136 | 6 | 均通过 | 一致 |
| 70707 | 105 | 333.533 | 4 | 均通过 | 一致 |
| 4242 | 105 | 333.170 | 4 | 均通过 | 一致 |

每个 seed 比较 8 项：原始 layout、danger_truth、原始 world/model、运行时 layout/mission_config、运行时 world/model，全部相同。world 比较包括物理求解器块。此项说明同一 seed 的静态场景和规划一致，不证明动态运动或检测召回必然相同。

## 现有测试

每棵树实际完成相同的 70 项单元测试，均为 65 项通过、4 项断言失败、1 项错误；两版失败项相同，没有发现 minimal 新引入的单元测试失败。红球检测 21 项、规划器任务 4 项、记录格式 3 项、家具可视化 10 项、录像 4 项均通过。

已有失败不可忽略，但不是迁移差异：

- stable20 两项仍断言 gain=1.45，当前两版配置均为 1.25。
- 三层配置测试仍要求首段走廊 `plane_forward_only`，当前两版均未满足。
- 红球校验器测试认为将 `red_ball_minimum_projected_gap_rad` 降到 0.05 应被拒绝，当前两版均接受。这是共同存在的校验边界问题，本次没有改参数或放宽测试。
- 一个 Docker 历史入口测试读取两份归档均不存在的 `run_three_floor_rl_docker.sh`。

三层完整测试套件在两版均达到本地设置的 360 秒限时，未完整结束。随后单独执行其中 23 个快速测试；上述 70 项统计包含这 23 项，不包含未完成的 4 个重场景测试。不能宣称整套测试全通过。

## 本轮部署修正

1. 接入 `.portable/devel` 的库与 Python 包路径，Python 源包优先于写死旧 `/workspace/0904_v2` 路径的转接文件。
2. 关节插件库同步 0904 的二进制（源码一致）。不改 RL 步态、路线、扫描或物理参数。
3. 修正第一版新增检查器的误报：先加载 Gazebo setup；在 gazebo_ros_control 和 default_robot_hw_sim 的依赖上下文中检查关节插件。
4. 用户在 Ubuntu 对 `libgazebo_ros_control.so`、`libdefault_robot_hw_sim.so`、`libunitree_legged_control.so`、深度相机、RGB 相机库的加载结果均为 OK。此前三项检查失败不能作为这些功能缺失的证据。
5. 修复两个空的 CMake 工作空间入口，以普通 include 文件代替归档丢失的软链；本轮没有执行 Linux 源码重编译。
6. helper 正确处理未探索时的 null 计时；不把准备失败判成摔倒，不补上 67.4 秒返程估计。启动前检查失败会直接显示原因，汇总标为 preflight failed。

检查器应使用与实际运行一致的环境，依据官方 [gzserver 启动脚本](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/noetic-devel/gazebo_ros/scripts/gzserver) 和 [gazebo_ros_control 依赖声明](https://github.com/ros-simulation/gazebo_ros_pkgs/blob/noetic-devel/gazebo_ros_control/CMakeLists.txt)。

## 尚待确认

- 安装修正版补丁后，Ubuntu 的整套启动前检查及全程仿真。
- 最终验收必须是 `three_floor_rl_acceptance.json` 的 passed=true；仅跑完楼层、只出现库加载 OK、或仅达到 600 秒内，都不能替代完整验收。
- Windows minimal 不带完整 third_party；用户已在 Ubuntu 补入。此次 Ubuntu 的控制器依赖检查通过，但没有逐字节核验其完整 Torch 库与原运行机的版本一致性。
- 原 0904 已存在的漏球、虚警、偶发摔倒和时间预算问题不会因目录对齐自动消失。

安装器保留备份；本报告及代码同步不覆盖历史验收证据，历史 minverify9 等记录不代表此次修改后的实跑结果。
