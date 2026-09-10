# 探索链迁移（进入 0904v2_minimal）—— 2026-09-09

把 `simenv_exploration_20260908`（三层探索链独立包：FUEL-lite 楼层探索 +
FAST-LIO 定位 + 楼梯上/下行 + Goal Executor，**无危险源检测/录像/视觉AB**）
按本 bundle 的结构迁移进来。**强约束：不改 0904 原始任何文件**——
`run.sh / preflight.sh / verify_bundle.sh / README.md / MANIFEST.sha256 /
SimEnv-master/ / runtime_assets/ / mission/run_native.sh /
mission/work/mounts/{SimEnv,SCAN-Planner}/` 全部原样（见文末验证）。

## 1. 新增文件（全部为新增目录/文件，无一修改原有文件）

```
0904v2_minimal/
├── run_exploration.sh                        ← 探索链唯一入口（对标 run.sh）
└── mission/work/mounts/ExplorationWS/        ← 对标 mounts/SimEnv 的任务挂载
    ├── README_exploration.md                 ← 本文档
    ├── src/CMakeLists.txt                    ← catkin toplevel（拷自 SCAN-Planner 约定）
    ├── src/simenv_exploration/               ← 探索算法包（23 节点 + 11 库 + 4 launch）
    ├── src/FAST_LIO/                         ← 定位前端（simenv_mid360.launch 指向新包名）
    ├── generated_building/elevator_three_floor_debug/   ← full_flow 2026-09-08 场景
    └── scripts/
        ├── run_exploration_three_floor.sh    ← runner（0904 风格：自建 build space、
        │                                        任务独占 Xvfb、torch 短名补链、只杀本进程组）
        └── aggregate_full_flow_stats.py 等 4 个统计工具
```

运行时产物：`.portable_build/exploration_{build,devel}`（对标 scanplanner_build_v2）、
`results/<RUN_NAME>/`。

## 2. 跑法

```bash
# 容器内（与 0904 同一镜像）：
docker exec <容器> bash /workspace/0904_v2/run_exploration.sh expl_smoke1

# 或原生机（Ubuntu 20.04 + ROS noetic + /opt/libtorch）：
bash run_exploration.sh my_run
```

- 单跑单目录：结果在 `results/<RUN_NAME>/`；成功判据
  `logs/stair_descent.json` phase=`FIRST_FLOOR_RETURNED`，退出码 0/1。
- runner 会先自建工作区（首次约 2-4 分钟，只构建 simenv_exploration +
  fastlio 两个包，不碰 SimEnv-master），再起任务独占 Xvfb(:99)，然后
  `roslaunch simenv_exploration fuel_semantic_fastlio_exploration.launch`
  （生产覆盖与独立交付版一致：descent 0.80 / ascent_third 0.70 / 等 6 项）。
- 聚合统计：`python3 mission/work/mounts/ExplorationWS/scripts/aggregate_full_flow_stats.py \
  results/expl_* --sim-time --output results/expl_stats.md`

## 3. 对 bundle 既有资产的**只读复用**（不拷贝、不修改）

| 资产 | 位置（0904v2_minimal 内） | 说明 |
|---|---|---|
| RL 平地/爬梯策略 | `runtime_assets/SimEnv/src/unitree_guide/logs/*.pt` | 与独立交付版逐字节相同（sha256 已核对） |
| junior_ctrl 控制器 | `SimEnv-master/.portable/devel/lib/unitree_guide/` | **0904 验证版**（27/27 PASS 的 840f79ab…，含 RC1 就绪门 0.26/混合设备/预加载），三份副本一致 |
| Gazebo 插件 | `SimEnv-master/.portable/devel/lib/lib{livox_laser_simulation,unitree_legged_control}.so` | GAZEBO_PLUGIN_PATH 只加不改 |
| unitree_guide / building_generator_* / Mid360 / uav_simulator / rpg_vikit 源 | `runtime_assets/SimEnv/src`（次序在后 `SimEnv-master/src`） | 与 0904 mission 相同的解析次序：探索挂载 → assets → master |

junior_ctrl 解析：`scripts/simenv_expl_devel_resolver.py`（对标 0904 的
devel_space_resolver）按 `SIMENV_EXPL_DEVEL_SPACE` → mount 本地 devel(devel_0803b)
的次序找可执行文件；runner 已把它指到 `.portable/devel`。

## 4. 迁移相对独立交付版的 5 处代码适配（均为可移植性，非行为改动）

1. `scripts/classic_trotting_controller.py`：junior_ctrl 由硬编码
   `devel_0803b/.private/...` 改为上述 resolver（原句为作者机器路径，bundle 内必失败）。
2. `launch/mapping_test.launch`：junior_ctrl 的 `LD_LIBRARY_PATH` 由硬编码
   `/root/autodl-tmp/deps/libtorch/lib` 改为
   `$(optenv LIBTORCH_LIBRARY_PATH /opt/libtorch/lib)`（0904 同款写法）。
3. runner 重写为 0904 风格：环境隔离 + 拒绝并发 ROS 会话 + 只杀本进程组
   （原版的 `pkill -9 -f roslaunch/gzserver` 全局清场**会误杀 0904 mission**，已废除）。
4. `scripts/second_floor_exploration_manager.py`：`~visualization_script` 参数加默认值
   `""`（可视化脚本不随包，launch 已设 `auto_generate_combined_visualization=false`，
   该路径永不使用；原写法在参数缺失时于 `__init__` 抛 KeyError，live 冒烟实测两个
   manager 实例秒死）。独立交付版同步修复。
5. runner 并发会话门收紧：原来对全命令行做子串匹配，`tail -F .../step_roslaunch.log`
   这类无辜进程（命令行里含 "roslaunch" 字样）也会被拒；改为 comm 精确匹配 +
   roslaunch 按路径锚定匹配，僵尸进程本就被 `stat !~ Z` 排除。

## 5. 与独立交付验证状态的已知偏差

- **控制器二进制**：独立交付的链路当年跑在 0803b-era 控制器上；bundle 内唯一
  可用的是 0904 验证版（RC1 FixedStand 就绪门 0.12→0.26 等）。差异只影响
  站立→RL 交接的稳妥性（更保守），步态策略文件相同。若要严格复刻，可把
  `SIMENV_EXPL_DEVEL_SPACE` 指向任一含 0803b junior_ctrl 的 devel 空间。
- **unitree_guide 支撑源**：用 runtime_assets 版（0904 27/27 验证所用），
  与 full_flow 2026-09-08 同步树的 unitree_guide 存在版本差（multi_floor
  launch 的出生关节角 0.9/-1.8 vs master 版 1.36/-2.65 等）。runner 起手即
  校验 launch 图与全部节点可执行，冒烟见 `results/`。
- 场景/策略/算法代码：与独立交付逐字节一致。

## 6. 已知随机失败（与独立交付 README §8 相同，重跑稀释）

FAST-LIO 失锁 ~30%；下梯 flight-B 跌倒 ~12%；SEGMENT_TURN 瞬翻/冻结批次相关；
上行 route_error 死锁与 F2→F3 z≈4.0 死锁低概率。失败轮先看
`step_roslaunch.log` 尾部 + 各段 json 相位再决定重跑。

## 7. 不影响原始功能的验证清单

- `bash verify_bundle.sh` 仍 PASS（MANIFEST 只校验列出的 3654 个原文件，
  新增文件不进入校验；新挂载无外部符号链接）。
- 探索 runner 全程不写 `SimEnv-master/`、`runtime_assets/`、既有 mounts、
  `mission/run_native.sh`；构建/结果只落在 `.portable_build/exploration_*` 与
  `results/<RUN_NAME>`。
- 0904 mission 的运行路径（run.sh → run_native.sh → simenv_bridge）未被触碰，
  ROS_PACKAGE_PATH 等环境仅在探索 runner 进程内设置。
