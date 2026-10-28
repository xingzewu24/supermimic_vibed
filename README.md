# SuperMimic on mjlab

基于 **mjlab + MuJoCo-Warp** 的 SuperMimic 非官方复现，使用完整 G1 Inspire 跟踪
默认跟踪 `data/hoi/motions/test` 中的 largebox HOI 动作：身体、手指与物体联合 tracking。
Python 环境统一由 **uv** 管理，直接在服务器上运行。

保留原有的参考动作处理、世界模型、可微想象训练、策略、价值网络及奖励函数；模拟器使用
mjlab 的 `Scene`、`Entity`、`BuiltinPositionActuatorCfg` 和 `Simulation`。
控制频率 50 Hz，每次控制执行 4 个 5 ms 物理子步，使用 `implicitfast` 积分器。
机器人直接加载 `data/tracking/robot/g1_inspire/g1_inspire.xml`，保留 53 个转动关节，
使用 41 个位置执行器和从 URDF 补齐的 12 条 mimic 等式约束。策略输出 41 维动作：身体 29 维、双手主动关节 12 维。

## 环境安装

支持 Linux x86_64、Python 3.11/3.12、NVIDIA GPU。默认 Python 3.11。
针对 **575 系列驱动**固定以下依赖，完整依赖见 `uv.lock`：

| 组件 | 版本 |
|---|---|
| mjlab | 1.6.0 |
| MuJoCo / MuJoCo-Warp | 3.11.0 / 3.11.0 |
| PyTorch / torchvision | 2.14.1 / 0.29.1，官方 CUDA 12.6 wheel |
| Warp | 1.14.0+cu12，官方 CUDA 12 wheel |

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后：

```bash
./scripts/setup.sh
# 等价于：
uv sync --locked
uv run --locked python tools/check_runtime.py
```

这里显式固定 Warp 的 CUDA 12 wheel，避免依赖更新后选择 CUDA 13。
Warp 初始化日志中的 `CUDA Toolkit 12.9` 是该 wheel 自带的编译库版本，
与本机安装的 Toolkit 分开；PyTorch 固定为 cu126，不改动你已有的 CUDA 12.6。
575 驱动可使用这套 CUDA 12 运行时，无需单独安装系统 CUDA Toolkit。
依据：[Warp 安装与驱动要求](https://nvidia.github.io/warp/v1.14/user_guide/installation.html)、
[PyTorch 官方安装版本](https://pytorch.org/get-started/previous-versions/)。
首次安装需要访问 PyPI、PyTorch wheel 源和 GitHub Releases。

## HOI 数据与控制

当前直接读取已经切分、50 Hz 的 HOI NPZ，无需重采样或重写原文件：

下面列出最初验收的三个示例；`motion_clips=null` 实际加载当前目录的全部 NPZ。关节和刚体名称优先读取文件内字段；缺失时读取 `data/hoi/manifest.json` 中的对应名称顺序，再进行 mimic/FK 校验。

```text
data/hoi/
├── manifest.json
├── motions/test/
│   ├── sub16_largebox_012_v00.npz   477 帧
│   ├── sub16_largebox_022_v00.npz   414 帧
│   └── sub16_largebox_038_v00.npz   385 帧
└── objects/largebox/
    ├── object.xml
    └── largebox.obj
```

运行时仍使用 `data/tracking/robot/g1_inspire` 中的完整机器人；loader 校验其 XML、URDF 和 mesh
与 HOI manifest 的哈希一致。关节与刚体按名称映射，内部关节顺序为身体 29、主动手指 12、从动手指 12。
坐标统一为 z 向上、米、`wxyz` 四元数；参考位置保留连杆原点，HOI 质心线速度依据模型 `body_ipos`
转换成连杆原点速度。三段动作分别建立边界，未来目标与 imagination 不跨段。

```bash
# CPU 数据、mimic、FK、越限检查；输出逐关节、逐帧报告，保留参考原值。
uv run --locked python tools/check_hoi.py --report /tmp/hoi_validation.json
# 全帧联合 reset + 物理一步 + 接触容量检查。
uv run --locked python tools/check_hoi.py --device cuda:0 --physics --num_envs 16
```

手部参考统一来自 motion。随机、固定、外部手部 command 入口已取消，配置不再接受 `hand_target`、
`hand.hold_steps` 等旧字段。Reset 同帧恢复身体、全部手指和物体位姿/速度；首版关闭独立 reset 噪声，
保留原有摩擦、身体关节偏移与 torso COM 的仿真上下文。物体按原 mesh、密度、质心及惯量进行真实物理仿真。

动作维度与原控制语义保持 41 维：身体 `default_q + offset + scale * action_body`；手部
`range_mid + range_half * tanh(action_hand)`。12 个独立手部目标按 URDF mimic 比例扩展后，
写入全部 24 个手指位置执行器；机器人总执行器数为 53。保留原有 12 条物理 mimic 约束，
实际从动关节状态由仿真求解并独立保存，不用主动关节重建。
策略的观测为 186 维，含 motion 当前手指参考及实际物体相对位姿/速度；未来 goal 每帧 112 维。
联合状态为 132 维，WM 输入 208 维、增量输出 130 维；身体、手部、物体 WM 损失分别平均和加权。
Replay 通过参考帧重建手指目标，真实环境和 imagination 的奖励都对齐下一帧。

保留既有身体奖励和正则项，手指角度 tracking 奖励已删除，`hand.track_weight/track_std` 配置也已移除；
手部角度、mimic 和独立 WM 手部误差诊断不再输出；当前与未来手指参考继续进入观测和 goal，
手部 WM 损失继续用于训练。物体位置与旋转两项指数奖励
对齐 `/root/g1_hoi_learning/configs/track/train.yaml` 的训练覆盖值：
`object.pos_weight=2 object.rot_weight=2 object.pos_std=0.2 object.rot_std=0.3`。
位置奖励为 `2 * exp(-||p-p_ref||² / 0.2²)`，旋转奖励为 `2 * exp(-θ² / 0.3²)`，
其中 θ 为最短 SO(3) 旋转角；两项均乘以控制周期 `dt=0.02 s`。默认总奖励上界为 `9 * dt = 0.18`，
真实环境、imagination、回报归一化与自适应 reset 使用一致定义。
容差单位分别为 m 和 rad。接触匹配、碰撞代理穿透、接触切向滑移速度、
期望抓握阶段失去手部接触和物体速度只做诊断。沿用原高度/倾斜终止条件。

使用 `control_version=5 task=g1_inspire_hoi state_version=1` 的新 checkpoint，包含本次奖励契约变更。
旧 LAFAN 或 HOI checkpoint 可通过 `init_from=logs/旧运行/last.pt` 初始化形状匹配的 WM/policy 参数；
新输入/输出层、normalizer、critic、optimizer 和 replay 重新建立。`resume` 只接受兼容 HOI checkpoint，
并校验数据与资产签名。实现及验收记录见 [HOI 方案](docs/g1_inspire_hoi_tracking_plan.md)。
参考中已报告 263 个“帧×主动关节”越限值，碰撞代理最大穿透约 4.57 cm；原始参考保持不变。
首版已通过 137 项回归及短训练链路检查，动作拟合效果仍需正式训练验证。

手部执行器参数对齐 `/root/g1_hoi_learning/source/g1_hoi_learning/g1_hoi_learning/robots/g1_inspire.py`
的 `fingers` 配置，主动与从动关节均使用：

| 参数 | 值 |
|---|---:|
| stiffness | 5.0 N·m/rad |
| damping | 0.5 N·m·s/rad |
| armature | 0.01 kg·m² |
| 单关节 effort limit | 1.0 N·m |

本次对齐范围是上述执行器参数与从动目标写入。参考仓库的 PhysX `velocity_limit_sim`
（拇指 pitch 1.7、yaw 和拇指从动 2.9、其余手指 4.3 rad/s）在 MuJoCo 的原生位置执行器中
没有对应的关节硬限速字段，当前未模拟该硬约束。手部动作仍使用本仓库的可行区间 tanh 映射，
物理 mimic 约束也予以保留，因此两个物理引擎的行为不能视为完全等价。

可微 FK 覆盖全部 53 个实际关节和 54 个刚体。`G1Kinematics.forward(*S[:6])` 默认保持
14 个 tracking 刚体输出；`all_bodies=True` 返回 `body_names` 顺序的全部刚体。
`forward_hands(*S[:6])` 返回 `hand_body_names` 顺序的双掌及 24 个指节的位置、朝向、
连杆原点线速度与世界角速度。掌面固定链接已合并到 wrist-yaw 刚体，掌部输出采用该连杆坐标系。
手部 FK 必须输入完整 53 维关节位置和速度；29 维输入仅用于身体输出。

`hand_surface(*S[:6])` 返回手掌及全部指节碰撞 mesh 支撑顶点的世界位置、速度，
可直接用于真实状态、`unflatten_state(replay_state)` 和 WM imagination；采样点包含编译后的 mesh
变换与 geom 偏移，速度包含 `ω × r`。这些是有限碰撞表面采样点，不能单独判定真实接触或承载。
HOI loader 同时报告原 tracking 刚体的 `fk_max_error` 与全身含手指的 `full_fk_max_error`，
并拒绝位置、朝向或速度与 FK 不一致的手部参考。

LAFAN 离线下载与 CSV 预处理工具仍保留；其产物不包含 HOI 手指/物体参考，不能用作当前任务的训练输入：

```bash
./scripts/download_data.sh
uv run --locked python tools/preprocess_motion.py \
  --source_dir data/tracking/motions/lafan1_raw --output_dir data/tracking/motions/lafan_g1_inspire --device cpu
```

## 检查环境

```bash
uv run --locked python tests/test_spawn.py --num_envs 64
uv run --locked python tests/test_env.py --num_envs 16
uv run --locked python tests/test_reset.py
uv run --locked pytest
```

检查 GPU 步进、正向运动学与机器人各连杆姿态/速度的一致性、状态写回、局部重置以及域随机化。
`test_spawn.py` 不需要动作 CSV，但需要完整 Inspire 资产。

原固定手型 Inspire 接入已在 A800 上验证模型拓扑、CPU/GPU 物理 mimic、非零手型保持、局部 reset、
身体 FK 位姿/速度与梯度、checkpoint 手型恢复、Viser 虚影、轨迹恢复和 EGL 渲染。
此前的 8 环境短训练、续训和 40 段动作评估录制用于验证固定手型流程；主动手部策略的正式性能需重新训练检验。
主动手部版本已通过 100 项回归检查，并用小网络跑通 8 环境训练、checkpoint 续训、40 段动作评估录制及
`torch.compile` 训练更新。该检查只验证链路与数据一致性，不代表策略已收敛。
具体结果和初始手部 PD 参数见 [接入方案及验证记录](docs/g1_inspire_tracking_plan.md)。

完整手部模型的倒地姿态可能产生超过 740 行接触和 mimic 约束，默认 `njmax=1536`。
训练、播放、评估及测试工具使用相同默认值；使用旧的保存配置时，显式传入 `njmax=1536`。
环境会在奖励及回报统计之前检查物理状态和容量溢出，并在采样前校验统计与概率；
异常会报告环境编号、参考帧和动作名称，便于定位最早的问题。

## 训练

训练和评估使用 **Hydra**，以 `key=value` 覆盖参数。默认配置使用 HOI 数据和原有训练超参数；
[完整参数模板](supermimic/conf/train.yaml) 包含所有训练参数、网络结构、优化器与注释。
[训练配置说明](docs/training.md) 提供参数含义、续训和论文逐项对应表。

```bash
# 当前 codebase 默认配置
./scripts/run_train.sh hoi_main
# 指定 GPU、CPU 亲和性以及训练参数：
CPUS=0-23 ./scripts/run_train.sh hoi_main device=cuda:1 num_envs=4096
# 也可以直接调用：
uv run --locked python -m supermimic.train run_name=hoi_main device=cuda:0
# 论文单 GPU 超参数（Table A.4/A.5，TF32，从第一轮起联合更新）
./scripts/run_train.sh paper --config-name paper device=cuda:0
# 只打印最终配置，不初始化 GPU 或启动训练
uv run --locked python -m supermimic.train --cfg job --resolve
uv run --locked python -m supermimic.train --config-name paper --cfg job --resolve
```

启动脚本在前台运行，日志同时写入 `logs/<run>/stdout.log`，可放在自己的 tmux 或作业调度器中。
默认使用 **SwanLab**，项目名为 `supermimic`，实验名取 `run_name`；每 `log_every` 次迭代同步训练指标，
同时保留本地 `train_log.csv`。在项目根目录的 `.env` 中设置 `SWANLAB_API_KEY=你的密钥`
（参考 [`.env.example`](.env.example)），也可通过环境变量设置；环境变量优先，密钥不进入训练配置或检查点。
`.env` 已被 Git 忽略。使用 `logger.mode=offline` 仅保存本地 SwanLab 日志，
`logger.backend=none` 关闭 SwanLab，`logger.project=项目名` 可指定项目。
SwanLab 指标按原生 `组名/指标名` 分组，本地 `train_log.csv` 保留原字段名，并追加诊断列：

| 分组 | 指标 |
|---|---|
| `performance` | `success_rate`、`progress`、`joint_mae`、`body_position_error`、`body_rotation_error`、`episode_return`、`episode_length` |
| `reward` | `total`、`body_position`、`body_orientation`、`body_linear_velocity`、`body_angular_velocity`、`root_position`、`root_orientation`、`object_position`、`object_orientation`、`action_rate`、`joint_limit`、`hand_action_rate`、`hand_joint_limit` |
| `failure` | `failure_fraction`、`height_fraction`、`tilt_fraction` |
| `world_model` | `wm_loss`、`wm_err8`、`wm_body_err8`、`wm_object_err8`、`wm_object_contact_err8` |
| `policy` | `loss`、`grad_norm`、`imagined_reward`、`imagined_survival` |
| `train` | `critic_loss`、`wall_hours`、`sim_steps`、`iterations_per_second`、`learning_rate` |

例如归一化训练回报显示为 `performance/episode_return`，归一化单步奖励显示为 `reward/total`。
`performance/success_rate` 是日志窗口内正常结束（未触发高度或倾斜失败）的 episode 占比，取值 0–1；
正常片段结束和达到训练步数上限都计为成功。`performance/progress` 是已结束 episode 完成比例的均值，
分母取该 episode 重置时的剩余片段步数和步数上限中较短者，分子取实际执行步数，结果限制在 0–1。
随机起始帧不计入已完成进度；失败可以有较高 progress，但仍不计为成功。评估从片段开头运行至结束，
训练指标反映随机起点、有限时长下的 episode 表现，不能直接当作完整片段评估成功率。
`joint_mae` 为 29 个身体关节的角度绝对误差均值（rad）；`body_position_error` 为跟踪节点的三维距离均值（m），
`body_rotation_error` 为跟踪节点的姿态角误差均值（rad）。三项与评估共享计算定义，按日志窗口内全部实际
transition 的样本数加权平均，包含结束前的最后一步，不包含重置后的状态；没有 episode 结束时也正常记录。
身体位置、姿态误差使用世界坐标原始参考，不做 reward 中的 XY/yaw 对齐，因此也能反映整体漂移；
训练记录包含探索噪声和环境随机化，数值可以与评估对照，但不代表相同的运行条件。
reward 分项包含实际权重、正负号和控制步长，统一除以单步奖励理论上限；日志窗口内取样本均值，
13 项贡献之和等于 `reward/total`。可以直接看出哪些追踪项不足、哪些惩罚在拉低奖励。
`failure` 只记录总失败占比和对应两个失败条件的占比，统一以窗口内结束的 episode 数为分母。
高度、倾斜失败可以同时发生，两个原因占比不要求相加为失败占比。正常片段结束、步数截断统一计入
`performance/success_rate`，它与 `failure/failure_fraction` 相加为 1。
迭代数用作图表横轴，不再单独生成曲线。只上报当前日志窗口有数据的指标：预训练阶段不创建 policy 曲线，
关闭 critic 时不记录 critic loss，没有 episode 结束时不记录 episode 指标。真实的零值正常保留；
缺失或非有限值不上报，CSV 对应单元格留空，终端显示 `--`。没有 episode 结束时不记录成功率、progress 和失败占比。
续训时自动扩展旧 CSV 表头，历史新增指标留空，不补零；已停用的诊断列只在旧 CSV 中保留历史值，后续留空，
新 CSV 只包含当前指标。
分组上报从下次启动训练时生效；已有 Scalar 图表需在 SwanLab 网页整理，历史指标不会自动重命名或删除。

`device=cuda:1` 选择当前进程可见的 GPU，亦可使用 `CUDA_VISIBLE_DEVICES=1`，此时进程内用 `device=cuda:0`。
默认 4096 个环境、每环境 1000 步回放缓存，占用较多显存；可先缩小参数验证：

```bash
./scripts/run_train.sh smoke --config-name smoke
```

正式训练默认启用 `torch.compile` 和 bf16 MLP；首次运行会编译内核。
`no_compile=true` 用于快速诊断，`amp=none` 关闭 bf16；`paper` 配置使用 `amp=none tf32=true`。
`paper` 只沿用论文公开的单卡超参数，当前输入为 HOI；论文主实验使用过滤后的 BONES-SEED。
数据、仿真资产及未公开的实现设置与实验复现的关系见[配置说明](docs/training.md)。

每 2000 次迭代写入 `ckpt_XXXXXX.pt`；`last.pt` 还包含优化器状态，可恢复训练：

```bash
./scripts/run_train.sh hoi_resume resume=logs/hoi_main/last.pt
# 原运行使用自定义配置时，加载保存的完整配置再恢复（例如 hist/goal_frames 有改动）：
uv run --locked python -m supermimic.train \
  --config-path /root/supermimic_vibed/logs/hoi_main --config-name config \
  run_name=hoi_resume log_dir=/root/supermimic_vibed/logs/hoi_resume \
  resume=/root/supermimic_vibed/logs/hoi_main/last.pt iterations=400000
```

先拟合单段时，可直接选择原目录中的文件，无需复制或再次切分：

```bash
./scripts/run_train.sh hoi_single 'motion_clips=[sub16_largebox_012_v00]'
# 三段一起训练时使用 motion_clips=null（默认）。
# 旧 LAFAN checkpoint 的匹配层初始化：
./scripts/run_train.sh hoi_warm init_from=logs/旧运行/last.pt
```

恢复时重新采集回放缓存，原检查点不包含回放数据。`iterations` 是训练结束时的总迭代数。
每个运行保存解析后的 `config.yaml`、兼容旧格式的 `args.json` 和 Hydra 的 `.hydra/` 配置记录。
配置字段与类型受校验；续训时会检查网络结构、`hist` 和 `goal_frames` 是否匹配。

## 评估与渲染

服务器上可以直接运行 checkpoint 策略，通过 **Viser** 在本地浏览器实时查看三维场景：

```bash
# 服务器：默认读取 logs/hoi_main/last.pt 和检查点保存的 motion_dir。
uv run --locked python -m supermimic.play run_dir=logs/hoi_main device=cuda:0
# 指定检查点、动作或端口：
uv run --locked python -m supermimic.play run_dir=logs/hoi_main \
  ckpt=logs/hoi_main/ckpt_002000.pt clip=sub16_largebox_012_v00 port=8080

# 本地电脑：保持 SSH 连接，浏览器打开 http://localhost:8080。
ssh -N -L 8080:127.0.0.1:8080 用户名@服务器地址
```

播放界面可切换动作、暂停、单步、重播、调速及调整相机；每次完成或失败后自动重播所选动作。
`Frame` 进度条可跳到指定帧（从 0 开始，50 帧 = 1 秒）；`Previous frame` 后退一帧，
`Single step` 前进一帧，`Back 5 s` / `Forward 5 s` 前后跳 5 秒。跳帧后自动暂停，取消 `Paused` 继续播放。
已计算的帧直接从缓存回看；首次跳到后面的帧会尽快推进策略和物理仿真，不按实时速度等待，
期间显示 `Seeking frame ...`。所有帧保留从片段起点运行的真实策略结果；若策略提前失败，只能跳到失败帧。
切换动作或点击 `Restart motion` 会清空缓存并重新仿真。
默认叠加同帧的**蓝色半透明机器人与物体 reference 虚影**，两者保留共同世界坐标。
在 Reference motion 面板可隐藏虚影、调整透明度；`align_reference=true` 只对齐机器人虚影，
物体目标仍保留全局参考，便于发现漂移。默认参数为 `show_reference=true align_reference=false reference_opacity=0.3`。
默认以 50 FPS 刷新三维场景，与 50 Hz 控制频率一致，可通过 `fps=...` 覆盖。
策略推理和物理仿真在服务器 GPU 上执行，浏览器负责三维显示，不需要 X11、EGL 或提前录制 NPZ。
仅在浏览器连接后推进仿真。默认绑定 `127.0.0.1`，可用 SSH 或 VS Code 端口转发；
`motion_dir=/path/to/npz` 可覆盖检查点中的动作目录。默认加载启动时的检查点，更新权重需重启播放进程。

离线评估、轨迹录制与视频导出仍可使用：

```bash
uv run --locked python -m supermimic.evaluate run_dir=logs/hoi_main watch=true
uv run --locked python -m supermimic.evaluate run_dir=logs/hoi_main \
  ckpt=logs/hoi_main/ckpt_014000.pt record=traj.npz
uv run --locked python tools/plot.py logs/hoi_main --out results/success_curve.png
uv run --locked python tools/render.py traj.npz out.mp4 --clips sub16_largebox_012_v00 --seconds 10
```

评估从每段动作首帧开始，关闭观测/动作噪声与域随机化，摩擦系数为 1.0。
评估额外输出 `mae`：29 个关节角相对同帧 reference 的平均绝对误差，单位 rad，按有效物理步加权，包含终止步。
该指标写入批量评估的 `eval.csv` 和 `*_eval.json`；已有 CSV 的历史记录保留，历史 MAE 留空。
评估同时输出物体位姿/速度及接触诊断；`success_rate` 仍表示未触发原有身体终止条件的完整动作比例，
不能单独代表物体跟踪成功。录制保存实际与参考手指/物体状态、41 维 `action` 和 `hand_reference`。
评估用 `device=cuda:1`；环境验证与数据处理工具保留 argparse 的 `--device cuda:1` 语法。
渲染默认使用 EGL，ffmpeg 由 `imageio-ffmpeg` 提供；服务器需要可用的系统 EGL 驱动库。

## 代码结构

| 路径 | 功能 |
|---|---|
| `supermimic/robot.py` | 原始 G1 资产、关节顺序、BeyondMimic 隐式 PD 参数与 mjlab 实体配置 |
| `supermimic/env.py` | mjlab 场景、物理步进、域随机化、基于回报的动作分桶重置采样 |
| `supermimic/motion.py`、`hoi.py`、`kinematics.py` | HOI 名称/坐标/速度适配、离线 LAFAN 工具、全身与手指可微 FK、碰撞面采样与未来目标 |
| `supermimic/mdp.py` | 观测、奖励、终止与仿真上下文 |
| `supermimic/networks.py`、`train.py` | 世界模型、策略/价值网络、BPTT 与 GAE 训练 |
| `supermimic/evaluate.py` | 检查点评估、持续监听及轨迹录制 |
| `scripts/` | 本机 uv 安装、数据准备与训练启动脚本 |
| `tools/` | 安装、数据下载、MJCF 转换、验证、绘图、渲染和性能分析 |
| `tests/` | 环境验证脚本、mjlab 集成与 NPZ 回归检查 |

仿真上下文为摩擦系数、29 个关节偏移及躯干质心偏移。
摩擦和质心偏移在环境初始化时逐环境采样，关节偏移每个 episode 重采样。
机器人禁用自碰撞，保留机器人与地面之间的接触。

## 原始复现结果与来源

本项目最初使用 Isaac Lab 3 / Newton，原作者记录了单 RTX PRO 6000 上约 1.47 小时、
归一化回报 0.829、40/40 动作无失败的结果。`results/` 中的图和动图来自原始实现，
`MILESTONES.md` 保留其历史记录；这些数值尚未在 mjlab 后端重新复现。
更换物理接口及求解器版本后，需要重新训练评估才能比较性能。

![原始实现的 LAFAN1 跟踪效果](results/supermimic_lafan1_highlight.gif)

方法：Q. Liao 等，*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable
World Model*，CoRL 2026，arXiv:2610.28480。本仓库是独立、非官方实现。
原始代码由 ETH Zurich 的 Chong Zhang 发布。

## 许可证

代码：[MIT](LICENSE)。mjlab、MuJoCo 与 MuJoCo-Warp：Apache-2.0；Warp：Apache-2.0。
LAFAN1 动作（Ubisoft La Forge）：CC BY-NC-ND 4.0；Unitree G1 描述和重定向数据的具体许可
见上游数据集。动作数据不随代码仓库分发，演示素材不受本仓库 MIT 许可覆盖。
