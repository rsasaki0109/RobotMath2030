# RobotMath2030

**RobotMath2030は、2030年のロボティクス・Physical AI・World Model・Humanoidを理解するための数学を、最小Pythonコードと可視化で学ぶOSSです。**

教科書ではありません。論文の再実装集でもありません。
各章は **数式 → 最小実装 → ロボット応用 → 失敗例** の形式で、論文を読む前の「数学の交通整理」を目指します。

> [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) の次、GTSAM / Ceres / LeRobot の前に読むもの。

[English README](README.md)

## デモ一覧

| Lie pose graph | Sinkhorn 点群 | 多峰性行動 |
|:--:|:--:|:--:|
| ![Pose graph](assets/animations/pose_graph_loop_closure.gif) | ![Sinkhorn](assets/animations/sinkhorn_point_clouds.gif) | ![Diffusion](assets/animations/diffusion_policy_2d.gif) |

GIF 再生成: `pip install -e ".[torch]" && python scripts/render_all_gifs.py`

## なぜ RobotMath2030 か

ロボティクス論文で次のような語に遭遇しても、通常の数学教材は「定義」は教えてくれますが、**なぜ地図・姿勢・行動・接触・世界モデルに現れるのか**までは教えてくれません。

RobotMath2030はそのギャップを埋めます。

## 最初のデモ

| 章 | 内容 |
|----|------|
| [01 — Pose is not a vector](chapters/01_pose_is_not_vector/) | SE(2) の合成・exp/log、Euler角の罠 |
| [02 — Tiny Lie graph optimizer](chapters/02_tiny_lie_graph_optimizer/) | pose graph SLAM；Euclidean vs Lie 残差 |
| [05 — Sinkhorn for point clouds](chapters/05_sinkhorn_point_clouds/) | 地図・点群のソフト対応；OT vs 素朴なマッチング |
| [06 — Wasserstein map evaluation](chapters/06_wasserstein_map_evaluation/) | 地図 drift / ghost obstacle；L2 vs W2 |
| [07 — Diffusion policy 2D](chapters/07_diffusion_policy_2d/) | 多峰性軌道；平均回帰 vs diffusion |
| [09 — Differentiable physics](chapters/09_differentiable_physics/) | mass-spring 同定；接触で gradient が壊れる |

```bash
pip install -e ".[torch]"
python chapters/09_differentiable_physics/demo.py
```

## 3ヶ月ロードマップ

```
Month 1  Geometry of State         Lie群 · pose graph · 多様体最適化
Month 2  Geometry of Distribution  Sinkhorn OT · diffusion · 自然勾配
Month 3  Geometry of Dynamics      微分可能物理 · tiny world model
```

詳細は [docs/roadmap.md](docs/roadmap.md)。

## インストール

```bash
pip install -e ".[dev]"
pytest
```

コア依存: **NumPy**, **Matplotlib**。Ch.07 以降は **PyTorch**（`pip install -e ".[torch]"`）。

## ライセンス

MIT — [LICENSE](LICENSE)
