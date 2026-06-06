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

## 全10章（MVP 完成）

完全ガイド: [docs/chapters.md](docs/chapters.md) · 索引: [docs/concept_index.yaml](docs/concept_index.yaml)

| 章 | 内容 |
|----|------|
| [01 — Pose is not a vector](chapters/01_pose_is_not_vector/) | SE(2) の合成・exp/log、Euler角の罠 |
| [02 — Tiny Lie graph optimizer](chapters/02_tiny_lie_graph_optimizer/) | pose graph SLAM；Euclidean vs Lie 残差 |
| [03 — Retraction vs projection](chapters/03_retraction_vs_projection/) | ランドマーク pose；retraction vs 投影 |
| [04 — Riemannian motion policy](chapters/04_riemannian_motion_policy/) | タスク metric 融合；素朴な APF 和 vs RMP |
| [05 — Sinkhorn for point clouds](chapters/05_sinkhorn_point_clouds/) | 地図・点群のソフト対応；OT vs 素朴なマッチング |
| [06 — Wasserstein map evaluation](chapters/06_wasserstein_map_evaluation/) | 地図 drift / ghost obstacle；L2 vs W2 |
| [07 — Diffusion policy 2D](chapters/07_diffusion_policy_2d/) | 多峰性軌道；平均回帰 vs diffusion |
| [08 — Flow matching vs diffusion](chapters/08_flow_matching_vs_diffusion/) | 同タスク；velocity field vs denoising |
| [09 — Differentiable physics](chapters/09_differentiable_physics/) | mass-spring 同定；接触で gradient が壊れる |
| [10 — Tiny world model](chapters/10_tiny_world_model/) | latent dynamics + imagination MPC |
| [11 — Information geometry](chapters/11_information_geometry/) | Fisher metric / natural gradient |
| [12 — SE(3)-equivariant preview](chapters/12_se3_equivariant_preview/) | 等変性 vs 素朴 MLP |

```bash
pip install -e ".[torch]"
python scripts/smoke_all_chapters.py
```

## Colab ノートブック

| ノートブック | 内容 |
|-------------|------|
| [01_geometry_of_state](notebooks/01_geometry_of_state.ipynb) | SE(2) + pose graph |
| [05_sinkhorn_optimal_transport](notebooks/05_sinkhorn_optimal_transport.ipynb) | Sinkhorn OT |
| [07_diffusion_policy](notebooks/07_diffusion_policy.ipynb) | Diffusion policy |

## 3ヶ月ロードマップ

```
Month 1  Geometry of State         Lie群 · pose graph · 多様体最適化
Month 2  Geometry of Distribution  Sinkhorn OT · diffusion · 自然勾配
Month 3  Geometry of Dynamics      微分可能物理 · tiny world model
```

詳細は [docs/roadmap.md](docs/roadmap.md)。**Docs:** [rsasaki0109.github.io/RobotMath2030](https://rsasaki0109.github.io/RobotMath2030/)

## インストール

```bash
pip install -e ".[dev]"
pytest
```

コア依存: **NumPy**, **Matplotlib**。Ch.07 以降は **PyTorch**（`pip install -e ".[torch]"`）。

## ライセンス

MIT — [LICENSE](LICENSE)
