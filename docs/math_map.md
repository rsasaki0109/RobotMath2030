# Math Map (preview)

Concept connections for robotics engineers and AI agents.

```
SE(2) / SE(3)
├── used in: pose estimation, SLAM, manipulation, humanoid state
├── libraries: Sophus, GTSAM, Ceres
├── failure: treating rotation as Euclidean vector
└── next:
    ├── manifold optimization (Ch. 03)
    ├── pose graph SLAM (Ch. 02) ✅
    ├── uncertainty on Lie groups
    └── equivariant networks (Ch. 06+)

Optimal Transport
├── used in: point cloud registration, map comparison, trajectory alignment
├── libraries: POT, GeomLoss, OTT-JAX
├── failure: L2 on unordered point sets
└── next: Sinkhorn (Ch. 05), Wasserstein maps (Ch. 06)

Diffusion / Flow Matching
├── used in: multimodal trajectories, manipulation policies
├── failure: unimodal regression → mode averaging
└── next: Ch. 07–08

World Models
├── used in: latent dynamics, imagination rollout, MPC
├── failure: open-loop rollout drift
└── next: Ch. 10
```

Each chapter's `concept.yaml` extends this map with machine-readable links.
