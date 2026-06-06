# Math Map

Concept connections for robotics engineers and AI agents.

Machine-readable chapter index: [concept_index.yaml](concept_index.yaml)

```
SE(2) / SE(3)
├── used in: pose estimation, SLAM, manipulation, humanoid state
├── libraries: Sophus, GTSAM, Ceres
├── failure: treating rotation as Euclidean vector
└── chapters:
    ├── Ch.01 Pose is not a vector ✅
    ├── Ch.02 Pose graph SLAM ✅
    ├── Ch.03 Retraction vs projection ✅
    └── Ch.04 Riemannian motion policies ✅

Optimal Transport
├── used in: point cloud registration, map comparison, trajectory alignment
├── libraries: POT, GeomLoss, OTT-JAX
├── failure: L2 on unordered point sets
└── chapters:
    ├── Ch.05 Sinkhorn ✅
    └── Ch.06 Wasserstein maps ✅

Diffusion / Flow Matching
├── used in: multimodal trajectories, manipulation policies
├── failure: unimodal regression → mode averaging
└── chapters:
    ├── Ch.07 Diffusion policy ✅
    └── Ch.08 Flow matching ✅

Information Geometry
├── used in: natural policy gradient, TRPO, Fisher preconditioning
├── failure: one lr on ill-conditioned policy parameters
└── chapter: Ch.11 Natural gradient ✅

Geometric Deep Learning
├── used in: 3D manipulation, point cloud policies
├── failure: flatten coordinates under rotation
    └── chapter: Ch.12 SE(3)-equivariant preview ✅

Neural Operators
├── used in: surrogate simulators, PDE / rollout amortization
├── failure: grid-flattened MLP without operator structure
└── chapter: Ch.13 DeepONet preview ✅

World Models & Differentiable Physics
├── used in: latent dynamics, imagination rollout, system ID
├── failure: open-loop drift, hard contact breaks gradients
└── chapters:
    ├── Ch.09 Differentiable physics ✅
    └── Ch.10 Tiny world model ✅
```

Each chapter's `concept.yaml` extends this map with machine-readable links.
