# Chapter 07 — Diffusion Policy for 2D Trajectories

## Robotics context

Robot actions are often **multimodal**: many valid ways to reach the same goal (left vs right around an obstacle).
Supervised learning with MSE learns the **average** — which may be invalid (collision).

Diffusion policies model a **distribution** over trajectories via iterative denoising.

## What this demo shows

1. Two-path 2D world with a central obstacle
2. **Mean regression** baseline → straight-line average → collision
3. **Tiny diffusion policy** (~150 lines) → samples left OR right paths

## Failure cases

| Wrong idea | What breaks |
|------------|-------------|
| MSE on all demos | Mode-averaged path hits obstacle |
| Single Gaussian policy | Same problem in continuous action spaces |
| Too few diffusion steps | Noisy / invalid trajectories |

## Run

Requires PyTorch:

```bash
pip install -e ".[torch]"
python chapters/07_diffusion_policy_2d/demo.py
```

## Related papers

- Diffusion Policy — conditional denoising for robot behavior

## Next

→ Chapter 08 — [Flow matching vs diffusion](../08_flow_matching_vs_diffusion/)
