# Chapter 10 — Tiny Latent World Model + Planning

## Robotics context

**World models** compress observations into a latent state, predict future latents, and let the robot
**imagine** action sequences before acting — core to Dreamer-style RL and many Physical AI stacks.

## What this demo shows

1. Random data in a **2D grid world**
2. Train a tiny **encoder → latent dynamics → decoder**
3. **Random-shooting MPC** in imagination
4. **Closed-loop replanning** vs **open-loop** rollout — open loop fails more often

## Failure cases

| Mistake | Symptom |
|---------|---------|
| Open-loop imagination | Error compounds; agent hits walls |
| Too-long horizon with tiny model | Predictions blur; plan is overconfident |
| No replanning on real robot | Sim plan diverges from reality |

## Run

```bash
pip install -e ".[torch]"
python chapters/10_tiny_world_model/demo.py
```

## Related

- Chapter 09 — Differentiable physics (learning dynamics parameters)
- DreamerV3 — general world-model RL
