# Contributing

Thanks for your interest in RobotMath2030.

## Scope

We welcome contributions that fit the project goal:

**math → minimal code → robotics context → failure cases**

Please avoid:

- Production-scale reimplementations of GTSAM, POT, or robot learning stacks
- Large dependency additions without a strong chapter need
- Paper reimplementations without teaching value

## How to contribute

1. Open an issue for larger changes
2. Keep PRs focused on one chapter or one tiny engine
3. Add or update tests for math properties
4. Include `concept.yaml` for new chapters
5. Run `pytest` and `python scripts/render_all_gifs.py` when visuals change

## Chapter checklist

- [ ] `README.md` with robotics context and failure cases
- [ ] `demo.py` runs on CPU in < 30 seconds
- [ ] `concept.yaml` metadata
- [ ] Tests for core math behavior
