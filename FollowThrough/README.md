# FollowThrough — Shooting-Form Pose Analysis

FollowThrough reduces a basketball jumpshot into a timeseries of joint positions, aggregates those into a per-player template, and supports comparing a user's shot against reference templates to surface mechanical differences. The goal is an analytic, objective form-comparison tool to complement qualitative coaching feedback.

> "Follow through" in basketball refers to relaxing the wrist downward and holding your posture constant after releasing the ball. It's taught as a key mechanical rule for shot accuracy — and it's the hardest part of a shot to self-diagnose without an outside eye. This subsystem is that outside eye.

## Motivation

Form critique in basketball is visual. Coaches say things like "your elbow's flaring", "your release is late", "your base is too narrow". The critiques are qualitative and depend on the coach's trained eye. Millions of YouTube views on shot-form breakdowns attest to demand — but there's no free platform where a player can compare their own shot against a library of pros and get objective, quantified feedback.

If a shot is reduced to a vector of joint positions over time, two shots become directly comparable: a user's template versus a library of reference-player templates. Distance in joint-vector space surfaces the closest reference; component-wise differences tell the user which joints deviate.

## Product vision

The pose-analysis engine supports a layered product surface:

1. **Compare.** Side-by-side comparison of a user's shot against a chosen NBA player's shot, with quantified joint-level diffs.
2. **Combine.** Build a hybrid target — e.g. one player's base with another player's release — and compare against that.
3. **Extend.** Move beyond the jumpshot to layups, handles, defensive stances, and other basketball-specific motions.
4. **Create.** A user defines their own target form and iterates toward it.

Throughout, the system surfaces contextual coaching tips ("looking at the backboard improves accuracy; looking at the ball improves arc"; "decrease knee bend by 7° for more leg power") keyed on the shape of the measured deviation.

Muse's current implementation covers the foundations of step 1 — pose extraction, smoothing, and template generation. Everything after that is scoped.

## Architecture

FollowThrough decomposes into two tasks. Every feature belongs to one of them.

1. **Skeleton smoothing.** Per-frame MediaPipe Pose estimates jitter. Task 1 smooths the joint timeseries (Savitzky–Golay) to produce a visually clean, temporally coherent skeleton.

2. **Shot-vector modelling.** Given a smoothed skeleton for a shooter: segment the jumpshot, average joint positions across reference frames into a template, build a user-model object keyed on that template, and compare new shots against a library of reference templates via distance in joint-vector space.

Task 1 is implemented. Task 2 is the main outstanding work (see **Status**).

## Layout

```
FollowThrough/
├── source/                # Canonical implementation (OOP, MediaPipe Pose)
│   ├── SkeletonMaker.py
│   └── test_skeleton_maker.py
├── PoseAnalysis.py        # Script-tier form analysis
├── save_pose_data.py      # Persist pose keypoints to CSV
├── smooth_pose_data.py    # Savitzky–Golay smoothing over CSV
├── legacy/                # Experimental reference code; see legacy/README.md
└── docs/                  # Reserved for subsystem design docs
```

## Canonical code

`source/SkeletonMaker.py` is an OOP wrapper around MediaPipe Pose with parameterised initialisation and docstringed methods:

- `process_video(...)` — runs MediaPipe Pose per frame, smooths the keypoint timeseries, writes annotated video and a keypoint CSV.
- `generate_pose_overlay_image(...)` — draws pose landmarks onto a single image.
- `generate_template_from_images(...)` — averages poses across a folder of reference images into a template.

`source/test_skeleton_maker.py` is an ad-hoc test module; a formal `pytest` setup is pending.

## Pipelines

**Video analysis (task 1):**

```
video → SkeletonMaker.process_video()
     → MediaPipe Pose inference per frame
     → Savitzky–Golay smoothing over keypoint timeseries
     → annotated video + keypoint CSV
```

**Template generation (task 2 — partial):**

```
reference images → SkeletonMaker.generate_template_from_images()
                → per-image keypoints
                → averaged template pose
```

## Run

```bash
source FollowThrough/venv/bin/activate

python3 FollowThrough/source/SkeletonMaker.py
python3 FollowThrough/source/test_skeleton_maker.py
```

## Status

Implemented:

- [x] Per-frame pose estimation on MediaPipe Pose
- [x] Savitzky–Golay smoothing over the keypoint timeseries
- [x] Overlay rendering (image and video)
- [x] Template generation from a folder of reference images

Not yet implemented:

- [ ] Shot segmentation from a full video
- [ ] User-model type (template + metadata + sample-accumulation)
- [ ] Reference-template library of known shooters
- [ ] Distance-based comparison (user shot → closest reference, component-wise diff)
- [ ] Contextual tip surfacing based on deviation patterns
- [ ] Product-vision phases beyond step 1 (combine / extend / create)
- [ ] Formal test framework

Script-tier files (`PoseAnalysis.py`, `save_pose_data.py`, `smooth_pose_data.py`) duplicate subsets of `SkeletonMaker`'s behaviour; consolidating them into the canonical OOP surface is pending.

## Roadmap

- Build out task 2 (user model, template library, comparison) on top of `SkeletonMaker`
- Add shot segmentation — automatically clip a jumpshot out of a full video
- Consolidate the script-tier files into `source/`
- Move hard-coded paths and windows into a config surface
- Formalise tests under `pytest`

## Legacy

`legacy/` holds earlier experimental code kept as algorithmic reference — see [`legacy/README.md`](legacy/README.md). Not imported by any canonical code path.
