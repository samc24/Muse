# FollowThrough -- Legacy

Experimental reference code. Retained because some files contain reference implementations (user-model shape, Euclidean shot comparison, template averaging) that inform the planned build-out of task 2 in the canonical `source/` tree.

Not imported by any canonical code path. Do not modify casually.

## Contents

| File | Role |
|---|---|
| `OpenPoseVideo.cpp`, `CMakeLists.txt`, `getModels.sh` | Reference C++ pose-inference runner and its build / model-fetch scripts. |
| `AveragePoseAnalysis.py`, `AverageImagePose.py` | Template-averaging references at video and image-batch granularity. |
| `PoseAnalysisPhoto.py`, `ImagePose.py` | Single-photo pose analysis and its per-image data model. |
| `UserModel.py` | Reference shape for a per-player shot-vector model -- directly relevant to task 2. |
| `model_test.py` | Harness that builds per-player models from a folder of sample videos (imports `UserModel`). |
| `swap_body_parts.py`, `testtest.py` | Scratch experiments. |
