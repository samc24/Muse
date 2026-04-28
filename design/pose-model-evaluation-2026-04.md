# Pose model evaluation -- 2026-04

**Status:** Decided 2026-04-27. Production model = **MediaPipe Pose Landmarker, Heavy variant**.
**Owner:** Sameer (solo).
**Next scheduled re-evaluation:** when a trigger condition fires (see end of doc).

## Use case (the constraints any candidate must satisfy)

- Single-person, full-body, monocular video. ~1-3 sec clips at 30fps.
- **Foot + heel + ankle keypoints required.** Foot positioning matters for shot mechanics; topologies without these are disqualifying.
- iPhone 17 deployment, iOS 17.4+.
- 30fps sustained on-device for live feedback during recording (≥10fps acceptable for offline-only).
- Open-source weights or commercial license. CoreML or TFLite deployable today.

## Candidate landscape

| Model | Keypoints | Foot/heel? | Benchmark | License | iOS deployable? | Status |
|---|---|---|---|---|---|---|
| **MediaPipe Pose Landmarker (BlazePose GHUM)**, Heavy variant | 33 | Yes (27-32) | n/a published | Apache 2.0 | Yes (TFLite, mature) | **Chosen v1 production** |
| RTMW-l (OpenMMLab, 2024) | 133 | Yes | 70.2 mAP COCO-Wholebody | Apache 2.0 | No published CoreML | Strong runner-up; spike deferred until trigger fires |
| Sapiens-1B (Meta, 2024) | 308 | Yes | SOTA | CC-BY-NC-4.0 | No published CoreML | Disqualified: non-commercial license |
| Apple Vision (`VNDetectHumanBodyPose3DRequest`) | 17 | **No** | proprietary | Apple SDK | Native | Disqualified: missing foot/heel |
| YOLO11-Pose (Ultralytics) | 17 (COCO-17) | **No** | n/a foot-specific | AGPL/Ultralytics | CoreML export available | Disqualified: missing foot/heel + license |
| ViTPose / ViTPose++ | 17 (COCO-17) | **No** | 74.1 mAP COCO | Apache 2.0 | Community ports | Disqualified: missing foot/heel |
| HRNet / HRFormer | 17 by default | **No** | benchmark leader | varies | slow on mobile | Disqualified: missing foot/heel + perf |

The 18-joint canonical schema we standardised on includes head, neck, both shoulders/elbows/wrists/hips/knees/ankles, **plus heels and foot_indices**. Anything that ships only the COCO-17 body skeleton is structurally wrong for our use case, regardless of how well it does on benchmarks.

## Decision: MediaPipe Pose Landmarker, Heavy variant

MediaPipe is the only candidate that simultaneously satisfies every hard constraint on day one:
- 33 keypoints with foot/heel/foot_index.
- Apache 2.0.
- Three years of production deployment in Google's own products and the open ecosystem (HomeCourt, NEX Team).
- Proven 30+fps on-device on Apple Silicon via TFLite + Neural Engine + GPU paths.
- 3D landmarks emitted (z-depth alongside x/y) for future release-angle reasoning.

The Heavy variant trades ~30MB bundle size and ~5-10x inference cost over Lite for materially better keypoint accuracy. iPhone 17 has the inference headroom. Bundle size is well under App Store on-demand-resource thresholds. Foot/heel/ankle precision is the entire game for shot-form analysis.

The keypoint mapping from MediaPipe's 33 BlazePose landmarks to our canonical 18-joint set lives in `MediaPipeKit/Sources/MediaPipeKit/MediaPipePoseProvider.swift` -- 16 direct mappings, plus HEAD derived from NOSE and NECK from the midpoint of L_SHOULDER + R_SHOULDER. A future provider with native head/neck keypoints won't need these derivations.

## Why this isn't an A/B test

The PoseKit / MediaPipeKit abstraction in the architecture plan exists exactly so the choice is reversible cheaply when there's a real reason to swap. We don't owe the architecture a preemptive benchmark sprint when the incumbent meets every constraint and we have zero signal it's failing. Spike work (labeling fixtures, converting alternative models to CoreML, building benchmark suites) is justified when a trigger fires, not preemptively.

This is the Sameer feedback `feedback_no_preemptive_evaluation` codified for this domain.

## Re-evaluation triggers

We swap MediaPipe Heavy for a different pose provider only if one of these fires:

1. **Real users report shot-form accuracy issues** that trace back to specific keypoint failures. Wrist tracking unreliable on dim courts; foot keypoints wobbly during pivot moves; visible misbehavior in the Compare view. Specific complaints, not abstract "could be better."
2. **A new SOTA pose model ships with proven iOS deployment.** Published CoreML or TFLite weights with iPhone-class benchmarks, Apache-or-similar license, explicit foot/heel topology. The exact thing the field is currently bad at.
3. **We acquire labeled basketball-shot data.** A real benchmark dataset (≥100 labeled clips with per-frame keypoints) makes empirical evaluation cheap. Without this, evaluation cost dominates value.
4. **Custom basketball-trained model becomes feasible.** Fine-tune of RTMW-l or a successor on Muse-collected shot data. Long-term roadmap; depends on (3).
5. **MediaPipe ships a meaningful version upgrade.** A 0.11 with materially different keypoint topology, accuracy, or deployment story. Re-validate against this fixture suite before auto-adopting.

## Re-evaluation flow when a trigger fires

1. Build labeled benchmark fixtures (~150-300 frames, manually labeled on the 6 critical joints: wrist, elbow, knee, ankle, heel, foot_index). One-time cost; the labels persist in `mobile/ios/Packages/PoseKit/Tests/Fixtures/` as the canonical benchmark.
2. Implement the candidate as a new `PoseProvider` in its own local Swift Package, sibling to `MediaPipeKit`.
3. Run contract tests (`PoseKit/Tests/PoseProviderContractTests.swift`) against the candidate to verify protocol conformance.
4. Run a per-keypoint pixel-error + iPhone-latency benchmark on the labeled fixtures. Compare against the incumbent.
5. Pre-register a success criterion (e.g., ≥10% improvement on ≥4 of 6 critical joints AND sustained ≥30fps on iPhone 17).
6. If the candidate wins: change one line at the app's call site (`let provider: PoseProvider = ...`). The architecture's whole point.
7. Document the result in a new `design/pose-model-evaluation-<YYYY-MM>.md`, leaving this 2026-04 doc as the historical baseline.

## Critical references

- Architecture plan: `design/MOBILE_ARCHITECTURE.md`
- v1 implementation plan: `design/FT_IOS_V1_PLAN.md`
- Vendoring rationale: `design/adr/0001-vendored-mediapipe-not-cocoapods.md`
- MediaPipe Pose Landmarker docs: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- RTMW paper (the documented next-evaluation candidate): https://arxiv.org/html/2407.08634v1
- Sapiens (license-blocked future-watch): https://github.com/facebookresearch/sapiens
- 2025 monocular 3D pose survey: https://www.mdpi.com/1424-8220/25/8/2409
