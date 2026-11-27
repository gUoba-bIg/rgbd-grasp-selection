# V1 code map

Paths in this document are relative to the repository root. Scientific computations are unchanged; the limited public metadata and wrapper-label adjustments are recorded in `manifests/publication_normalization.json`.

| Stage | Main source |
| --- | --- |
| RGB-D preprocessing and frozen P3/P5 inference | `project/IP140_R2_R4dev72_cross_dim_explicit_resume01/implementation/policy_network.py` |
| Original shared eligibility and Raw/A/C0 score calculation | `project/IP140_R2_R4dev72_cross_dim_explicit_resume01/implementation/policy_rank.py` |
| Geometry, association and error propagation helpers | `project/src/p56_geometry_v51.py`, `project/src/risk_observations_v31.py` |
| RGB-D network wrapper | `project/src/modal_fusion_v38.py` |
| Instance-mask decoding | `project/scripts/extract_mask_scores_v17.py` |
| Frozen calibration and scope | `project/configs/risk_calibration_v31.json`, `project/configs/p56_joint_v51r2.json` |
| Common candidate/source checks | `project/IP140_R2_R4dev72_cross_dim_explicit_resume01/implementation/boundary_replay.py`, `intent_provenance.py`, `cross_unit_audit.py` in the same directory |
| Old-E selection, static checking and execution | `platform_select.py`, `platform_static.py`, `platform_trial.py`, `platform_episode.py` in the same implementation directory |
| R6 final continuation and exact result classification | `project/IP140_R2_R6formal720_static_distance_uncertified_resume01/code/` and `implementation/` |
| R6 original design and input sealing | `project/IP140_R2_R6formal720_raw_a_c0_paired01/code/` |
| R7 mechanism ablations | `project/IP140_R2_R7ablation240_mechanism01/code/` and `implementation/` |
| IP140 registered execution baseline | `project/IP140_R2_smoke22_miss_negative_controls_full_execution01/`, including its historical imported helpers |
| V1 supplemental robustness study | `supplement/robustness_admission/` |
| Expanded single-factor study | `supplement/singlefactor_expand20/` |
| Historical statistical recomputation | `analysis/r8/recalculate.py` |
| Final cold-start adapter | `reproduction/final04/adapter/` and `reproduction/final04/ops/` |

The upstream P3 model/decoder and its compiled operators are external. The exact archived files under `runtime/build/graspnet/` are listed in `manifests/external_files.csv`. The runtime decoder correction is `torch.cross(axis_x, axis_y, dim=-1)`; an arbitrary unmodified upstream checkout is not a substitute for the frozen runtime.

## Reading the score

The authoritative implementation is `policy_rank.py`. It uses the raw candidate score multiplied by `contact_support` and `(1 - collision_proxy)` for A/C0. A uses the `none` branch; C0 uses the frozen `fixed` prior. Its historical support and tie-breaking behavior are intentionally preserved. This release does not replace them with later three-state evidence semantics.

Candidate ranking is not the same as the finally executable candidate: the original common eligibility and `E` selection must also be applied. Do not infer an execution result from score alone.

## Historical helpers

The directory names encode provenance. References to earlier development or audit directories in frozen source do not mean those experiments are being rerun or included in the manuscript denominator. Removal/renaming of these directories may break source-path or hash-based registration checks. Data and registration files excluded from the source release remain listed in the external manifest.
