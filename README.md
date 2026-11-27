# RGB-D Grasp Selection

Source release accompanying **Six-Degree-of-Freedom Robotic Grasp Selection Based on RGB-D Instance Segmentation and Geometric Uncertainty Modeling**.

This repository makes the V1 method implementation, frozen experiment wrappers, supplementary-analysis code, and reproducibility manifests available for inspection. It covers Raw, A and C0, the R6 formal experiment, the R7 mechanism ablations, and the V1 robustness supplements. It does not incorporate the later V2/V3 methods or their results.

**Release status:** source and configuration snapshot. The model checkpoints, third-party runtime, custom native binaries, complete observations and full raw evidence are external dependencies. A clone alone is **not** a complete runnable reproduction package. Public download links for the author-held archives have not yet been attached; see [Data and runtime availability](docs/DATA_AVAILABILITY.md).

## Start here

1. Read the [code map](docs/CODE_MAP.md) for the active V1 entry points.
2. Read [reproducibility requirements](docs/REPRODUCIBILITY.md) before running any scientific script.
3. Check the downloaded source files without importing a model or simulation engine:

   ```sh
   git clone https://github.com/gUoba-bIg/rgbd-grasp-selection.git
   cd rgbd-grasp-selection
   python tools/verify_release.py
   ```

The verification command uses only the Python standard library and performs file-integrity checks. It does not run an experiment.

## Methods and experiment scope

- **Raw:** original P3 candidate score, after the frozen common eligibility procedure.
- **A:** the original RGB-D instance/geometry score with the deterministic (`none`) risk treatment.
- **C0:** the same score structure with the frozen (`fixed`) geometric-error propagation.
- All three methods share the candidate pool, common nonnegative-score domain, observed collision filtering, NMS and the historical executable-set (`E`) selection procedure.
- The historical decoder correction explicitly sets `dim=-1` in the cross product. The external corrected decoder file is identified by its full hash, rather than replaced by a different network.
- R6 comprises 720 method attempts; R7 comprises 240 exploratory ablation attempts. The supplementary single-factor study has 240 method results, including the reused 72-result initial subset; these are not 240 additional R6 trials.

The repository publishes the implementation and does not claim that C0 outperforms A. Historical failed attempts and the original experiment denominators are not reclassified by this source release.

## Layout

| Directory | Purpose |
| --- | --- |
| `project/` | First-party source and selected text inputs from the frozen execution archive, with the original relative directory structure and the declared publication metadata normalizations |
| `supplement/robustness_admission/` | V1 robustness and admission-diagnostic wrapper source |
| `supplement/singlefactor_expand20/` | V1 single-factor sample-expansion and statistics source |
| `analysis/r8/` | Historical manuscript statistics and plotting scripts |
| `reproduction/frozen_v1/` | Original frozen reproduction control source and locked requirements |
| `reproduction/final04/` | Final V1 reproduction adapter and its orchestration source |
| `engine_patches/` | Archived custom-engine build/parameter patches |
| `manifests/` | File identities, source mappings and external dependency hashes |
| `tools/` | Publication-only integrity tools |

Historical `smoke` directories inside `project/` remain because the original execution code imports helpers and registration records across them. They are not additional method contributions. The supplementary and reproduction directories are organized for source review; their original deployment layouts and external data are still required to execute them.

## Provenance

The V1 frozen identity is:

```text
tag:    paper_v1_r6r7_frozen
commit: e184a77ab06fe1f1c0a0f1514b04da65befcf8ed
```

That identity belongs to the preserved original archive. This GitHub repository begins a separate publication history and does not recreate or move the original tag. `manifests/source_files.csv` records the original and published SHA256 values. Of 3,179 source records, 3,166 are byte-identical and 13 have explicitly recorded publication-only metadata or wrapper-label normalization. Scientific scoring, control and evaluation computations are unchanged. See `manifests/publication_normalization.json` for the exact scope. Newly written documentation and the integrity checker are publication material.

No model inference, simulation, reranking or scientific reanalysis was performed to prepare this repository. Some host-runtime paths are represented by `${PYTHON_HOME}`, and unrelated host process listings are omitted from the public metadata. Other historical paths remain. These are documented publication views, not claims of new runtime observations or portable defaults; exact V1 reproduction must use the protected original archives.

## Licensing and external software

No new blanket software license is granted for the first-party source in this release. Third-party components remain subject to their original terms. In particular, the archived GraspNet baseline has a separate noncommercial research license and is not republished here. See [third-party notices](THIRD_PARTY_NOTICES.md). Do not treat this repository as permission to redistribute checkpoints, datasets, native packages or third-party code.

## Citation

The associated manuscript is not assigned a DOI or publication status in this repository. Use the final journal citation when it becomes available; no publication venue or acceptance is implied by this source release.

