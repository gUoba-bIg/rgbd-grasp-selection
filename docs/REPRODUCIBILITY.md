# Reproducibility scope

## What is verified in this publication

The execution archive and V1 control archive were checked against their frozen SHA256 values before extraction. Scientific computations were not edited. Supplement and final-adapter files were checked against the previously prepared submission manifest. Thirteen files subsequently received declared host-path, process-list or wrapper-label normalization in the public copy. Both original and published hashes are retained. The release checker validates the files in this repository only.

The original V1 frozen tag identifies the historical baseline, not the new publication commit. Historical cold-start results must not be confused with running a new cold start from this Git clone.

## What is still external

Execution requires the frozen model checkpoints, RGB-D observations/stable scene state, URDF and collision assets, P3 source/runtime, compiled operators, the exact custom PyBullet build, locked wheels, and the original case/input registries. Their absence is not replaced by stock PyBullet, freshly sampled candidates or a new model. The source repository does not include every archived input and result.

The frozen Windows setup used CPython 3.12, Torch 2.6.0+cu126, the recorded NVIDIA environment and separate vision/ranking/physics environments. Exact per-environment versions and wheel hashes are in:

- `reproduction/frozen_v1/locks/requirements_vision.txt`
- `reproduction/frozen_v1/locks/requirements_rank.txt`
- `reproduction/frozen_v1/locks/requirements_physics.txt`
- `reproduction/frozen_v1/metadata/binary_bindings.json`

These are hash-locked historical installation identities, not a claim that a fresh public package-index install currently supplies every matching custom wheel.

## Intended restoration workflow

1. Obtain the complete author-held archive set and applicable third-party permissions. Check `manifests/archive_sha256.txt` and the file-level external manifest.
2. Use a new workspace and output directory, retaining the archive originals as read-only input.
3. Restore the original archive layout and use the final04 adapter, including its registered runtime supplement, rather than assuming the original pre-final04 control scripts already incorporate every packaging fix.
4. Verify the actual loaded module paths, engine SHA256, model hashes, case identity and seeds before executing a case.
5. Run only the registered workflow with complete inputs; retain failures and compare to the frozen expected categories. Never use the old results as replacement execution output.

The three original representative case identities are R6FORMAL-007 (success), R6FORMAL-004 (empty E) and R6FORMAL-001 (static entry not certified). Their frozen input/acceptance metadata is under `reproduction/frozen_v1/metadata/`. This publication has not rerun those cases.

## Path portability

Original wrappers may use historical absolute Windows paths or assume a particular parent-directory layout. In the public metadata, `${PYTHON_HOME}` is a documented deployment placeholder, not an automatically expanded or observed runtime path. One publication wrapper uses the `research/` branch namespace. The organized `supplement/` and `reproduction/` folders are source-delivery locations, not an independently validated drop-in installation layout. Exact source-identity checks and cold-start reproduction require the protected original archives, not substitution of the normalized metadata into the frozen baseline.

## Safe inspection

```sh
python tools/verify_release.py
```

This reads repository files and returns a failure status for missing, altered or unsafe manifest paths. It does not load the experiment code, install dependencies, connect PyBullet, invoke models or write into historical outputs.
