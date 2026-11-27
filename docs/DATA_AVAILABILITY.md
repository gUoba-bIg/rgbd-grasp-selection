# Data and runtime availability

The present repository distributes the first-party source snapshot, selected text configuration/registration inputs, reproduction wrappers and integrity manifests. It does not upload the complete local project, private submission documents, SSH credentials or operating-system installation files.

| Material | Current availability |
| --- | --- |
| First-party V1 implementation and selected text inputs | Included under `project/`, individually hashed |
| R6/R7 experiment and statistics wrapper source | Included |
| V1 robustness/single-factor wrapper source | Included |
| Original V1 archive identities | `manifests/archive_sha256.txt` |
| Omitted runtime files and assets | File-level path, size and SHA256 in `manifests/external_files.csv` |
| P3/P5 checkpoints and full observations | Retained in the author-held archives; not uploaded to this repository |
| Full R6/R7 raw evidence and supplementary results | Retained separately; not reproduced or regenerated for this upload |
| Custom PyBullet and compiled operators | External frozen binaries; required hashes retained |
| Complete offline Python/wheel environment | External archive set |
| Third-party P3 source | Obtain from the upstream project under its own terms and reconcile with the frozen runtime identity |

**Public download links for the author-held archives are not yet provided.** A manifest is an integrity reference, not a downloadable replacement for a missing file. Readers who require full replay should request the matching archive set through this repository's issue tracker until stable authorized links are supplied. No availability promise or access credential is implied by historical filesystem paths.

Upstream project references:

- GraspNet baseline: <https://github.com/graspnet/graspnet-baseline>
- GraspNet API: <https://github.com/graspnet/graspnetAPI>
- Bullet/PyBullet: <https://github.com/bulletphysics/bullet3>

Upstream repositories are discovery locations, not assertions that their latest revisions match the frozen project. Check the recorded file identities before reproduction. Redistribution rights for upstream software, datasets and weights remain separate from access to this repository.
