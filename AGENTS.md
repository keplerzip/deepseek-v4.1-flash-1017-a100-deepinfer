# DeepSeek 1017 A100 maintenance

The deployment owner designated R9 as the formal stable baseline on 2026-11-16.

- Treat tag `r9-stable-20261116` and branch `stable/r9` as fixed historical refs. Do not move, overwrite, force-push or add updates to them.
- Do not change the frozen R9 runtime, configuration, image or offline archive. Its identities and approved settings are in FINAL-RELEASE.json and docs/freeze-policy.md.
- Future implementation work must use a separate version directory and development branch, starting with R10, based on the stable R9 snapshot. Keep the existing R9 deployment and deliverables intact.
- Preserve 1017, port 8005 and the established connection/authentication settings unless the user explicitly requests a change for a new version. Performance testing remains separate from startup and keeps the API available; no automatic operator tests.
- CPU KV offload stays disabled; Engram parameter CPU offload remains enabled. Do not add approximate FFN/replay changes to R9.
- The latest R9 operator report is benchmark PASS, diagnosis PASS, contamination []. Do not invent diagnosis subgroup numbers or quality/tool/actual-512K acceptance that was not provided.
- Runtime build strings (EXPERIMENTAL/candidate) identify the original bytes; promoting the existing artifact to stable is not permission to rebuild or relabel its payload.
- This current publication may register the stable status and policy, then freeze it. Subsequent changes require a new version as directed above.
