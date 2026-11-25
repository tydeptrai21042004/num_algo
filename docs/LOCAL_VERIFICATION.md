HISTORICAL FC-AOIP DOCUMENT — superseded by README.md and docs/OMIP_REVISION.md for the current protocol.

# Delivery verification

- Original source baseline: `FC_AOIP_One_Proposal_5_Baselines_Ablation_Video_2026-11-23(1).zip`.
- Frozen core fingerprint: `53b1478fe0ff8cef2c420192c84dd29decae9131c3dbeb94c35c41d6b9492f97`.
- Original 60-attack protocol fingerprint: `ff35f6d9f1b5544a45d23147ac11691600067ef90e65159b0c0ba5a546edd16e`.
- Final pytest: **23 passed**, including all 13 original tests. Added tests cover exact split ID selection, duplicate image rejection, local transfer and cache tamper detection, safe extraction, resume fingerprints, 60-condition failure coverage, DP/heap reference, 8K schedule and pixel-exact FFV1 decoding.
- Executed during implementation on bundled legacy image: mathematical smoke (84 rows, 78 ok, 6 expected infeasible; no mismatches), 11 hyperparameter cases (660 attack rows, 600 successful attacks, 60 not embedded), 10 baseline calibration smoke cases (600 attack rows), held-out runner interface smoke on legacy data (360 rows; NOT independent validation).
- Six-method full-60 smoke: 360 attacked rows, all returned; all five ablations: 300 attacked rows, all returned. Compute smoke: 50 rows, 46 ok and 4 infeasible. Key discrimination: 30 rows. Video smoke: 18 marked frame attempts, 16 qualified; 36 decoded codec frame rows across FFV1 and H.264. Six stream smoke cases ran. Paper-ready summary and figure generators ran against these results.
- After the final video/reference and payload additions: 2-frame FC-AOIP FFV1 and H.264 CRF23 smoke returned all frames; FFV1 pixel equality 2/2, BER 0; H.264 qualified mean BER 0.4976806640625 (not robust). Payload robustness 120 rows (two payload lengths x 60). Video frame robustness 60 rows. All returned.
- A one-shot smoke launcher invocation hit the execution-time limit after its first four groups; remaining groups were exercised separately. It is not claimed that the entire final launcher completed in one uninterrupted invocation. Changes to dataset download/profile handling were unit-tested; full publisher downloads failed DNS access in this environment. The real-data pipeline, long endurance, 8K stress, all codecs, camera hardware, and cross-hardware experiments remain to be executed on the target machine. No synthetic data is substituted for independent datasets.

These are verification runs, not final scientific results. Timing from concurrent smoke execution is not suitable for paper performance comparison. The new release has a clean results directory; no historical performance data is passed off as a new experiment.
