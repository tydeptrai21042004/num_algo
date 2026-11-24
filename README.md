# FC-AOIP — frozen proposal, expanded experimental suite

One original FC-AOIP implementation, five classical paper-derived modulation adapters, and the original 60-attack protocol. This release extends evaluation only. No ECC, temporal payload repetition, denoising, attack-dependent embedding, or new watermarking proposal. MM-Watermark/WAVES are not included.

## Ubuntu / WSL2

```bash
sudo apt-get update && sudo apt-get install -y python3-venv ffmpeg curl
bash setup.sh
source .venv/bin/activate
# Offline test on bundled legacy image and synthetic video:
bash run_all.sh --profile smoke --resume
# Complete independent-data experiment (automatically downloads data):
bash run_all.sh --profile paper --resume
# Optional 8K, 100,000-operation, multi-stream and large raw video stress:
bash run_all.sh --profile stress --resume
```

Use a new output directory if source code, configuration, data, or environment changed. `--resume` is strict: it skips only atomically committed cases with identical fingerprints. Recorded failure rows remain part of the experiment; retry a failure in a new run. No attack subsets in the expanded robustness runners, including smoke. The paper profile can require many hours or longer and substantial disk space. DIV2K official full HR archives are downloaded even for the lite profile. Raw 4K stress videos add several GB; lossless intermediate video files are removed by default.

## Data protocol

| Partition | Paper/stress images | Use |
|---|---:|---|
| Legacy | 12 bundled BMPs | Previously used; not unseen |
| Train | 100 DIV2K training IDs | Hyperparameter analysis, baseline calibration |
| Validation | 50 disjoint DIV2K training IDs | Held-out verification of calibrated settings |
| Test | 100 DIV2K official validation + 24 Kodak | Final independent evaluation |

The official DIV2K validation partition is used as our watermark test split (not DIV2K's hidden challenge test). There is no learned model. Split seed 2026, source-level split before resize, exact SHA-256 and decoded-pixel hashes (native and processed) checked for duplicate leakage, including legacy overlap. Standard comparison uses RGB 512x512 Lanczos resize and 4096 bits. Native source files remain unmodified. The lite profile uses nested 6/3/10 train/validation/test subsets (6 DIV2K validation + 4 Kodak). Test results are not used for calibration. Default FC-AOIP remains unchanged; calibrated baseline strengths are chosen using training data, evaluated on validation, then frozen. Both original-strength and calibrated-strength baselines are reported separately. Matching is approximate; report actual PSNR and qualification instead of asserting exact distortion matching.

Automatic video data: Big Buck Bunny development, Sintel validation (W3C-hosted Blender trailers); Xiph vidyo1/3/4 and FourPeople test clips (lossy source WebM); stress adds native 4K crowd_run and 1080p ducks_take_off Y4M. Separate source sequences, not randomly split frames. UVG is not bundled or automatically downloaded in this release because a stable direct publisher asset URL was not verified; authorized local UVG files can be tested via `--input`. Video source attribution and probe metadata are stored. See `docs/DATA_SOURCES.md`. Synthetic footage is only a labeled offline diagnostic. Download failures stop the data stage; no synthetic fallback.

```bash
python scripts/download_images.py --profile paper
python scripts/download_videos.py --profile paper
python scripts/verify_data_splits.py
python scripts/verify_frozen_core.py
```

## Experiments

| Group | Coverage |
|---|---|
| mathematics | DP/heap exact cost references, separate exhaustive frontier model, quantizer threshold sharpness, clipping/feasibility, global margin budget reference |
| hyperparameters | 115 settings: PSNR, steps, offsets, maximum margin, carrier policy, 81 joint combinations |
| calibration | Strength sweeps for all five baselines, training-only |
| validation | Held-out check of selected baseline strengths and frozen proposal |
| robustness60 | Original 60 attack conditions for six methods, legacy and independent test |
| ablation | Five original ablations, full 60 conditions |
| payload_robustness | 64, 256, 512, 1024, 2048, 4096 bits, full 60 attacks |
| security | Correct-key, wrong-key, unmarked controls and detection thresholds |
| compute | Resolution (up to 8192x8192 in stress), payload, 200 repeat x 5 round timing, kernel-only timing, worst-case inputs, memory, endurance, 1–16 workers |
| video | Actual RGB FFV1 lossless control, MPEG-4, H.264/H.265 CRF 18/23/28/35, VP9, AV1, RGB lossless H.264, chroma 4:4:4, recompression; paced file and webcam queue/deadline tests, multiple streams |
| video_frames60 | Sample actual video frames and evaluate exactly the original 60 image attacks |

Full paper launcher uses 12 training hosts and one key for the large 115-configuration hyperparameter sweep, all 100 training hosts and one key for calibration, all 50 validation and 124 test hosts with five keys. The original six-method comparison has 244,800 scheduled attacked rows across legacy and test, plus the separate five calibrated baseline configurations; this includes explicit failure rows, not necessarily successful recovery. You can expand hyperparameter sweeps to all training hosts by invoking the independent runner without `--limit`.

```bash
# Full training sweep (all 100 selected training images)
python scripts/run_experiments.py --group hyperparameters --profile paper --split train --keys 1 --out results/hyper_all --resume
# Full six-method timing / compute stress
python scripts/run_experiments.py --group compute --profile stress --out results/stress/compute --resume
# Optional duration stress, at least 10,000 operations and 3600 seconds per operation type
python scripts/run_experiments.py --group compute --profile stress --operations 10000 --seconds 3600 --out results/hour_compute --resume
# Final independent image benchmark, after running the paper calibration/freeze stages
python scripts/run_experiments.py --group robustness60 --profile paper --split test --frozen results/paper/frozen_config.json --out results/paper/robustness60 --resume
# Extra multi-key robustness, same 60 attacks (expensive)
python scripts/run_experiments.py --group robustness60 --profile paper --split test --keys 25 --methods FC-AOIP --frozen results/paper/frozen_config.json --out results/multikey25 --resume
# Real video codec tests, optionally keep encoded videos
python scripts/run_experiments.py --group video --profile paper --split test --video-mode codec --out results/video_codec --resume
# Local authorized source / UVG clip
python scripts/run_experiments.py --group video --profile paper --input /path/to/video.mp4 --methods FC-AOIP --out results/local_video --resume
# Paced file endurance. Looping is explicit; not native-video accuracy evidence.
python scripts/run_experiments.py --group video --profile paper --input data/videos/test/vidyo1.webm --methods FC-AOIP --video-mode stream --fps-list 60 --seconds 1800 --loop-input --out results/video_endurance30m --resume
# Actual webcam (Linux /dev/video0 or supported WSL camera passthrough)
python scripts/run_experiments.py --group video --profile paper --input camera:0 --methods FC-AOIP --video-mode stream --fps-list 30 --seconds 600 --out results/webcam10m
# Original 60 attacks on sampled video frames
python scripts/run_experiments.py --group video_frames60 --profile paper --split test --frames 10 --out results/video60 --resume
```

Use `--fps-list 24 30 60 120 240` for paced load; these are target arrival rates, not measured source-native rates. For 10/30/60 minute tests use `--seconds 600/1800/3600`. No automatic access to webcam. Use `FCAOIP_THREADS=1` (default) for primary latency comparisons and explicitly record alternative hardware/thread settings. The batch worker test measures end-to-end spawned-process throughput, including startup; isolated kernel is not compared to full baseline APIs.

## Outputs and interpretation

- Each experiment directory has manifest, checkpoint SQLite, per-case CSV tables, counts, failures JSONL. Video retains sidecar public metadata and source manifests; encoded media is opt-in with `--keep-videos`. CSV uses `inf` for identical images.
- `python scripts/summarize_results.py --root results/paper --out results/paper_ready`
- `python scripts/generate_figures.py --root results/paper --out results/paper_ready/figures`
- Results report all attempts, qualified clean/PSNR cases, paired host/attack results, host-cluster bootstrap intervals, BER/NC/SSIM/SSE and latency quantiles. Pure compute and certificate tests do not apply attacks. Video codec transforms are a distinct application protocol, not additional image attack IDs.
- Baseline adapters are limited to 512x512 RGB / 4096 bits. Only FC-AOIP runs variable resolution/payload. No faster-than-all or robustness guarantee is inferred from tests. Restricted feature quantization certificate is not JPEG/H.264 certification. Solver margin and signed feature magnitude use different units and are logged separately. Adaptive carrier returns the first feasible mode; do not call it a globally optimal selection across both carriers.
- FFV1 RGB pixel equality is explicitly checked. H.264/H.265 etc. are actual encoded and decoded frames. `psnr_to_marked_db` is codec-induced distortion relative to marked RGB, not host-to-decoded PSNR. Codec read/evaluation wall time includes reference evaluation; paced streaming timing excludes codec and is labeled accordingly. Camera hardware frame loss is unknown; queue drops are measured. Finite source exhaustion is reported, never silently replaced with repeated frames. Raw bit and mode metadata is frame-indexed; no synchronization recovery. Peak process RSS is lifetime high-water, not isolated allocation size.
- `docs/LOCAL_VERIFICATION.md` describes the executed tests and limitations. Full independent data and long stress runs have **not** been executed in the delivery environment. Smoke results are not publication timing evidence.

## Layout

```
fcaoip/        frozen proposal and exact reference
baselines/     frozen five adapters
src/           frozen 60 attack definitions and baseline support
experiments/   dataset, checkpoint, math, image, compute, video modules
configs/       frozen source hashes and experiment catalog
scripts/       data download, experiment runner, freeze, summaries, figures
 data/         bundled legacy images and synthetic diagnostic
 tests/        original and added verification tests
 docs/         mathematics, provenance, experiment documentation
 results/      clean output tree
```
