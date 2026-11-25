# OMIP — Optimal-Margin Integer Projection

Official publication protocol: 12 bundled hosts, 4 calibration + 8 held-out evaluation, original C/CC logos, five keys, 4096 bits, 55 dB, and 60 image attacks. No additional image dataset or validation set. The internal method identifier `FC-AOIP` is retained for backward-compatible CSV readers; it denotes the new OMIP implementation in this release, not the previous algorithm.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/verify_frozen_core.py
python -m pytest -q
bash run_12_images.sh --resume
```

The official `main` stage runs attack audit, mathematical tests, baseline calibration, hyperparameter sensitivity, image robustness, ablation, payload, security, clean video, compute, summary, paired statistics and result audit. Outputs are in `results/omip_final60_cleanvideo/`. Resume accepts only matching source/config/environment fingerprints. Unexpected errors are retried; recorded distortion-limit outcomes are not silently replaced. Use `--stage` to run a particular stage. `--video-profile lite` reduces clean resolution scaling; `--skip-download` uses the verified four-video cache. No video attack stage exists. FFV1 is lossless storage and pixel-integrity verification only. Synthetic video is for tests only. Official video downloads use the four Xiph test clips; C only, fixed key per clip, 4096 bits, native aspect ratio (rounded to an 8-pixel grid), resolution scaling and 24/30/60 FPS paced processing. Native marked clips are retained. Paced timing excludes file encoding; drops/deadlines are reported separately.

## Method

Single fixed DCT44 sign pattern applied to `3G-R-B`. Only green samples change, using exact bounded integer water filling. Maximize signed feature margin under the explicit SSE budget, then minimize SSE at that margin. Natural upper bound is the minimum per-block signed feature plus three times the directional green capacity. No carrier switching, fallback, public mode descriptor, denoising, ECC, repetition, registration, or attack-dependent embedding/extraction. SHAKE256-derived keyed ordering replaces the affine block schedule (128-bit integer key interface; experiment keys are reproducible and not confidential). The restricted scalar quantization threshold is a mathematical guarantee for that model only, not a guarantee against image attacks. If the budget cannot support the requested margin, report `distortion_limit`. Never substitute an unmarked frame. Alternative carriers and sequential allocation are experimental controls only.

## Attack protocol

`src/attack_suite60.py` is the canonical registry. 45 classical manuscript conditions plus 15 distinct additions; IDs A001–A060; C000 separate. Normalized Gaussian variances 0.003, 0.005, 0.05; lossy JPEG/JPEG2000/WebP use actual bitstreams. Combined operations run in declared order. Low-pass uses OpenCV's 5×5 Gaussian kernel (automatic sigma); sharpness 0.5 means softening using Pillow. Rotation/translation/shear use black borders and bilinear interpolation; 25% central occlusion and 50% crop refer to area; border crop removes 16 pixels per side then pads black. Resizing restores original canvas, using area for downsampling and bicubic for upsampling. Histogram equalization is luminance-only YCrCb. These conventions resolve unspecified details in the reference manuscript and are not a claim of byte-identical reproduction of its original implementation. All 60 conditions are used only for image experiments. Real visual evidence is saved from the numerical evaluation, with hashes, pixel differences, attack PSNR/SSIM/MAE/MSE, BER, binary NC, bipolar NC and balanced accuracy. Difference-map amplification is labeled. The main visual subset is the first held-out host, key index 0, both original logos; the preflight audit uses Airplane/C. No best-result selection.

## Experiments

Five baseline adapters are unchanged. Calibration uses the four training hosts, one key per logo, then freezes the chosen strengths; hyperparameter sweeps are descriptive, not test-set tuning. Main comparison: 480 cases and 28,800 scheduled attack rows. Ablation: full, fixed minimum margin, cyclic two-unit sequential allocation, green-only and fixed checkerboard control. The sequential control visits pixels in fixed raster cycles, taking up to two integer units per visit; it does not minimize SSE and avoids the pathological saturation of one pixel before proceeding. Equal-margin sequential SSE is exported for full ablation cases; matched-budget robustness is evaluated for all variants. Identical output hashes are reported, not manipulated. Payload and security are proposal only. Security includes the old equivalent-key regression and fixed-logo/constant/random controls; these are not proof of cryptographic security. Compute includes calibrated baseline timing on held-out hosts and separate stress tests. Summary reports returned and qualified results; paired sign tests use eight independent hosts, Holm adjustment and coverage. Previous results are incompatible with the revised algorithm and attacks.

## Validation

See `docs/OMIP_REVISION.md` for the new specification and test limitations. Old FC-AOIP verification reports in this repository are historical, not evidence for OMIP. Run a fresh full experiment before updating manuscript numbers.
