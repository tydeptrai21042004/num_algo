# Original C/CC payload update — local verification

Source: num_algo-main(2).zip. Changes are limited to experimental drivers, reporting, tests and documentation. The two original PNG files and frozen algorithm/baseline/attack sources are unchanged.

- Python compileall: passed.
- Frozen core: 53b1478fe0ff8cef2c420192c84dd29decae9131c3dbeb94c35c41d6b9492f97.
- Attack protocol: ff35f6d9f1b5544a45d23147ac11691600067ef90e65159b0c0ba5a546edd16e.
- Full pytest run: 33 passed in 40.43 s (includes two-logo image attacks, payload tests, security, FFV1/H.264 smoke, and statistical pairing).
- Additional clean verification: 8 held-out BMP hosts x 5 original schedule keys x 2 logos = 80/80 qualified; minimum PSNR 55.00136276616609 dB; maximum clean BER 0.0.
- C contains 2059 white bits; CC contains 2689 white bits after white alpha compositing and thresholding. The payloads are distinct and unchanged across keys.
- No new complete paper-profile attack/video/compute run has been executed. The original random-payload scientific results must not be reused as logo results.

Use `bash run_12_images.sh --resume` for the compact logo study (new output root: `results/compact12_real_watermarks`). Optional compute: `bash run_12_images.sh --stage compute --resume` followed by `bash run_12_images.sh --stage summary --resume`.
