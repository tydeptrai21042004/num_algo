# Five classical baseline adapters — primary research sources

The following are paper-derived modulation adapters inherited from the user's supplied `ALgo-main(4).zip`. They are **not guaranteed source-code-level reproductions of the authors' original experimental pipelines**. Shared block selection, bit payload, RGB handling, image PSNR, attack protocol and timing are explicitly declared in the code.

1. **Scalar-QIM** — Brian Chen and Gregory W. Wornell, *Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding*, IEEE Trans. Information Theory 47(4), 2001. https://doi.org/10.1109/18.923725
2. **STDM-QIM** — Chen and Wornell, same foundational QIM/dither-modulation paper, spread-transform dither implementation; uses a seeded projection over the supplied DCT adapter.
3. **SCS (Scalar Costa Scheme)** — Joachim J. Eggers, Richard Bäuml, Roman Tzschoppe and Bernd Girod, *Scalar Costa Scheme for Information Embedding*, IEEE Trans. Signal Processing 51(4), 2003. https://doi.org/10.1109/TSP.2003.809366
4. **ISS (Improved Spread Spectrum)** — Henrique S. Malvar and Dinei A. F. Florêncio, *Improved Spread Spectrum: A New Modulation Technique for Robust Watermarking*, IEEE Trans. Signal Processing 51(4), 2003. https://doi.org/10.1109/TSP.2003.809385
5. **SS (additive spread spectrum)** — spread-spectrum reference principle discussed by Malvar and Florêncio (2003), using the declared paper-derived image adapter.

**Method naming is not an endorsement of theoretical or experimental exact equivalence with the original papers.** Parameter selection and unqualified results are fully reported, not silently replaced by other methods.

Open W3C multimedia demo video reference: https://www.w3.org/2010/05/video/mediaevents.html (uses `https://media.w3.org/2010/05/sintel/trailer.mp4`). The downloader works on direct authorized HTTPS media and does not scrape services.
