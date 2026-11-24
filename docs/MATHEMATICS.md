# Mathematical contracts, exact results, and unsolved research gap

## 1. What is actually certified by the practical algorithm?

On each 8×8 RGB block, the two candidate integer features are

\[
F_0(Y)=\sum_{r,c}w_{rc}^{44}(3G_{rc}-R_{rc}-B_{rc}),\qquad
F_1(Y)=\sum_{r,c}(-1)^{r+c}(2G_{rc}-R_{rc}-B_{rc}),
\]

where \(w^{44}=p\otimes p\), \(p=(1,-1,-1,1,1,-1,-1,1)\). The block decoder returns \(\mathbf1[F_m(Y)>0]\). Only green values are edited. With a fixed carrier and signed margin \(sF_m(Y)\ge m\), the minimum sum of squared **integer** edits reduces to a separable convex resource allocation; the exact marginal costs are \(1,3,5,\ldots\). The kernel uses an exact discrete water-filling formula, preserving clipping capacities in \([0,255]\). The global budget over disjoint blocks is exactly the sum of those optima. This does **not** make carrier selection globally optimal: mode 0 is tried first, mode 1 only if mode 0 is infeasible.

For a finite, predeclared integer attack-orbit \(Q_{q,o}(F)=\lfloor(F+o+\lfloor q/2\rfloor)/q\rfloor\), requiring a strictly positive quantized index for bit 1 and a strictly negative index for bit 0 is certified by the sharp **uniform signed magnitude**

\[
 m_0=\max_{q,o}\Bigl\{\lceil q/2\rceil-o,\;\lfloor q/2\rfloor+o+1\Bigr\}.
\]

The embedding algorithm enforces \(sF_m\ge m_0\) and automatically uses the largest feasible higher margin up to its declared maximum. The sharpness qualifier concerns the smallest *uniform* \(m_0\) sufficient for every modeled \((q,o)\), not the global minimum distortion of watermarking under all attacks.

**Important:** actual JPEG recompression, JPEG2000, H.264/H.265 video, frame geometry, chroma subsampling, RGB/YCbCr conversion and spatial rounding are not contained in that quantizer model. No actual-codec theorem is claimed.

## 2. Separate two-feature exact attack-frontier reference

A stronger mathematics-only module `frontier_exact_reference.py` solves the following distinct model:

\[
\min_{y_u,y_v\in\{0,\dots,M\}^{n}}
\|y_u-x_u\|_2^2+\|y_v-x_v\|_2^2
\quad\text{subject to}\quad
\mathbf1[Q_{t,1}(\sum y_u)>Q_{t,2}(\sum y_v)]=b
\quad \forall t.
\]

For nondecreasing scalar quantizers, fixing \(U=\sum y_u\) yields a one-dimensional frontier bound on \(V=\sum y_v\). Quantizer-index signatures are piecewise constant, so it suffices to search quantizer-signature intervals, choosing the closest sum within each interval. Separable integer marginal-cost tables give the exact cost at the chosen sums. Brute-force random tests verify the optimizer. **This reference theorem currently remains disconnected from the practical sign-carrier decoder and JPEG pipelines.**

## 3. Research gap not yet solved

A genuinely novel mathematical paper would require an original, nontrivial theorem extending to a meaningfully **coupled** image transform/quantization/reconstruction model, sharp feasibility or distortion bounds and a constructive solver that surpasses established integer resource allocation, QIM or known quantization-frontier optimization. Current results do not establish that theorem. Demonstrating 300/300 feasibility on a dataset also does not establish universal capacity/robustness.

## 4. Claims forbidden by current evidence

- Not “fastest watermarking method ever.”
- Not “JPEG/H.264 certified robust.”
- Not “globally optimal blind watermarking”.
- Not “first exact integer quantizer/waterfilling algorithm.”
- Not “all hosts and keys satisfy PSNR55.”
- Not “full 12-host × 6-method × 60-attacks comparison complete.”
