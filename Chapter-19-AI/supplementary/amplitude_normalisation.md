# Why whitened amplitudes broke the model, and the one line that fixed it

This note expands on Box 3 of the chapter. It explains where the factor of 32
comes from, why a neural network should care about a linear rescaling at all
when a matched filter does not, and why the fix is architectural rather than a
division.

Everything here can be reproduced from this directory. The three sensitivity
curves referred to below are in `original_model/figures/` and
`updated_model/figures/`, and a longer write-up with the full derivation is in
`whitening_normalisation.pdf`.

---

## The symptom

The first model trained on Gaussian noise and evaluated on MLGWSC-1 dataset 4
reached roughly 350 Mpc at the loosest false-alarm rates and collapsed below
100 Mpc by a few hundred false alarms per month
(`original_model/figures/sensitive_distance_unscaled.png`).

Dividing both the training data and the dataset-4 strain by 32 — changing no
weight, no hyperparameter and no training sample — lifted it to about 700 Mpc
and pushed the collapse back
(`original_model/figures/sensitive_distance_scaled.png`).

Both runs were internally consistent: the same scaling was applied at training
and at evaluation in each case. This is therefore **not** a train/test
mismatch. It is the absolute scale of the numbers entering the first layer.

## Where 32 comes from

Whitening is defined by what it does to the *spectrum*, not to the amplitude.

PyCBC's forward transform approximates the continuous Fourier integral, so it
carries the sampling interval, and the inverse carries the reciprocal of the
segment length:

```
x̃(f_k) = Δt · Σ_n x_n e^{-2πikn/N}        x_n = (1/T) · Σ_k x̃(f_k) e^{+2πikn/N}
```

With the one-sided PSD `S(f)` defined so that the expected squared transform is
`T·S(f)/2`, dividing by the amplitude spectral density gives a whitened series
whose own one-sided PSD is exactly 1 per hertz:

```
w̃(f) = x̃(f) / √S(f)     ⟹     S_w(f) ≡ 1
```

That is the whole content of whitening: it flattens the spectrum to unity, not
to unit variance. The variance of a time series is the integral of its
one-sided PSD over the analysed band, so for a flat unit spectrum that integral
is simply the bandwidth:

```
σ_w² = ∫₀^{f_Nyq} S_w(f) df = f_Nyq = f_s/2        σ_w = √(f_s/2)
```

At 2048 Hz that is **exactly 32**. The factor is the square root of the Nyquist
frequency and scales with the sample rate: the same pipeline at 4096 Hz would
produce amplitudes larger by another √2.

Verified numerically by synthesising noise with a known coloured spectrum,
whitening it with PyCBC's convention written out explicitly, and measuring:

| sample rate | √(f_s/2) | measured σ | whitened PSD |
|---|---|---|---|
| 1024 Hz | 22.627 | 22.582 | 1.000 |
| 2048 Hz | 32.000 | 31.922 | 1.000 |
| 4096 Hz | 45.255 | 45.186 | 1.000 |

A consistency check closes the loop: if whitened noise is Gaussian with σ = 32,
the largest excursions in a long record sit near 4.5σ ≈ 144. The ±150 range
observed in the data is exactly what that predicts, so those values were never
anomalous.

Packages that return unit-variance output — `ml4gw`, Bilby — have by
construction divided by the same quantity. Dividing PyCBC's output by 32 does
not merely rescale it into a convenient range; it converts it into the
convention those packages use, and therefore the convention that published
architectures and their default initialisations were designed around.

## Why the network cared

A matched filter is immune: its statistic is normalised by the template, so the
scale cancels. A network has no such protection, and the reason is specific
enough to be worth stating.

**The positional embedding was swamped.** The tokenizer is a strided
convolution whose output is proportional to its input. The learned positional
embedding is initialised with a standard deviation of 0.02 and — in the
original architecture — was added to the tokenizer output *before any
normalisation layer*:

| input σ | token std | pos. emb. std | ratio |
|---|---|---|---|
| 1 | 0.58 | 0.02 | 29× |
| 32 | 18.51 | 0.02 | **925×** |

At unit input the positional signal is about one part in thirty of the token it
rides on — small, but recoverable, since the embedding is learned and gradient
descent can grow it. At σ = 32 it is one part in nine hundred. The first
LayerNorm inside the encoder then rescales the whole token, preserving that
ratio, so the positional contribution is effectively erased. Attention is
permutation invariant without it, which leaves the network largely unable to
use *where* in the window a feature occurred — precisely the information a
search needs to localise a chirp and check that two detectors agree.

**The residual stream was dominated.** The encoder uses pre-normalisation, so
each block computes an order-unity update and adds it to the running
representation. With tokens of magnitude 0.58 such an update changes the
representation by most of its value; with tokens of magnitude 18.5 it perturbs
it by about five percent. The stack is pushed towards the identity and most of
its capacity goes unused.

**The initialisation assumption was violated.** Kaiming and Xavier schemes
assume activations have roughly unit variance so that signal neither explodes
nor vanishes with depth. An input variance 1024 times larger than intended
breaks that at the very first layer. AdamW mitigates the resulting
gradient-scale mismatch, since its per-parameter normalisation is scale
invariant once the moment estimates settle, but it does nothing about either of
the two effects above, which are properties of the forward pass.

## Is this only a transformer problem?

Partly. A convolutional network has no positional embedding — convolution is
translation equivariant by construction and position is carried implicitly by
the feature map — so the worst of the three does not arise.

But a plain convolutional stack acquires a different failure, because ReLU is
positively homogeneous: `relu(a·z) = a·relu(z)` for positive `a`. A scale
factor at the input passes through every layer untouched and emerges
undiminished at the output. Measured on a five-layer ReLU stack:

| input σ | logit std | with BatchNorm |
|---|---|---|
| 1 | 0.030 | 0.0508 |
| 32 | 0.905 | 0.0508 |

The logits scale by exactly 32. Biases stop mattering — their contribution to
the first layer falls from about 12% of the signal to 0.4% — and inflated
logits drive the cross-entropy into its flat region as training proceeds. A
normalisation layer placed early removes the input scale outright, which is why
the BatchNorm column is identical at both scales.

**The principle that governs it** is not about convolutions or attention: *an
architecture is exposed to the amplitude convention of its input exactly as far
as that amplitude travels before the first normalisation, and exactly where it
meets a parameter whose scale is fixed independently of the data.*

## The fix

```python
# original — positional embedding added to the raw tokenizer output
tokens = self.tokenizer(x).transpose(1, 2)
tokens = tokens + self.positional

# corrected — tokens normalised first
tokens = self.conv3(x).transpose(1, 2)
tokens = self.encoder(self.tok_norm(tokens) + self.pos)
```

A LayerNorm immediately before the addition fixes the token magnitude at unity
whatever the input scale, so the positional embedding always meets tokens of
the size it was initialised for.

The control that settles it: the corrected model, given the **unscaled** data
that produced the collapse, reproduces the scaled result almost exactly
(`updated_model/figures/sensitive_distance_unscaled.png`). The scaling had
never been supplying information. It had been compensating for a property of
the architecture.

**The corrected architecture is what every result in the chapter from this
point on uses.** It is defined in
[`common/coherent_transformer_model.py`](../common/coherent_transformer_model.py),
and the coincident variant in
[`common/coincident_transformer_model.py`](../common/coincident_transformer_model.py)
places its token normalisation the same way.

## Practical advice

- Divide PyCBC-whitened strain by `sqrt(f_s/2)` as part of the whitening step,
  not as a separate scaling applied later. Deriving the factor from the sample
  rate removes a silent dependence; a hard-coded 32 is correct only at 2048 Hz.
- Record the convention in the dataset. A `whitened_sigma` attribute written
  alongside the data lets a consumer verify it rather than assume it.
- Check the amplitude whenever a new whitening path is introduced. A single
  comparison of the measured standard deviation against the training set would
  have caught this before a model was trained.
- Prefer a fixed constant to a per-window normalisation. InstanceNorm would
  also hide the convention, but it divides each window by its own standard
  deviation, and a loud signal inflates that — suppressing the amplitude by
  about 8% at SNR 20 and 33% at SNR 50, which is exactly the information a
  search needs. A constant preserves every amplitude ratio exactly.
