# Riemannian Conditional Latent Flow for Text Continuation

This project studies text continuation in the latent space of a frozen encoder-decoder model. Stage 1 learns a compact latent representation for BERT token sequences. Stage 2 learns a conditional flow that maps Gaussian latent noise into suffix latents, conditioned on a prompt prefix. The main idea is to replace a flat Euclidean flow-matching objective with a learned diagonal Riemannian metric over the latent space.

## Paper Thesis

Language latents produced by a neural encoder-decoder are not uniformly meaningful under the standard Euclidean metric. Small perturbations in different latent directions can have very different effects on decoded text. We therefore learn a conditional diagonal metric and use it to weight the flow-matching objective. This lets the model emphasize latent directions that are more important under the learned geometry while keeping generation in the same frozen decoder space.

One possible paper title:

**Prompt-Conditioned Riemannian Flow Matching in Frozen Language Latent Space**

## Model Overview

The system has two stages.

**Stage 1: Frozen latent autoencoder**

An encoder maps token ids and attention masks to hidden states. A parallel decoder compresses those hidden states into latent vectors and reconstructs tokens from latent sequences. After Stage 1 training, both encoder and decoder are frozen.

Let the frozen encoder-decoder produce latent sequence:

```text
z = [z_1, ..., z_S],     z_i in R^d
```

The first `PROMPT_LEN` latents are used as the condition:

```text
c = [z_1, ..., z_P]
```

The target suffix is:

```text
x_1 = [z_{P+1}, ..., z_S]
```

**Stage 2: Conditional Riemannian flow**

The flow network learns a velocity field:

```text
v_theta(z_t, t, c, p)
```

where `z_t` is an interpolated noisy latent, `t` is diffusion/flow time, `c` is the prompt condition, and `p` is normalized token position.

The metric network learns a positive diagonal metric:

```text
g_phi(z_t, t, c, p) in R^d_{>0}
```

The implementation normalizes each diagonal metric so its mean is one. This keeps the metric from solving the loss by globally shrinking or inflating all dimensions.

## Flow Matching Objective

For each suffix latent sequence, sample Gaussian noise:

```text
x_0 ~ N(0, sigma^2 I)
```

Interpolate between noise and target:

```text
z_t = (1 - t) x_0 + t x_1
```

The ground-truth flow velocity is:

```text
u_t = x_1 - x_0
```

The Riemannian flow-matching loss is:

```text
L_metric = E[ mean_d( g_phi(z_t, t, c, p) * (v_theta(z_t, t, c, p) - u_t)^2 ) ]
```

The Euclidean loss is still logged for comparison:

```text
L_euclid = E[ ||v_theta(z_t, t, c, p) - u_t||^2 ]
```

but current training sets:

```text
METRIC_LOSS_WEIGHT = 1.0
EUCLIDEAN_LOSS_WEIGHT = 0.0
```

so the actual flow objective is metric-weighted.

## Auxiliary Losses

The endpoint reconstruction loss predicts the clean suffix latent from the current interpolation point:

```text
hat{x}_1 = z_t + (1 - t) v_theta(z_t, t, c, p)
L_x0 = ||hat{x}_1 - x_1||^2
```

Despite the variable name `x0_loss` in the script, this is an endpoint/clean-latent reconstruction term.

A small decoder cross-entropy anchor is also used:

```text
L_decode = CE( decoder([prompt_latents, hat{suffix_latents}]), target_tokens )
```

This loss is intentionally weak. It anchors generated latents to the frozen decoder manifold without allowing decoder CE to dominate the geometry.

The full objective is:

```text
L = lambda_m L_metric
  + lambda_e L_euclid
  + lambda_x L_x0
  + lambda_d L_decode
  + lambda_g L_gate
  + lambda_r L_metric_reg
```

Current defaults:

```text
lambda_m = 1.0
lambda_e = 0.0
lambda_x = 1.0
lambda_d = 0.03
lambda_g = 0.0
lambda_r = 1e-4
```

## Architecture Notes

`FlowNet` is a prompt-conditioned sequence model over suffix latents. Each block combines:

- depthwise temporal convolution,
- self-attention over suffix positions,
- cross-attention from suffix latents to prompt latents,
- small learnable gates for self-attention and cross-attention.

`MetricNet` is a lightweight MLP that predicts a diagonal metric for each latent vector. The metric depends on:

- current latent `z_t`,
- time `t`,
- pooled prompt condition,
- normalized position.

This keeps the Riemannian part simple enough to explain and ablate.

## Generation

At inference time, generation starts from Gaussian suffix noise and integrates the learned velocity field with Heun steps:

```text
z_{t+dt} = z_t + 0.5 * (v_t + v_{t+dt}) * dt
```

Classifier-free guidance is supported by training with prompt dropout and combining conditional/unconditional velocities:

```text
v_guided = v_uncond + s * (v_cond - v_uncond)
```

The generated suffix latents are optionally calibrated to the empirical Stage 1 latent mean and standard deviation before decoding.

## Important Diagnostics

The validation loop prints several metrics:

- `val flow loss`: held-out training objective.
- `real latents mean/std`: distribution of true suffix latents.
- `gen latents mean/std`: distribution of generated suffix latents.
- `metric diag mean/std`: spread of the learned metric.
- `cosine sim`: similarity between real and generated latent means.
- `decoder CE real/gen/gap`: how far generated latents are from the frozen decoder manifold.
- qualitative conditional samples.

The most important diagnostic for text quality is usually:

```text
decoder CE gap = CE(generated latents) - CE(real latents)
```

If real CE is near zero but generated CE is high, the flow is producing latents with reasonable global statistics but poor decoder-manifold alignment.

## What To Report In The Paper

Recommended ablations:

- Euclidean flow matching versus Riemannian metric flow.
- With and without endpoint latent reconstruction.
- With and without decoder CE anchor.
- Different guidance scales.
- Metric regularization strength.
- Metric diagonal spread over training.

Recommended tables:

- validation flow loss,
- decoder CE gap,
- latent mean/std gap,
- qualitative sample quality,
- parameter count and training cost.

Recommended figures:

- training curves for `metric_loss`, `euclidean_loss`, and `decoder CE gap`,
- histogram of learned metric diagonal values,
- qualitative prompt/target/oracle/generated examples,
- schematic of frozen Stage 1 and trainable Stage 2.

## Draft Abstract

We propose a prompt-conditioned Riemannian flow-matching model for text continuation in a frozen language latent space. Rather than generating tokens directly, we first learn a latent autoencoder over BERT token sequences and freeze its encoder-decoder. We then train a conditional flow to transform Gaussian suffix noise into continuation latents conditioned on prompt latents. Because the frozen decoder induces a non-uniform geometry over latent space, we learn a positive diagonal metric and use it to weight the flow-matching loss. The resulting model keeps the simplicity of flow matching while allowing different latent dimensions to contribute according to a learned geometry. We evaluate the method using latent distribution statistics, decoder cross-entropy gap, and qualitative continuation samples, and compare against a flat Euclidean flow baseline.

## Draft Method Paragraph

Given a token sequence, a frozen encoder-decoder maps it into a latent sequence `z`. We split the sequence into a prompt prefix `c` and target suffix `x_1`. During Stage 2 training, we sample Gaussian noise `x_0` and construct linear interpolants `z_t = (1 - t)x_0 + tx_1`. The flow network predicts the velocity from `z_t` to `x_1`, conditioned on the prompt and suffix position. In parallel, a metric network predicts a positive diagonal metric `g_phi(z_t, t, c, p)`, normalized to unit mean per latent vector. The primary loss is a metric-weighted flow-matching objective, with auxiliary clean-endpoint reconstruction and a weak decoder cross-entropy anchor. At generation time, we integrate the learned velocity field from noise to data using Heun integration and decode the resulting latent suffix with the frozen decoder.

## Current Entry Points

Train Stage 2:

```bash
python train_stage2_conditional.py
```

Run conditional inference:

```bash
python inference_stage2_conditional.py
```

Expected checkpoints:

```text
stage1_best.pt
stage2_conditional_best.pt
```

