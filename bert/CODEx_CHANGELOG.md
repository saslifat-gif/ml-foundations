# Codex Change Log

Short log for code and architecture changes made by Codex.

Current protected architecture:

```text
stage1 frozen BERT latent autoencoder
stage2 Riemannian metric flow over latent space
FlowNet + MetricNet
natural velocity / metric-aware sampling
```

## 2026-05-08

- Added this change log.
- Rule going forward: before editing, Codex must state what will be added, deleted, modified, whether it conflicts with the current Riemannian architecture, expected impact, and risk.
- Future code changes should add a short entry here.
- Added stage1 decode debug prints in `parallel_decoder.py` for blank `original` / `predicted` output. No architecture impact; this only exposes token ids/tokens when decoding becomes blank.
- Added validation for `bert-base-uncased` tokenizer/config loading in `parallel_decoder.py`. No architecture impact; prevents poisoned local caches that map normal text to `[UNK]` ids like `[2, 1, 1, ...]`.
- Added Riemannian diagnostics in `train_stage2_conditional.py`: metric regularization, metric min/max, initial/raw/generated latent scale, and decoder CE before/after metric-flow sampling. No architecture impact.
- Added Riemannian rollout loss in `train_stage2_conditional.py`, training through short `flow / metric` ODE paths used by inference. No architecture conflict; increases memory/time modestly.
- Added rollout norm preservation loss and raw-norm validation scoring in `train_stage2_conditional.py` to counter latent norm collapse observed during Riemannian rollout training. No architecture conflict; may trade some CE gains for healthier latent scale.
- Changed `MetricNet` log-diagonal bounding from hard clamp to smooth `tanh` in stage2 training and inference, and reduced rollout norm loss weight after norm stabilization. No architecture conflict; restores metric gradients near the bound and softens radial overconstraint.
- Added exact-identity initialization for fresh `MetricNet` heads and a short metric regularization warmup in `train_stage2_conditional.py` after scratch tanh-bound runs showed rapid early metric anisotropy expansion. No architecture conflict; should slow metric gaming while FlowNet learns.
- Added zero initialization for fresh `FlowNet.out_proj` in `train_stage2_conditional.py` after scratch runs showed large step-0 rollout norm spikes from random vector-field initialization. No architecture conflict; starts fresh ODE rollouts at zero velocity before learning.
- Added a separate higher-LR optimizer group and gradient diagnostics for FlowNet self/cross attention gates in `train_stage2_conditional.py` after gate values stayed nearly static. No architecture conflict; reveals whether gates are gradient-starved or just LR-starved.
- Increased fresh FlowNet attention gate initialization and gate optimizer multiplier in `train_stage2_conditional.py` after diagnostics showed nonzero gate gradients but negligible metric impact. No architecture conflict; tests whether gated attention is underpowered or redundant.
- Lengthened the stage2 metric regularization warmup from 500 to 1000 steps in `train_stage2_conditional.py` after repeated gate-gradient drops appeared near the same warmup multiplier. No architecture conflict; delays metric anisotropy release so gates can train longer.
- Increased direct and rollout decoder CE loss weights in `train_stage2_conditional.py` after warmup/gate tuning improved cosine and val score but left raw decoder CE stuck. No architecture conflict; shifts pressure toward decoder-manifold accuracy.
- Restored direct and rollout decoder CE loss weights to `0.05` and added weighted decoder-loss logging in `train_stage2_conditional.py` after higher CE weights inflated total `rloss`, regressed cosine, and caused token-lock repetition. No architecture conflict.
- Made stage2 WikiText loading prefer the local Hugging Face datasets cache before falling back online in `train_stage2_conditional.py`, avoiding startup stalls from remote `HEAD` request timeouts. No architecture impact.
- Increased stage2 rollout training from 4 to 8 ODE steps and reduced rollout batch from 128 to 64 in `train_stage2_conditional.py` after generated samples stayed repetitive despite improving cosine/CE. No architecture conflict; better matches 16-step inference while containing cost.
- Reverted stage2 rollout training to 4 steps with batch 128 in `train_stage2_conditional.py` after the 8-step / batch-64 trial regressed val score, raw CE, and sample quality. No architecture conflict.
- Added fixed seeding, seeded dataloader shuffling/workers, deterministic cuDNN selection, and deterministic validation sampling in `train_stage2_conditional.py` after identical-config runs showed large metric variance. No architecture impact.
- Added configurable persistent DataLoader workers in `train_stage2_conditional.py` to reduce Python multiprocessing `/tmp/pymp-* Directory not empty` cleanup races after validation. No architecture impact.
- Added validation-time argmax token-collapse diagnostics in `train_stage2_conditional.py` after cosine/raw CE improved while samples stayed repetitive. No architecture impact; reports entropy, unique-token ratio, max-token fraction, and dominant tokens for generated vs oracle decodes.
- Added rollout decoder-hidden manifold loss in `train_stage2_conditional.py` after collapse diagnostics showed generated latents produce high-entropy decoder logits with biased argmax repetition. No architecture conflict; stage1 remains frozen while stage2 is nudged toward decoder-internal hidden states.
- Disabled the rollout hidden-loss probe and added rollout oracle-logit KL distillation in `train_stage2_conditional.py` after hidden matching lowered entropy but worsened token collapse. No architecture conflict; stage2 now gets direct pressure to match frozen decoder token-rank distributions from real latents.
- Disabled rollout oracle-logit KL, tightened the metric log bound, added rollout pairwise latent-diversity loss, and included token-collapse penalties in validation scoring in `train_stage2_conditional.py` after KL improved cosine but worsened argmax collapse. No architecture conflict; stage1 remains frozen and stage2 still trains the Riemannian metric flow.
- Changed rollout diversity loss in `train_stage2_conditional.py` from sequence-mean pairwise matching to capped valid-token pairwise matching, and increased its weight after sequence-level diversity reduced metric expansion but did not fix token-basin collapse. No architecture conflict; decoder remains frozen and stage2 still trains the Riemannian metric flow.
- Added optional decoder adaptation in `train_stage2_conditional.py` after flow-only token-level diversity kept metric geometry tight but left generated argmax collapse above threshold. Architecture intentionally changes for this labeled experiment: encoder and decoder BERT body stay frozen, only decoder `project_up` / `to_logits` train with real/generated CE and frozen-teacher preservation KL, saved separately as `stage2_conditional_decoder_adapt_best.pt`.
- Updated `inference_stage2_conditional.py` to auto-prefer `stage2_conditional_decoder_adapt_best.pt` when present and load an adapted decoder from the stage2 checkpoint. Architecture impact matches the labeled decoder-adaptation experiment; preserves fallback to the flow-only checkpoint.
