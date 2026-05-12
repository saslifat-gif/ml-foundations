import torch
import torch.nn as nn

from stage2_config import (
    BASE_NOISE_STD,
    CALIBRATE_GENERATED_LATENTS,
    CROSS_GATE_SCALE,
    FLOW_DEPTH,
    FLOW_HIDDEN_DIM,
    FLOW_REFINE_SCALE,
    GATE_INIT,
    GATE_REG_WEIGHT,
    MAX_SEQ_LEN,
    METRIC_HIDDEN_DIM,
    METRIC_LOG_BOUND,
    ODE_STEPS,
    PROMPT_LEN,
    SELF_GATE_SCALE,
    START_MLP_HIDDEN_DIM,
    START_NOISE_STD_FRAC,
    START_TRANSFORMER_HEADS,
    START_TRANSFORMER_HIDDEN_DIM,
    START_TRANSFORMER_LAYERS,
    STRUCTURED_START_ALPHA,
    STRUCTURED_TARGET_START,
    TARGET_LATENT_MEAN,
    TARGET_LATENT_STD,
    TOKEN_RESIDUAL_SCALE,
    TOKEN_SHARED_BLOCK,
    TOKEN_SHARED_BLOCK_SCALE,
    VELOCITY_CLAMP,
)


def clamp_velocity(v):
    if VELOCITY_CLAMP is None or VELOCITY_CLAMP <= 0:
        return v
    return VELOCITY_CLAMP * torch.tanh(v / VELOCITY_CLAMP)


class FlowNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=FLOW_HIDDEN_DIM, depth=FLOW_DEPTH):
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prompt_pos = nn.Parameter(torch.zeros(1, PROMPT_LEN, hidden_dim))
        self.cond_proj = nn.Linear(PROMPT_LEN * hidden_dim, latent_dim)
        self.in_proj = nn.Linear(latent_dim * 2 + 2, hidden_dim)
        self.pos_proj = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "conv": nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=5,
                    padding=2,
                    groups=hidden_dim,
                ),
                "mix": nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ),
                "self_norm": nn.LayerNorm(hidden_dim),
                "self_attn": nn.MultiheadAttention(
                    hidden_dim,
                    num_heads=8,
                    batch_first=True,
                ),
                "cross_norm": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(
                    hidden_dim,
                    num_heads=8,
                    batch_first=True,
                ),
            })
            for _ in range(depth)
        ])
        self.self_gates = nn.ParameterList([nn.Parameter(torch.tensor(GATE_INIT)) for _ in range(depth)])
        self.cross_gates = nn.ParameterList([nn.Parameter(torch.tensor(GATE_INIT)) for _ in range(depth)])
        self.token_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.token_residual_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.token_block[-1].weight)
        nn.init.zeros_(self.token_block[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.token_residual_proj.weight)
        nn.init.zeros_(self.token_residual_proj.bias)
        self._last_token_block_norm = torch.tensor(0.0)
        self._last_token_block_ratio = torch.tensor(0.0)
        self._last_token_hidden_norm = torch.tensor(0.0)
        self._last_velocity_out_norm = torch.tensor(0.0)
        self._last_out_proj_weight_norm = torch.tensor(0.0)

    def forward(self, z_t, t, z_cond, pos, mask=None, return_hidden=False):
        squeeze = z_t.dim() == 2
        if squeeze:
            z_t = z_t.unsqueeze(1)
            t = t.unsqueeze(1)
            if z_cond.dim() == 2:
                z_cond = z_cond.unsqueeze(1)
            pos = pos.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        prompt_h = None
        prompt_key_padding_mask = None
        if z_cond.dim() == 3 and z_cond.size(1) == PROMPT_LEN:
            prompt_h = self.prompt_proj(z_cond) + self.prompt_pos
            cond = self.cond_proj(prompt_h.reshape(prompt_h.size(0), -1))
            prompt_key_padding_mask = z_cond.abs().sum(dim=-1) == 0
            if prompt_key_padding_mask.all(dim=1).any():
                prompt_key_padding_mask = prompt_key_padding_mask.clone()
                prompt_key_padding_mask[prompt_key_padding_mask.all(dim=1), 0] = False
        elif z_cond.dim() == 3:
            cond = z_cond.mean(dim=1)
        else:
            cond = z_cond
        cond = cond.unsqueeze(1).expand(-1, z_t.size(1), -1)

        inp = torch.cat([z_t, cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        h = self.in_proj(inp) + self.pos_proj(pos.unsqueeze(-1))
        if mask is not None:
            h = h * mask.to(h.dtype).unsqueeze(-1)
            self_key_padding_mask = mask == 0
            if self_key_padding_mask.all(dim=1).any():
                self_key_padding_mask = self_key_padding_mask.clone()
                self_key_padding_mask[self_key_padding_mask.all(dim=1), 0] = False
        else:
            self_key_padding_mask = None

        for block_idx, block in enumerate(self.blocks):
            residual = h
            x = block["norm"](h)
            x = block["conv"](x.transpose(1, 2)).transpose(1, 2)
            x = block["mix"](x)
            h = residual + x
            self_in = block["self_norm"](h)
            self_out, _ = block["self_attn"](
                self_in,
                self_in,
                self_in,
                key_padding_mask=self_key_padding_mask,
                need_weights=False,
            )
            h = h + SELF_GATE_SCALE * self.self_gates[block_idx].tanh() * self_out
            if prompt_h is not None:
                cross_in = block["cross_norm"](h)
                cross_out, _ = block["cross_attn"](
                    cross_in,
                    prompt_h,
                    prompt_h,
                    key_padding_mask=prompt_key_padding_mask,
                    need_weights=False,
                )
                h = h + CROSS_GATE_SCALE * self.cross_gates[block_idx].tanh() * cross_out
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)

        h_before_token = h
        if TOKEN_SHARED_BLOCK:
            token_delta = TOKEN_SHARED_BLOCK_SCALE * self.token_block(h_before_token)
            self._last_token_block_norm = token_delta.detach().norm(dim=-1).mean()
            self._last_token_hidden_norm = h_before_token.detach().norm(dim=-1).mean()
            self._last_token_block_ratio = self._last_token_block_norm / self._last_token_hidden_norm.clamp_min(1e-6)
            h = h + token_delta
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)
        else:
            self._last_token_block_norm = h_before_token.detach().new_tensor(0.0)
            self._last_token_hidden_norm = h_before_token.detach().norm(dim=-1).mean()
            self._last_token_block_ratio = h_before_token.detach().new_tensor(0.0)

        hidden = self.out_norm(h)
        out = self.out_proj(hidden)
        if TOKEN_RESIDUAL_SCALE > 0:
            out = out + TOKEN_RESIDUAL_SCALE * self.token_residual_proj(hidden)
        out = clamp_velocity(out)
        self._last_velocity_out_norm = out.detach().norm(dim=-1).mean()
        self._last_out_proj_weight_norm = self.out_proj.weight.detach().norm()
        if squeeze:
            out = out.squeeze(1)
            hidden = hidden.squeeze(1)
        if return_hidden:
            return out, hidden
        return out


class AuxTokenHead(nn.Module):
    def __init__(self, hidden_dim=FLOW_HIDDEN_DIM, vocab_size=30522):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, hidden):
        return self.net(hidden)


class StartMLP(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=START_MLP_HIDDEN_DIM):
        super().__init__()
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_prompt, pos, mask=None):
        prompt_summary = z_prompt.mean(dim=1)
        prompt_summary = prompt_summary.unsqueeze(1).expand(-1, pos.size(1), -1)
        pos_h = self.pos_proj(pos.unsqueeze(-1))
        z_start = self.net(torch.cat([prompt_summary, pos_h], dim=-1))
        if mask is not None:
            z_start = z_start * mask.to(z_start.dtype).unsqueeze(-1)
        return z_start


class _StartTransformerLayer(nn.Module):
    def __init__(self, latent_dim, num_heads, ffn_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, latent_dim),
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)

    def forward(self, x, z_prompt):
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(x + x2)
        x2, _ = self.cross_attn(x, z_prompt, z_prompt)
        x = self.norm2(x + x2)
        x = self.norm3(x + self.ffn(x))
        return x


class StartTransformer(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        num_layers=START_TRANSFORMER_LAYERS,
        num_heads=START_TRANSFORMER_HEADS,
        ffn_dim=START_TRANSFORMER_HIDDEN_DIM,
    ):
        super().__init__()
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_prompt, pos, mask=None):
        x = self.pos_proj(pos.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x, z_prompt)
        x = self.out_proj(self.out_norm(x))
        if mask is not None:
            x = x * mask.to(x.dtype).unsqueeze(-1)
        return x


class DenoisingPrior(nn.Module):
    """
    x0-prediction denoising prior.
    Input : z_t (noisy suffix latents), z_prompt, alpha (noise level), pos
    Output: pred_z_real  (residual on z_t, zero-init)
    """

    def __init__(self, latent_dim=256, hidden_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.alpha_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.layers = nn.ModuleList([
            _StartTransformerLayer(latent_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t, z_prompt, alpha, pos, mask=None):
        # alpha: [B] — one scalar noise level per sample
        alpha_emb = self.alpha_embed(alpha.unsqueeze(-1)).unsqueeze(1).expand(-1, z_t.size(1), -1)
        pos_emb = self.pos_proj(pos.unsqueeze(-1))
        x = z_t + alpha_emb + pos_emb
        for layer in self.layers:
            x = layer(x, z_prompt)
        pred = z_t + self.out_proj(self.out_norm(x))  # residual: at init pred == z_t
        if mask is not None:
            pred = pred * mask.to(pred.dtype).unsqueeze(-1)
        return pred


class DenoisingPriorSampler(nn.Module):
    """
    Wraps DenoisingPrior to expose the start_mlp(z_prompt, pos, mask) interface.

    use_oracle=True  (stage2 training with oracle z_t):
        set_oracle_target(z_real) before forward() →
        z_t = alpha * z_real + beta * noise  (matches prior oracle training distribution)
    use_oracle=False (inference-matched training / actual inference):
        set_oracle_target() is a no-op; forward() runs the chain:
        z = noise → prior(chain_alphas[0]) → … → prior(chain_alphas[-1])
    """

    def __init__(self, prior, latent_dim=256, alpha=0.5,
                 chain_alphas=None, use_oracle=True):
        super().__init__()
        self.prior = prior
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.chain_alphas = chain_alphas if chain_alphas is not None else [0.3, 0.5, 0.7]
        self.use_oracle = use_oracle
        self._oracle_target = None

    def set_oracle_target(self, z_target):
        """Set before forward() to use oracle z_t.
        No-op when use_oracle=False — sampler always runs chain mode."""
        if self.use_oracle:
            self._oracle_target = z_target

    def forward(self, z_prompt, pos, mask=None):
        B, T = z_prompt.size(0), pos.size(1)

        if self._oracle_target is not None:
            # Oracle mode: single denoising step from alpha-noised real latent
            alpha_val = self.alpha
            beta = (1.0 - alpha_val ** 2) ** 0.5
            z_t = alpha_val * self._oracle_target[:B].detach() + beta * torch.randn(
                B, T, self.latent_dim, device=z_prompt.device, dtype=z_prompt.dtype
            )
            self._oracle_target = None
            if mask is not None:
                z_t = z_t * mask.to(z_t.dtype).unsqueeze(-1)
            alpha_t = z_prompt.new_full((B,), alpha_val)
            return self.prior(z_t, z_prompt, alpha_t, pos, mask)
        else:
            # Chain mode: pure noise → prior(chain_alphas[0]) → … → prior(chain_alphas[-1])
            z = TARGET_LATENT_STD * torch.randn(
                B, T, self.latent_dim, device=z_prompt.device, dtype=z_prompt.dtype
            ) + TARGET_LATENT_MEAN
            if mask is not None:
                z = z * mask.to(z.dtype).unsqueeze(-1)
            for alpha_val in self.chain_alphas:
                alpha_t = z_prompt.new_full((B,), alpha_val)
                z = self.prior(z, z_prompt, alpha_t, pos, mask)
            return z


class DraftPriorSampler(DenoisingPriorSampler):
    """
    Draft-conditioned prior sampler.

    Training/inference path:
        set_draft_target(z_draft) before forward() →
        z_t = alpha * z_draft + beta * noise → prior(z_t, z_prompt, alpha, pos)

    It keeps DenoisingPriorSampler's fallback behavior so old call sites remain safe,
    but stage2 should provide draft targets when DRAFT_PRIOR is enabled.
    """

    def __init__(self, prior, latent_dim=256, alpha=0.7):
        super().__init__(prior, latent_dim=latent_dim, alpha=alpha, use_oracle=False)
        self._draft_target = None

    def set_draft_target(self, z_draft):
        self._draft_target = z_draft

    def forward(self, z_prompt, pos, mask=None):
        if self._draft_target is None:
            return super().forward(z_prompt, pos, mask)

        B, T = z_prompt.size(0), pos.size(1)
        z_draft = self._draft_target[:B].detach()
        self._draft_target = None
        alpha_val = self.alpha
        beta = (1.0 - alpha_val ** 2) ** 0.5
        z_t = alpha_val * z_draft + beta * (
            torch.randn(B, T, self.latent_dim, device=z_prompt.device, dtype=z_prompt.dtype)
            * TARGET_LATENT_STD
            + TARGET_LATENT_MEAN
        )
        if mask is not None:
            z_t = z_t * mask.to(z_t.dtype).unsqueeze(-1)
        alpha_t = z_prompt.new_full((B,), alpha_val)
        return self.prior(z_t, z_prompt, alpha_t, pos, mask)


def flow_token_block_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)

    def scalar(name):
        value = getattr(model, name, None)
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return value.detach().float().item()
        return float(value)

    return {
        "token_block_norm": scalar("_last_token_block_norm"),
        "token_block_ratio": scalar("_last_token_block_ratio"),
        "token_hidden_norm": scalar("_last_token_hidden_norm"),
        "velocity_out_norm": scalar("_last_velocity_out_norm"),
        "out_proj_weight_norm": scalar("_last_out_proj_weight_norm"),
    }


class MetricNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=METRIC_HIDDEN_DIM, log_bound=METRIC_LOG_BOUND):
        super().__init__()
        self.log_bound = log_bound
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, t, z_cond, pos):
        inp = torch.cat([z_t, z_cond, t.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
        log_g = self.net(inp)
        log_g = log_g - log_g.mean(dim=-1, keepdim=True)
        log_g = self.log_bound * torch.tanh(log_g / self.log_bound)
        g_diag = torch.exp(log_g)
        return g_diag / g_diag.mean(dim=-1, keepdim=True).clamp_min(1e-6)


class LatentProjector(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        depth=3,
        residual_scale=0.10,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_gen, z_prompt, mask=None):
        prompt = z_prompt.mean(dim=1)
        prompt_h = self.prompt_proj(prompt).unsqueeze(1).expand(-1, z_gen.size(1), -1)
        h = self.in_proj(torch.cat([z_gen, prompt.unsqueeze(1).expand_as(z_gen)], dim=-1))
        h = h + prompt_h
        for block in self.blocks:
            h = h + block(h)
            if mask is not None:
                h = h * mask.to(h.dtype).unsqueeze(-1)
        raw_delta = self.out_proj(self.out_norm(h))
        delta = self.residual_scale * torch.tanh(raw_delta)
        if mask is not None:
            delta = delta * mask.to(delta.dtype).unsqueeze(-1)
        return z_gen + delta, delta


def prompt_condition(z_data, attention_mask, prompt_len=PROMPT_LEN):
    prompt_z = z_data[:, :prompt_len, :]
    prompt_mask = attention_mask[:, :prompt_len].to(prompt_z.dtype).unsqueeze(-1)
    return prompt_z * prompt_mask


def suffix_positions(batch_size, suffix_len, device, dtype=torch.float32):
    pos = torch.arange(PROMPT_LEN, PROMPT_LEN + suffix_len, device=device, dtype=dtype)
    pos = pos / max(MAX_SEQ_LEN - 1, 1)
    return pos.unsqueeze(0).expand(batch_size, suffix_len)


def attention_gate_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    return (
        SELF_GATE_SCALE * torch.stack([gate.detach().tanh().abs() for gate in model.self_gates]).mean().item(),
        CROSS_GATE_SCALE * torch.stack([gate.detach().tanh().abs() for gate in model.cross_gates]).mean().item(),
    )


def attention_gate_regularizer(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    self_reg = torch.stack([gate.tanh().pow(2) for gate in model.self_gates]).mean()
    cross_reg = torch.stack([gate.tanh().pow(2) for gate in model.cross_gates]).mean()
    return GATE_REG_WEIGHT * (self_reg + cross_reg)


def attention_gate_parameters(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)
    return list(model.self_gates.parameters()) + list(model.cross_gates.parameters())


def non_gate_flow_parameters(flow_net):
    gate_param_ids = {id(param) for param in attention_gate_parameters(flow_net)}
    return [param for param in flow_net.parameters() if id(param) not in gate_param_ids]


def attention_gate_grad_stats(flow_net):
    model = getattr(flow_net, "_orig_mod", flow_net)

    def mean_abs_grad(gates):
        grads = [gate.grad.detach().abs().mean() for gate in gates if gate.grad is not None]
        if not grads:
            return 0.0
        return torch.stack(grads).mean().item()

    return mean_abs_grad(model.self_gates), mean_abs_grad(model.cross_gates)


def calibrate_latents(z, mask=None, target_mean=TARGET_LATENT_MEAN, target_std=TARGET_LATENT_STD, eps=1e-6):
    if not CALIBRATE_GENERATED_LATENTS:
        return z
    if mask is not None:
        valid = mask.bool()
        if valid.any():
            mean = z[valid].mean()
            std = z[valid].std().clamp_min(eps)
        else:
            mean = z.mean()
            std = z.std().clamp_min(eps)
    else:
        mean = z.mean()
        std = z.std().clamp_min(eps)
    return (z - mean) * (target_std / std) + target_mean


def natural_velocity(flow_net, metric_net, z, t, z_cond, pos):
    v = flow_net(z, t, z_cond, pos)
    pooled_cond = z_cond.mean(dim=1).unsqueeze(1).expand_as(z)
    g = metric_net(
        z.reshape(-1, z.size(-1)),
        t.reshape(-1),
        pooled_cond.reshape(-1, z.size(-1)),
        pos.reshape(-1),
    ).reshape_as(z)
    return clamp_velocity(v / g.clamp_min(1e-3)), g


def structured_target_start(z_target, mask=None, alpha=STRUCTURED_START_ALPHA):
    noise = torch.randn_like(z_target) * TARGET_LATENT_STD + TARGET_LATENT_MEAN
    beta = max(0.0, 1.0 - alpha * alpha) ** 0.5
    z = alpha * z_target + beta * noise
    if mask is not None:
        z = z * mask.to(z.dtype).unsqueeze(-1)
    return z


def generate_suffix(
    flow_net,
    metric_net,
    z_cond,
    batch_size,
    suffix_len,
    latent_dim,
    device,
    steps=ODE_STEPS,
    mask=None,
    start_mlp=None,
    z_target_start=None,
):
    pos = suffix_positions(batch_size, suffix_len, device)
    if STRUCTURED_TARGET_START and z_target_start is not None:
        z = structured_target_start(z_target_start, mask)
    elif start_mlp is not None:
        z = start_mlp(z_cond, pos, mask)
        z = z + (START_NOISE_STD_FRAC * TARGET_LATENT_STD) * torch.randn_like(z)
    else:
        z = torch.randn(batch_size, suffix_len, latent_dim, device=device) * BASE_NOISE_STD
    if mask is not None:
        z = z * mask.to(z.dtype).unsqueeze(-1)
    z_initial = z.clone()
    dt = 1.0 / steps
    metric_snapshot = None
    for i in range(steps):
        t = torch.full((batch_size, suffix_len), i / steps, device=device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            v, metric_snapshot = natural_velocity(flow_net, metric_net, z, t, z_cond, pos)
        z = z + FLOW_REFINE_SCALE * v * dt
        if mask is not None:
            z = z * mask.to(z.dtype).unsqueeze(-1)
    z_uncalibrated = z.clone()
    z = calibrate_latents(z, mask)
    return z, metric_snapshot, z_initial, z_uncalibrated
