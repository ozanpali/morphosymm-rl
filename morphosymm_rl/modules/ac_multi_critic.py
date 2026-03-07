# Multi-Critic Actor-Critic architecture.
#
# Implements the observation-conditioned multi-critic approach described in:
#   "Multi-Critic Actor Learning: Teaching RL Policies to Act with Style"
#   (Mysore et al., AAAI 2022)
#
# Two operating modes are supported (selected via ``multi_critic_mode``):
#
# **"paper"** – faithful MCN-PPO:
#   ``evaluate()`` returns ``[batch, num_critics]`` (all critic heads).
#   The PPO algorithm uses observation-based masking for GAE and value loss,
#   exactly as in the reference code.  Requires the companion PPO changes.
#
# **"routed"** – simplified routing:
#   ``evaluate()`` returns ``[batch, 1]`` (scalar selected by one-hot signal).
#   No changes to PPO/storage – drop-in replacement for a single-critic.
#
# The actor is a Mixture-of-Experts (MoE) network (reusing MoE_net from ac_moe)
# and the critic consists of *num_critics* independent MLP heads.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation
from typing import Any, NoReturn
from tensordict import TensorDict

from .ac_moe import MLP_net, MoE_net


class MultiMLPCritic(nn.Module):
    """Bank of *num_critics* independent MLP value heads.

    Each critic is a standard ``MLP_net`` with scalar output.

    Args:
        obs_dim: Dimensionality of the critic observation.
        num_critics: Number of independent critic heads.
        hidden_dims: Hidden-layer sizes shared across all heads.
        activation: Activation function name (resolved via ``resolve_nn_activation``).
    """

    def __init__(
        self,
        obs_dim: int,
        num_critics: int,
        hidden_dims: list[int],
        activation: str = "elu",
    ):
        super().__init__()
        self.num_critics = num_critics
        act = resolve_nn_activation(activation)
        self.critics = nn.ModuleList(
            [MLP_net(obs_dim, hidden_dims, 1, act) for _ in range(num_critics)]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the concatenated values of all critics.

        Returns:
            Tensor of shape ``[batch, num_critics]``.
        """
        values = [c(obs) for c in self.critics]  # list of [batch, 1]
        return torch.cat(values, dim=-1)  # [batch, num_critics]

    def forward_routed(self, obs: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """Return a single scalar value per sample using the routing signal.

        Args:
            obs: Critic observations ``[batch, obs_dim]``.
            routing: One-hot (or soft) routing signal ``[batch, num_critics]``.

        Returns:
            Value estimate ``[batch, 1]``.
        """
        all_values = self.forward(obs)  # [batch, num_critics]
        # Weighted combination (reduces to hard selection when routing is one-hot)
        value = (all_values * routing).sum(dim=-1, keepdim=True)  # [batch, 1]
        return value


class ActorCriticMultiCritic(nn.Module):
    """Actor-Critic with MoE actor and observation-conditioned Multi-Critic.

    The last ``num_critics`` elements of the *actor* observation are treated as
    the one-hot routing / masking signal that identifies the active critic for
    each sample.

    Two modes are supported via ``multi_critic_mode``:

    **"paper"** (default) – faithful MCN-PPO implementation:
        * ``evaluate()`` returns ``[batch, num_critics]`` – all critic heads.
        * The PPO algorithm is responsible for masking the per-critic TD errors
          and value-loss terms using the routing signal, exactly as described in
          the reference MCN-PPO code::

              delta = ((rew + gamma * V_next - V) * routing).sum(-1)
              value_loss = MSE(routing * V, routing * ret)

    **"routed"** – simplified scalar routing:
        * ``evaluate()`` returns ``[batch, 1]`` – routing-weighted scalar.
        * Standard PPO – no changes to storage or GAE needed.

    The actor is either a ``MoE_net`` (Mixture-of-Experts) or a plain ``MLP_net``
    depending on the ``use_moe_actor`` flag (default ``True``).
    MoE configuration is supplied via ``**moe_cfg``.

    Example routing convention (from the locomotion env):
        [1, 0, 0] → critic 0 – all legs fine
        [0, 1, 0] → critic 1 – three legs working (one failed)
        [0, 0, 1] → critic 2 – two legs working (two failed)
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        # Multi-critic parameters (from multicritic_cfg)
        num_critics: int = 3,
        log_critic_stats: bool = False,
        multi_critic_mode: str = "paper",
        use_moe_actor: bool = True,
        # MoE parameters (from moe_cfg, passed via **kwargs)
        **moe_cfg: dict[str, Any],
    ):
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256, 256]

        # ---- operating mode ----
        assert multi_critic_mode in ("paper", "routed"), (
            f"multi_critic_mode must be 'paper' or 'routed', got '{multi_critic_mode}'"
        )
        self.multi_critic_mode = multi_critic_mode

        # ---- observation dimensions ----
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticMultiCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticMultiCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # ---- multi-critic config ----
        self.num_critics = num_critics
        self.log_critic_stats = log_critic_stats

        # ---- actor ----
        self.use_moe_actor = use_moe_actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            print("State-dependent std not supported yet, switching to off.")
            self.state_dependent_std = False

        act_fn = resolve_nn_activation(activation)

        if self.use_moe_actor:
            # Parse MoE kwargs
            num_experts = moe_cfg.get("num_experts", 4)
            raw_top_k = moe_cfg.get("top_k", -1)
            top_k = -1 if raw_top_k is None else int(raw_top_k)
            use_gate_loss = moe_cfg.get("use_gate_loss", False)
            use_explicit_expert = moe_cfg.get("use_explicit_expert", False)
            explicit_expert_epsilon = moe_cfg.get("explicit_expert_epsilon", 0.8)
            gate_hidden_dims = moe_cfg.get("gate_hidden_dims", None)
            jitter_noise = moe_cfg.get("jitter_noise", 0.0)
            self.use_load_balance_loss = moe_cfg.get("use_load_balance_loss", False)
            self.log_expert_stats = moe_cfg.get("log_expert_stats", False)
            use_shared_backbone = moe_cfg.get("use_shared_backbone", False)
            log_gate_distribution = moe_cfg.get("log_gate_distribution", False)
            use_shared_gate = moe_cfg.get("use_shared_gate", False)

            # Optional shared gate between actor MoE experts
            shared_gate = None
            if use_shared_gate:
                gate_layers: list[nn.Module] = []
                last_dim = num_actor_obs
                for h in (gate_hidden_dims or []):
                    gate_layers += [nn.Linear(last_dim, h), act_fn]
                    last_dim = h
                gate_layers.append(nn.Linear(last_dim, num_experts))
                shared_gate = nn.Sequential(*gate_layers)

            self.actor = MoE_net(
                obs_dim=num_actor_obs,
                act_dim=num_actions,
                hidden_dims=actor_hidden_dims,
                gate_hidden_dims=gate_hidden_dims,
                activation=activation,
                num_experts=num_experts,
                top_k=top_k,
                use_gate_loss=use_gate_loss,
                use_explicit_expert=use_explicit_expert,
                explicit_expert_epsilon=explicit_expert_epsilon,
                jitter_noise=jitter_noise,
                use_shared_backbone=use_shared_backbone,
                log_gate_distribution=log_gate_distribution,
                gate=shared_gate,
            )
        else:
            # Plain MLP actor
            self.use_load_balance_loss = False
            self.log_expert_stats = False
            self.actor = MLP_net(num_actor_obs, actor_hidden_dims, num_actions, act_fn)

        # ---- actor observation normalization ----
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            from rsl_rl.modules.normalizer import EmpiricalNormalization
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # ---- critic (multi-head) ----
        self.critic = MultiMLPCritic(
            obs_dim=num_critic_obs,
            num_critics=num_critics,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        # ---- critic observation normalization ----
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            from rsl_rl.modules.normalizer import EmpiricalNormalization
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # ---- action noise ----
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Populated in _update_distribution
        self.distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

        # ---- critic routing bookkeeping (for logging) ----
        self._last_routing: torch.Tensor = torch.empty(0)

        actor_type = "MoE" if self.use_moe_actor else "MLP"
        print(f"Actor ({actor_type}) structure:\n{self.actor}")
        print(f"Critic (MultiCritic, {num_critics} heads, mode={multi_critic_mode}) structure:\n{self.critic}")

    # ------------------------------------------------------------------
    # Standard interface
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # ------------------------------------------------------------------
    # MoE actor helpers (only usable when use_moe_actor=True)
    # ------------------------------------------------------------------

    def gate_entropy(self) -> torch.Tensor:
        """Mean gate entropy from the last forward pass (MoE actor only)."""
        if not self.use_moe_actor:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        w = self.actor._last_gate_weights.squeeze(1)  # [batch, K]
        mean_w = w.mean(dim=0)  # [K]
        return -(mean_w * torch.log(mean_w + 1e-8)).sum()

    def load_balance_loss(self) -> torch.Tensor:
        """Aggregate load-balancing loss from the actor MoE (MoE actor only)."""
        if not self.use_moe_actor:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.actor.load_balance_loss()

    def get_expert_stats(self) -> dict[str, float]:
        """Per-expert utilization stats for logging (prefixed with ``MoE/``). MoE actor only."""
        if not self.use_moe_actor:
            return {}
        raw = self.actor.expert_utilization_stats()
        return {f"MoE/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Multi-critic helpers
    # ------------------------------------------------------------------

    def _extract_routing(self, obs: TensorDict) -> torch.Tensor:
        """Extract the one-hot routing signal from the actor observation.

        The routing signal occupies the **last** ``num_critics`` elements of the
        flattened actor observation.
        """
        actor_obs = self.get_actor_obs(obs)
        routing = actor_obs[:, -self.num_critics:]  # [batch, num_critics]
        return routing

    def get_critic_stats(self) -> dict[str, float]:
        """Per-critic utilization statistics for the last batch.

        Returns a dict with keys prefixed by ``MultiCritic/``.
        """
        stats: dict[str, float] = {}
        if self._last_routing.numel() == 0:
            return stats

        # Compute per-critic utilization (fraction of samples routed to each critic)
        routing = self._last_routing  # [batch, num_critics]
        # Hard assignment: which critic has highest routing weight
        hard_idx = routing.argmax(dim=-1)  # [batch]
        batch_size = routing.shape[0]

        for i in range(self.num_critics):
            count = (hard_idx == i).sum().float()
            stats[f"MultiCritic/critic_{i}_usage_pct"] = (count / batch_size).item()
            stats[f"MultiCritic/critic_{i}_mean_routing_weight"] = routing[:, i].mean().item()

        return stats

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def _update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    # ------------------------------------------------------------------
    # Act / Evaluate
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the value function for the given observations.

        Behaviour depends on ``multi_critic_mode``:

        **"paper"**:
            Returns ``[batch, num_critics]`` – all critic heads. The PPO
            algorithm must handle masking/aggregation using the routing signal.

        **"routed"**:
            Returns ``[batch, 1]`` – routing-weighted scalar value. Standard
            PPO can be used without modification.
        """
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        routing = self._extract_routing(obs)  # [batch, num_critics]

        # Cache for logging
        self._last_routing = routing.detach()

        if self.multi_critic_mode == "paper":
            # Return all critic values; PPO will handle masking
            return self.critic(critic_obs)  # [batch, num_critics]
        else:
            # "routed" mode – return weighted scalar
            return self.critic.forward_routed(critic_obs, routing)  # [batch, 1]

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model parameters.

        Returns:
            Whether this training resumes a previous run (used by the runner
            to decide how to load further parameters).
        """
        super().load_state_dict(state_dict, strict=strict)
