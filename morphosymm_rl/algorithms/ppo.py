# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable

# Optional import for type checking
from morphosymm_rl.modules.ac_multi_critic import ActorCriticMultiCritic


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # Resolve the data augmentation function (supports string names or direct callables)
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Create the optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # ---- Multi-Critic "paper" mode detection ----
        # When the policy is ActorCriticMultiCritic in "paper" mode, evaluate()
        # returns [batch, num_critics] instead of [batch, 1].  We handle the
        # conversion to a storage-compatible scalar inside act/compute_returns/update.
        self._is_mc_paper = (
            isinstance(policy, ActorCriticMultiCritic)
            and getattr(policy, "multi_critic_mode", "routed") == "paper"
        )
        if self._is_mc_paper:
            nc = policy.num_critics
            T = storage.num_transitions_per_env
            N = storage.num_envs
            # Parallel buffer for full multi-critic values [T, N, num_critics]
            self._mc_values = torch.zeros(T, N, nc, device=device)
            # Parallel buffer for the routing one-hot [T, N, num_critics]
            self._mc_routing = torch.zeros(T, N, nc, device=device)
            self._mc_step = 0
            print(f"[PPO] Multi-Critic paper mode enabled with {nc} critics.")

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()

        if self._is_mc_paper:
            # evaluate() returns [N, num_critics] in paper mode
            mc_values = self.policy.evaluate(obs).detach()  # [N, K]
            routing = self.policy._extract_routing(obs).detach()  # [N, K]
            # Store full multi-critic data in parallel buffers
            self._mc_values[self._mc_step] = mc_values
            self._mc_routing[self._mc_step] = routing
            self._mc_step += 1
            # Compute routing-weighted scalar for standard storage  [N, 1]
            self.transition.values = (mc_values * routing).sum(dim=-1, keepdim=True)
        else:
            self.transition.values = self.policy.evaluate(obs).detach()

        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage

        if self._is_mc_paper:
            # ---- Paper-faithful multi-critic GAE ----
            # Compute last-step multi-critic values
            mc_last_values = self.policy.evaluate(obs).detach()  # [N, K]
            last_routing = self.policy._extract_routing(obs).detach()  # [N, K]

            nc = self.policy.num_critics
            mc_vals = self._mc_values  # [T, N, K]
            mc_rout = self._mc_routing  # [T, N, K]

            # Per-critic returns: discount_cumsum of (scalar reward broadcast to K critics)
            # Since all critics share the same reward, we compute per-critic returns
            # and advantages using the approach from Mysore et al.:
            #   delta = ((r + gamma * V_next - V) * routing).sum(-1)   → scalar
            #   advantage = discount_cumsum(delta, gamma * lam)
            #   per-critic return = discount_cumsum(reward, gamma)  (same for all critics)

            mc_advantage = torch.zeros(st.num_envs, nc, device=self.device)
            advantage_scalar = torch.zeros(st.num_envs, 1, device=self.device)

            for step in reversed(range(st.num_transitions_per_env)):
                if step == st.num_transitions_per_env - 1:
                    next_mc_vals = mc_last_values
                    next_routing = last_routing
                else:
                    next_mc_vals = mc_vals[step + 1]
                    next_routing = mc_rout[step + 1]

                next_is_not_terminal = 1.0 - st.dones[step].float()  # [N, 1]

                # Per-critic TD error, masked by routing, summed to scalar
                # Exactly: delta = ((r + gamma * V_next - V) * routing).sum(-1)
                rewards_expanded = st.rewards[step].expand(-1, nc)  # [N, K]
                per_critic_delta = rewards_expanded + next_is_not_terminal * self.gamma * next_mc_vals - mc_vals[step]
                # Mask by current step's routing and sum → scalar
                routing_step = mc_rout[step]  # [N, K]
                delta_scalar = (per_critic_delta * routing_step).sum(dim=-1, keepdim=True)  # [N, 1]

                # Standard GAE accumulation (scalar)
                advantage_scalar = delta_scalar + next_is_not_terminal * self.gamma * self.lam * advantage_scalar

                # Return and advantage (stored as scalar in standard storage)
                # Return = advantage + V_active (routing-weighted scalar)
                v_active = (mc_vals[step] * routing_step).sum(dim=-1, keepdim=True)  # [N, 1]
                st.returns[step] = advantage_scalar + v_active

            # Scalar advantages
            st.advantages = st.returns - st.values

            # Normalize advantages
            if not self.normalize_advantage_per_mini_batch:
                st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

            # Reset MC step counter for next rollout
            self._mc_step = 0
        else:
            # ---- Standard single-critic GAE ----
            last_values = self.policy.evaluate(obs).detach()
            advantage = 0
            for step in reversed(range(st.num_transitions_per_env)):
                next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
                next_is_not_terminal = 1.0 - st.dones[step].float()
                delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
                advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
                st.returns[step] = advantage + st.values[step]
            st.advantages = st.returns - st.values
            if not self.normalize_advantage_per_mini_batch:
                st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)

            if self._is_mc_paper:
                # Paper-faithful multi-critic value loss:
                # evaluate() returns [batch, num_critics]; we compute a masked loss
                # so that only the active critic (selected by routing) is trained.
                mc_value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
                # [batch, K]
                routing_batch = self.policy._extract_routing(obs_batch)  # [batch, K]

                # Routing-weighted scalar for surrogate loss compatibility
                value_batch = (mc_value_batch * routing_batch).sum(dim=-1, keepdim=True)  # [batch, 1]

                # Masked per-critic value loss (Mysore et al. Eq. in supplementary):
                #   loss_v = MSE(routing * V_all, routing * ret_expanded)
                nc = self.policy.num_critics
                returns_expanded = returns_batch.expand(-1, nc)  # [batch, K]

                if self.use_clipped_value_loss:
                    target_expanded = target_values_batch.expand(-1, nc)  # [batch, K]
                    mc_value_clipped = target_expanded + (mc_value_batch - target_expanded).clamp(
                        -self.clip_param, self.clip_param
                    )
                    mc_losses = (routing_batch * (mc_value_batch - returns_expanded)).pow(2)
                    mc_losses_clipped = (routing_batch * (mc_value_clipped - returns_expanded)).pow(2)
                    value_loss = torch.max(mc_losses, mc_losses_clipped).mean()
                else:
                    value_loss = (routing_batch * (mc_value_batch - returns_expanded)).pow(2).mean()
            else:
                value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])

                # Standard value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # MoE losses
            if hasattr(self.policy.actor, "use_gate_loss") and self.policy.actor.use_gate_loss:
                gate_entropy = self.policy.gate_entropy()
                gate_entropy_coef = 0.0001
                loss -= gate_entropy_coef * gate_entropy

            # Load-balancing auxiliary loss (Fedus et al., 2022)
            if hasattr(self.policy, "use_load_balance_loss") and self.policy.use_load_balance_loss:
                lb_loss = self.policy.load_balance_loss()
                load_balance_coef = 0.0001
                loss += load_balance_coef * lb_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        # MoE expert utilization stats (logged per-expert to detect dead experts)
        if hasattr(self.policy, "log_expert_stats") and self.policy.log_expert_stats:
            loss_dict.update(self.policy.get_expert_stats())

        # Multi-Critic utilization stats
        if hasattr(self.policy, "log_critic_stats") and self.policy.log_critic_stats:
            loss_dict.update(self.policy.get_critic_stats())

        return loss_dict

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
