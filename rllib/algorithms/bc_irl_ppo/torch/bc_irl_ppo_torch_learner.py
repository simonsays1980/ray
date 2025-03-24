import contextlib
from typing import Any, Dict, Tuple
from ray.rllib.algorithms.bc_irl_ppo.bc_irl_ppo_learner import BCIRLPPOLearner
from ray.rllib.algorithms.marwil.torch.marwil_torch_learner import MARWILTorchLearner
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from ray.rllib.core import ALL_MODULES
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import (
    Learner,
    ENTROPY_KEY,
    POLICY_LOSS_KEY,
    VF_LOSS_KEY,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets
from ray.rllib.utils.postprocessing.zero_padding import (
    split_and_zero_pad_n_episodes,
    unpad_data_if_necessary,
)
from ray.rllib.utils.typing import ModuleID, ParamDict, TensorType

torch, nn = try_import_torch()


class ForwardTrainWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        # Call the module's forward_train method.
        return self.module.forward_train(*args, **kwargs)


class BCIRLPPOTorchLearner(MARWILTorchLearner, BCIRLPPOLearner):
    def configure_optimizers_for_module(self, module_id, config=None):
        # Receive the module.
        module = self._module[module_id]
        # TODO (simon): Create a constant for Reward Module.
        if module_id == "reward_module":
            # Define the optimizer for the reward model.
            params_reward = self.get_parameters(
                module.rf_encoder
            ) + self.get_parameters(module.rf)
            optim_reward = torch.optim.Adam(params_reward, eps=1e-7)

            self.register_optimizer(
                module_id=module_id,
                optimizer_name="rf",
                optimizer=optim_reward,
                params=params_reward,
                lr_or_lr_schedule=config.reward_lr,
            )
        else:
            params = self.get_parameters(module)
            higher_optimizer = torch.optim.Adam(params, eps=1e-7)

            self.register_optimizer(
                module_id=module_id,
                optimizer=higher_optimizer,
                params=params,
                lr_or_lr_schedule=config.lr,
            )

        # TODO (simon): Use here mainly the optimizer for PPO. Maybe derive
        # from both classes.
        # else:
        #   return super().configure_optimizers_for_module(module_id, config)

    @override(Learner)
    def _update(
        self, batch: Dict[str, Any], params: Dict[str, TensorType], module_id=ModuleID
    ) -> Tuple[Any, Any, Any]:
        # The first time we call _update after building the learner or
        # adding/removing models, we update with the uncompiled update method.
        # This makes it so that any variables that may be created during the first
        # update step are already there when compiling. More specifically,
        # this avoids errors that occur around using defaultdicts with
        # torch.compile().
        if (
            self._torch_compile_complete_update
            and not self._compiled_update_initialized
        ):
            self._compiled_update_initialized = True
            return self._uncompiled_update(batch, params, module_id)
        else:
            return self._possibly_compiled_update(batch, params, module_id)

    def _uncompiled_update(
        self,
        batch: Dict,
        params: Dict[str, TensorType],
        module_id: ModuleID,
        **kwargs,
    ):
        """Performs a single update given a batch of data."""
        # Activate tensor-mode on our MetricsLogger.
        self.metrics.activate_tensor_mode()

        # TODO (sven): Causes weird cuda error when WandB is used.
        #  Diagnosis thus far:
        #  - All peek values during metrics.reduce are non-tensors.
        #  - However, in impala.py::training_step(), a tensor does arrive after learner
        #    group.update(), so somehow, there is still a race condition
        #    possible (learner, which performs the reduce() and learner thread, which
        #    performs the logging of tensors into metrics logger).
        self._compute_off_policyness(batch)

        if module_id == "reward_model":
            fwd_out = self.module.forward_train(batch)
        else:
            # TODO: Make a functional_call inside the RLModule.
            fwd_out = {
                module_id: torch.func.functional_call(
                    self.module["default_policy"], params, batch["default_policy"]
                )
            }
            fwd_out["default_policy"].update(
                self.module["reward_model"].forward_train(batch["default_policy"])
            )
        loss_per_module = self.compute_losses(fwd_out=fwd_out, batch=batch)
        gradients = self.compute_gradients(loss_per_module, params)

        with contextlib.ExitStack() as stack:
            if self.config.num_learners > 1:
                for mod in self.module.values():
                    # Skip non-torch modules, b/c they may not have the `no_sync` API.
                    if isinstance(mod, torch.nn.Module):
                        stack.enter_context(mod.no_sync())
            # postprocessed_gradients = self.postprocess_gradients(gradients)
            self.apply_gradients(postprocessed_gradients)

        # Deactivate tensor-mode on our MetricsLogger and collect the (tensor)
        # results.
        return fwd_out, loss_per_module, self.metrics.deactivate_tensor_mode()

    @override(Learner)
    def compute_gradients(
        self, loss_per_module: Dict[ModuleID, TensorType], params, **kwargs
    ) -> ParamDict:
        grads = {}
        for module_id in set(loss_per_module.keys()) - {ALL_MODULES}:
            for optim_name, optim in self.get_optimizers_for_module(module_id):
                # Zero out the gradients.
                optim.zero_grad(set_to_none=True)
            if module_id == "reward_model":
                loss_per_module[module_id].backward()
                grads.update(
                    {
                        pid: grads[pid] + p.grad.clone()
                        if pid in grads
                        else p.grad.clone()
                        for pid, p in self.filter_param_dict_for_optimizer(
                            self._params, optim
                        ).items()
                    }
                )
            else:
                grads.update(
                    {
                        pid: grads
                        for pid, grads in zip(
                            params.keys(),
                            torch.autograd.grad(
                                loss_per_module[module_id],
                                list(params.values()),
                                create_graph=True,
                                allow_unused=True,
                            ),
                        )
                    }
                )

        return grads

    def compute_loss_for_module(self, *, module_id, config=None, batch, fwd_out):

        if module_id != "reward_module":
            # return super().compute_loss_for_module(module_id=module_id, config=config, batch=batch, fwd_out=fwd_out)
            module = self.module[module_id].unwrapped()

            action_dist_class_train = module.get_train_action_dist_cls()
            action_dist_class_exploration = module.get_exploration_action_dist_cls()

            curr_action_dist = action_dist_class_train.from_logits(
                fwd_out[Columns.ACTION_DIST_INPUTS]
            )
            # TODO (sven): We should ideally do this in the LearnerConnector (separation of
            #  concerns: Only do things on the EnvRunners that are required for computing
            #  actions, do NOT do anything on the EnvRunners that's only required for a
            #   training update).
            prev_action_dist = action_dist_class_exploration.from_logits(
                batch[Columns.ACTION_DIST_INPUTS]
            )

            logp_ratio = torch.exp(
                curr_action_dist.logp(batch[Columns.ACTIONS])
                - batch[Columns.ACTION_LOGP]
            )

            curr_entropy = curr_action_dist.entropy()
            mean_entropy = torch.mean(curr_entropy)

            advantages = self._compute_general_advantage(batch, fwd_out, module, config)

            surrogate_loss = torch.min(
                advantages * logp_ratio,
                advantages
                * torch.clamp(logp_ratio, 1 - config.clip_param, 1 + config.clip_param),
            )

            # Compute a value function loss.
            if config.use_critic:
                value_fn_out = module.compute_values(
                    batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
                )
                vf_loss = torch.pow(value_fn_out - batch[Columns.VALUE_TARGETS], 2.0)
                vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)
                mean_vf_loss = torch.mean(vf_loss_clipped)
                mean_vf_unclipped_loss = torch.mean(vf_loss)
            # Ignore the value function -> Set all to 0.0.
            else:
                z = torch.tensor(0.0, device=surrogate_loss.device)
                value_fn_out = (
                    mean_vf_unclipped_loss
                ) = vf_loss_clipped = mean_vf_loss = z

            total_loss = torch.mean(
                -surrogate_loss
                + config.vf_loss_coeff * vf_loss_clipped
                - (
                    self.entropy_coeff_schedulers_per_module[
                        module_id
                    ].get_current_value()
                    * curr_entropy
                )
            )

            # Log important loss stats.
            self.metrics.log_dict(
                {
                    POLICY_LOSS_KEY: -torch.mean(surrogate_loss),
                    VF_LOSS_KEY: mean_vf_loss,
                    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                        batch[Columns.VALUE_TARGETS], value_fn_out
                    ),
                    ENTROPY_KEY: mean_entropy,
                },
                key=module_id,
                window=1,  # <- single items (should not be mean/ema-reduced over time).
            )
            # Return the total loss.
            return total_loss
        else:
            # Run only each `reward_update_freq` time an update of the reward model.
            if (
                config.reward_update_freq != -1
                and self._n_updates % config.reward_update_freq == 0
            ):
                super().compute_loss_for_module(
                    module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
                )
                self._n_updates += 1
                return

    def _compute_general_advantage(
        self,
        batch,
        fwd_out,
        module,
        config,
    ):
        with torch.no_grad():
            batch[Columns.VF_PREDS] = module.compute_values(batch)
        episode_lens = torch.unique(batch[Columns.EPS_LENS])
        batch[Columns.VALUE_TARGETS] = self._compute_value_targets(
            values=unpad_data_if_necessary(episode_lens, batch[Columns.VF_PREDS]),
            rewards=unpad_data_if_necessary(episode_lens, fwd_out[Columns.REWARDS]),
            terminateds=unpad_data_if_necessary(
                episode_lens, batch[Columns.TERMINATEDS].float()
            ),
            truncateds=unpad_data_if_necessary(
                episode_lens, batch[Columns.TRUNCATEDS].float()
            ),
            gamma=config.gamma,
            lambda_=config.lambda_,
        )
        # assert module_value_targets.shape[0] == episode_lens.sum()

        module_advantages = batch[Columns.VALUE_TARGETS] - batch[Columns.VF_PREDS]

        module_advantages = (module_advantages - module_advantages.mean()) / max(
            1e-4, module_advantages.std()
        )

        return module_advantages

    def _compute_value_targets(
        self,
        values,
        rewards,
        terminateds,
        truncateds,
        gamma: float,
        lambda_: float,
    ):
        """Computes value function (vf) targets given vf predictions and rewards.

        Note that advantages can then easily be computed via the formula:
        advantages = targets - vf_predictions
        """
        lambda_ = torch.Tensor([lambda_])
        gamma = torch.Tensor([gamma])
        # Force-set all values at terminals (not at truncations!) to 0.0.
        orig_values = flat_values = values * (1.0 - terminateds)

        flat_values = torch.concatenate((flat_values, torch.zeros((1,))), dim=-1)
        intermediates = rewards + gamma * (1.0 - lambda_) * flat_values[1:]
        continues = 1.0 - terminateds

        Rs = []
        last = flat_values[-1]
        for t in reversed(range(intermediates.shape[0])):
            last = intermediates[t] + continues[t] * gamma * lambda_ * last
            Rs.append(last)
            if truncateds[t]:
                last = orig_values[t]

        # Reverse back to correct (time) direction.
        value_targets = torch.concatenate(list(reversed(Rs)))

        return value_targets
