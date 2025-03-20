from ray.rllib.algorithms.marwil.torch.marwil_torch_learner import MARWILTorchLearner
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance

torch, nn = try_import_torch()


class BCIRLPPOTorchLearner(MARWILTorchLearner):
    def configure_optimizers_for_module(self, module_id, config=None):
        # TODO (simon): Create a constant for Reward Module.
        if module_id == "reward_module":
            # Receive the module.
            module = self._module[module_id]

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
        # TODO (simon): Use here mainly the optimizer for PPO. Maybe derive
        # from both classes.
        # else:
        #   return super().configure_optimizers_for_module(module_id, config)

    def compute_loss_for_module(self, *, module_id, config=None, batch, fwd_out):
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
            curr_action_dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
        )

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = torch.mean(curr_entropy)

        surrogate_loss = torch.min(
            batch[Columns.ADVANTAGES] * logp_ratio,
            batch[Columns.ADVANTAGES]
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
            value_fn_out = mean_vf_unclipped_loss = vf_loss_clipped = mean_vf_loss = z

        total_loss = torch.mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss_clipped
            - (
                self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
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
