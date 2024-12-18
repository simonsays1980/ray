from typing import Optional

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    OFFLINE_SAMPLING_TIMER,
    TIMERS,
)

torch, nn = try_import_torch()


class RLHFConfig(PPOConfig):
    def __init__(self, algo_class=None):

        super().__init__(algo_class=algo_class or RLHF)
        # The SFT model ID on huggingface.
        self.sft_model_id: Optional[str] = None
        self.missing_eos_penalty: float = 1.0
        self.invalid_logprob_value: float = 1.0

    def training(
        self,
        *,
        sft_model_id: Optional[str] = NotProvided,
        missing_eos_penalty: Optional[float] = NotProvided,
        invalid_logprob_value: Optional[float] = NotProvided,
        **kwargs,
    ) -> "RLHFConfig":

        super().training(**kwargs)

        if sft_model_id is not NotProvided:
            self.sft_model_id = sft_model_id
        if missing_eos_penalty is not NotProvided:
            self.missing_eos_penalty = missing_eos_penalty
        if invalid_logprob_value is not NotProvided:
            self.invalid_logprob_value = invalid_logprob_value

        return self

    @override(PPOConfig)
    def get_default_rl_module_spec(self) -> RLModuleSpec:
        from ray.rllib.algorithms.rlhf.rlhf_catalog import RLHFCatalog

        if self.framework_str == "torch":
            from ray.rllib.algorithms.rlhf.torch.rlhf_torch_rl_module import RLHFTorchRLModule

            return RLModuleSpec(module_class=RLHFTorchRLModule, catalog_class=RLHFCatalog)
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use 'torch'."
            )


class RLHF(PPO):
    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return RLHFConfig()

    @override(PPO)
    def training_step(self) -> None:

        with self.metrics.log_time((TIMERS, OFFLINE_SAMPLING_TIMER)):

            # Get a tokenized and padded batch.
            batch = self.offline_data.sample(
                num_samples=self.config.train_batch_size_per_learner,
                return_iterator=False,
                # No multi-learner setup for the RLHF trainer.
                num_shards=1,
            )

            # TODO (simon): Maybe moving this into the learner's
            # `compute_loss_for_module`.
            output = self.learner_group._learner.module["default_policy"].generate(batch)

            # Note, `generate` returns only responses.
            batch[Columns.ACTIONS] = output[Columns.ACTIONS] 
            batch["query_responses"] = torch.cat(
                (batch["input_ids"], batch[Columns.ACTIONS]),
                dim=-1
            )

            # TODO (simon): Check, if this indeed needed or can be done inside the loss.
            action_dist_class = (
                self.learner_group._learner.module["default_policy"].get_exploration_action_dist_cls()
            )
            action_dist = action_dist_class.from_logits(
                output[Columns.ACTION_DIST_INPUTS]
            )
            batch[Columns.ACTION_LOGP] = action_dist.logp(batch[Columns.ACTIONS])

            # Reference policy.
            reference_output = self.learner_group._learner.module["default_policy"].reference_forward(
                batch,
            )
            reference_action_dist = action_dist_class.from_logits(
                reference_output[Columns.ACTION_DIST_INPUTS]
            )
            batch["reference_action_logp"] = reference_action_dist.logp(batch[Columns.ACTIONS])

            # Value model.
            batch[Columns.VF_PREDS] = self.learner_group._learner.module["default_policy"].compute_values(
                batch
            )

            # Postprocess tokens for reward computation. Note, this postprocessing is needed
            # for the reward model.
            postprocessed_batch = self.offline_data.postprocess(batch)

            # Run the reward model.
            batch["scores"] = self.learner_group._learner.module["default_policy"].compute_rewards(batch)
            
            # Postprocess rewards.
            postprocessed_batch = self.offline_data.postprocess_rewards(batch)

        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update_from_batch(
                batch=postprocessed_batch,
                timesteps={
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: batch["input_ids"].shape[0],
                }
            )

            self.metrics.merge_and_log_n_dicts(learner_results, key=LEARNER_RESULTS)
