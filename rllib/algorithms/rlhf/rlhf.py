from typing import Optional

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    OFFLINE_SAMPLING_TIMER,
    TIMERS,
    ALL_MODULES,
)

torch, nn = try_import_torch()


class RLHFConfig(PPOConfig):
    def __init__(self, algo_class=None):

        super().__init__(algo_class=algo_class or PPO)
        # The SFT model ID on huggingface.
        self.sft_model_id: Optional[str] = None

    def training(
        self,
        *,
        sft_model_id: Optional[str] = NotProvided,
        **kwargs,
    ) -> "RLHFConfig":

        super().training(**kwargs)

        if sft_model_id is not NotProvided:
            self.sft_model_id = sft_model_id

        return self


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
            output = self.learner_group._learner.module.generate(batch)

            # Note, `generate` returns only responses.
            batch["query_responses"] = torch.cat(
                (batch["input_ids"], output[Columns.ACTIONS])
            )

            # TODO (simon): Check, if this indeed needed or can be done inside the loss.
            action_dist_class = (
                self.learner_group._learner.module.get_exploration_action_dist_cls()
            )
            action_dist = action_dist_class.from_logits(
                output[Columns.ACTION_DIST_INPUTS]
            )
            batch[Columns.ACTION_LOGP] = action_dist.logp(output[Columns.ACTIONS])

            # Reference policy.
            reference_output = self.learner_group._learner.module.reference_forward(
                batch,
            )
            reference_action_dist = action_dist_class.from_logits(
                reference_output[Columns.ACTION_DIST_INPUTS]
            )
            batch["reference_action_logp"] = action_dist.logp(batch[Columns.ACTIONS])

            # Postprocess tokens for reward computation.
