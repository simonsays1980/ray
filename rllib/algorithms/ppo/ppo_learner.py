from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.connector_context_v2 import ConnectorContextV2
from ray.rllib.core.learner.learner import Learner, LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.evaluation.postprocessing_v2 import compute_advantages_for_episodes
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import EpisodeType


LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY = "vf_loss_unclipped"
LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY = "vf_explained_var"
LEARNER_RESULTS_KL_KEY = "mean_kl_loss"
LEARNER_RESULTS_CURR_KL_COEFF_KEY = "curr_kl_coeff"
LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY = "curr_entropy_coeff"


@dataclass
class PPOLearnerHyperparameters(LearnerHyperparameters):
    """Hyperparameters for the PPOLearner sub-classes (framework specific).

    These should never be set directly by the user. Instead, use the PPOConfig
    class to configure your algorithm.
    See `ray.rllib.algorithms.ppo.ppo::PPOConfig::training()` for more details on the
    individual properties.
    """

    use_kl_loss: bool = None
    kl_coeff: float = None
    kl_target: float = None
    use_critic: bool = None
    clip_param: float = None
    vf_clip_param: float = None
    entropy_coeff: float = None
    entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = None
    vf_loss_coeff: float = None


class PPOLearner(Learner):
    @override(Learner)
    def build(self) -> None:
        super().build()

        # Expand the (user defined) learner connector by PPO's advantage computing
        # connector. Place this connector piece at the end, such that we then
        # already have the ready (module-specific) batch to be used to compute the value
        # function outputs first (then compute advantages from these).
        self._learner_connector.append(
            PPOLearnerConnector(ctx=self._learner_connector_ctx)
        )

        # Dict mapping module IDs to the respective entropy Scheduler instance.
        self.entropy_coeff_schedulers_per_module: Dict[
            ModuleID, Scheduler
        ] = LambdaDefaultDict(
            lambda module_id: Scheduler(
                fixed_value_or_schedule=(
                    self.hps.get_hps_for_module(module_id).entropy_coeff
                ),
                framework=self.framework,
                device=self._device,
            )
        )

        # Set up KL coefficient variables (per module).
        # Note that the KL coeff is not controlled by a Scheduler, but seeks
        # to stay close to a given kl_target value in our implementation of
        # `self.additional_update_for_module()`.
        self.curr_kl_coeffs_per_module: Dict[ModuleID, Scheduler] = LambdaDefaultDict(
            lambda module_id: self._get_tensor_variable(
                self.hps.get_hps_for_module(module_id).kl_coeff
            )
        )

    @override(Learner)
    def remove_module(self, module_id: str):
        super().remove_module(module_id)
        self.curr_kl_coeffs_per_module.pop(module_id)
        self.entropy_coeff_schedulers_per_module.pop(module_id)

    @override(Learner)
    def additional_update_for_module(
        self,
        *,
        module_id: ModuleID,
        hps: PPOLearnerHyperparameters,
        timestep: int,
        sampled_kl_values: dict,
    ) -> Dict[str, Any]:
        results = super().additional_update_for_module(
            module_id=module_id,
            hps=hps,
            timestep=timestep,
            sampled_kl_values=sampled_kl_values,
        )

        # Update entropy coefficient via our Scheduler.
        new_entropy_coeff = self.entropy_coeff_schedulers_per_module[module_id].update(
            timestep=timestep
        )
        results.update({LEARNER_RESULTS_CURR_ENTROPY_COEFF_KEY: new_entropy_coeff})

        return results


class PPOLearnerConnector(ConnectorV2):
    """The connector that PPO always prepends to the learner connector pipeline.

    Computes advantages
    """
    def __call__(
        self,
        *,
        input_: Any,
        episodes: List[EpisodeType],
        ctx: ConnectorContextV2,
        **kwargs,
    ):
        """Calculate advantages and value targets."""
        advantages = compute_advantages_for_episodes(
            batch=input_,
            episode=episodes,
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
            use_gae=self.config.use_gae,
            use_critic=True,
            rl_module=ctx.rl_module,
        )
        input_[SampleBatch.ADVANTAGES] = advantages

        return input_
