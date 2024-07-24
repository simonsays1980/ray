from typing import Any, Dict, List, Optional, Tuple, Union

from ray.rllib.connectors.common.add_observations_from_episodes_to_batch import (
    AddObservationsFromEpisodesToBatch,
)
from ray.rllib.connectors.learner.add_next_observations_from_episodes_to_train_batch import (  # noqa
    AddNextObservationsFromEpisodesToTrainBatch,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import Learner
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.annotations import (
    override,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
)
from ray.rllib.utils.typing import EpisodeType, ModuleID, TensorType

LEARNER_RESULTS_MOVING_AVG_SQD_ADV_NORM_KEY = "moving_avg_sqd_adv_norm"
LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY = "vf_explained_variance"


# TODO (simon): Check, if the norm update should be done inside
# the Learner.
class MARWILLearner(Learner):
    @override(Learner)
    def build(self) -> None:
        super().build()

        # Dict mapping module IDs to the respective moving averages of squared
        # advantages.
        self.moving_avg_sqd_adv_norms_per_module: Dict[
            ModuleID, TensorType
        ] = LambdaDefaultDict(
            lambda module_id: self._get_tensor_variable(
                self.config.get_config_for_module(
                    module_id
                ).moving_average_sqd_adv_norm_start
            )
        )

        # Prepend a NEXT_OBS from episodes to train batch connector piece (right
        # after the observation default piece).
        if (
            self.config.add_default_connectors_to_learner_pipeline
            and self.config.enable_env_runner_and_connector_v2
        ):
            self._learner_connector.insert_after(
                AddObservationsFromEpisodesToBatch,
                AddNextObservationsFromEpisodesToTrainBatch(),
            )

    @override(Learner)
    def remove_module(self, module_id: ModuleID) -> None:
        super().remove_module(module_id)
        self.moving_avg_sqd_adv_norms_per_module.pop(module_id)
