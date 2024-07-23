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
LEARNER_RESULTS_VF_EXPLAINED_VARIANCE_KEY = "vf_explained_variance"


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
    def _update_from_batch_or_episodes(
        self,
        *,
        batch=None,
        episodes=None,
        **kwargs,
    ):
        # First perform GAE computation on the entirety of the given train data (all
        # episodes).
        if self.config.enable_env_runner_and_connector_v2:
            batch, episodes = self._compute_gae_from_episodes(episodes=episodes)

        # Now that GAE (advantages and value targets) have been added to the train
        # batch, we can proceed normally (calling super method) with the update step.
        return super()._update_from_batch_or_episodes(
            batch=batch,
            episodes=episodes,
            **kwargs,
        )

    def _compute_gae_from_episodes(
        self,
        *,
        episodes: Optional[List[EpisodeType]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[EpisodeType]]]:
        """Computes GAE advantages (and value targets) given a list of episodes.

        Note that the episodes may be SingleAgent- or MultiAgentEpisodes and may be
        episode chunks (not starting from reset or ending prematurely).

        The GAE computation here is performed in a very efficient way via elongating
        all given episodes by 1 artificial timestep (last obs, actions, states, etc..
        repeated, last reward=0.0, etc..), then creating a forward batch from this data
        using the connector, pushing the resulting batch through the value function,
        thereby extracting the bootstrap values (at the artificially added time steos)
        and all other value predictions (all other timesteps) and then reducing the
        batch and episode lengths again accordingly.
        """
        if not episodes:
            raise ValueError(
                "`MARWILLearner._compute_gae_from_episodes()` must have the `episodes` "
                "arg provided! Otherwise, GAE/advantage computation can't be performed."
            )

        batch = {}

        sa_episodes_list = list(
            self._learner_connector.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=False
            )
        )
        # Make all episodes one ts longer in order to just have a single batch
        # (and distributed forward pass) for both vf predictions AND the bootstrap
        # vf computations.
        orig_truncateds_of_sa_episodes = add_one_ts_to_episodes_and_truncate(
            sa_episodes_list
        )

        # Call the learner connector (on the artificially elongated episodes)
        # in order to get the batch to pass through the module for vf (and
        # bootstrapped vf) computations.
        batch_for_vf = self._learner_connector(
            rl_module=self.module,
            data={},
            episodes=episodes,
            shared_data={},
        )
        # Perform the value model's forward pass.
        vf_preds = convert_to_numpy(self._compute_values(batch_for_vf))

        for module_id, module_vf_preds in vf_preds.items():
            # Collect new (single-agent) episode lengths.
            episode_lens_plus_1 = [
                len(e)
                for e in sa_episodes_list
                if e.module_id is None or e.module_id == module_id
            ]

            # Remove all zero-padding again, if applicable, for the upcoming
            # GAE computations.
            module_vf_preds = unpad_data_if_necessary(
                episode_lens_plus_1, module_vf_preds
            )
            # Compute value targets.
            module_value_targets = compute_value_targets(
                values=module_vf_preds,
                rewards=unpad_data_if_necessary(
                    episode_lens_plus_1,
                    convert_to_numpy(batch_for_vf[module_id][Columns.REWARDS]),
                ),
                terminateds=unpad_data_if_necessary(
                    episode_lens_plus_1,
                    convert_to_numpy(batch_for_vf[module_id][Columns.TERMINATEDS]),
                ),
                truncateds=unpad_data_if_necessary(
                    episode_lens_plus_1,
                    convert_to_numpy(batch_for_vf[module_id][Columns.TRUNCATEDS]),
                ),
                gamma=self.config.gamma,
                lambda_=self.config.lambda_,
            )

            # Remove the extra timesteps again from vf_preds and value targets. Now that
            # the GAE computation is done, we don't need this last timestep anymore in
            # any of our data.
            module_vf_preds, module_value_targets = remove_last_ts_from_data(
                episode_lens_plus_1,
                module_vf_preds,
                module_value_targets,
            )
            module_advantages = module_value_targets - module_vf_preds
            # Drop vf-preds, not needed in loss. Note that in the PPORLModule, vf-preds
            # are recomputed with each `forward_train` call anyway.
            # Standardize advantages (used for more stable and better weighted
            # policy gradient computations).
            module_advantages = (module_advantages - module_advantages.mean()) / max(
                1e-4, module_advantages.std()
            )

            # Restructure ADVANTAGES and VALUE_TARGETS in a way that the Learner
            # connector can properly re-batch these new fields.
            batch_pos = 0
            for eps in sa_episodes_list:
                if eps.module_id is not None and eps.module_id != module_id:
                    continue
                len_ = len(eps) - 1
                self._learner_connector.add_n_batch_items(
                    batch=batch,
                    column=Postprocessing.ADVANTAGES,
                    items_to_add=module_advantages[batch_pos : batch_pos + len_],
                    num_items=len_,
                    single_agent_episode=eps,
                )
                self._learner_connector.add_n_batch_items(
                    batch=batch,
                    column=Postprocessing.VALUE_TARGETS,
                    items_to_add=module_value_targets[batch_pos : batch_pos + len_],
                    num_items=len_,
                    single_agent_episode=eps,
                )
                batch_pos += len_

        # Remove the extra (artificial) timesteps again at the end of all episodes.
        remove_last_ts_from_episodes_and_restore_truncateds(
            sa_episodes_list,
            orig_truncateds_of_sa_episodes,
        )

        return batch, episodes

    @override(Learner)
    def remove_module(self, module_id: ModuleID) -> None:
        super().remove_module(module_id)
        self.moving_avg_sqd_adv_norms_per_module.pop(module_id)
