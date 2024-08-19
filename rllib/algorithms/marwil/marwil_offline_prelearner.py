import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.offline.offline_prelearner import OfflinePreLearner, SCHEMA

from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override, OverrideToImplementCustomLogic
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets
from ray.rllib.utils.postprocessing.episodes import (
    add_one_ts_to_episodes_and_truncate,
    remove_last_ts_from_data,
    remove_last_ts_from_episodes_and_restore_truncateds,
)
from ray.rllib.utils.postprocessing.zero_padding import unpad_data_if_necessary
from ray.rllib.utils.typing import EpisodeType, TensorType


class MARWILOfflinePreLearner(OfflinePreLearner):
    @OverrideToImplementCustomLogic
    @override(OfflinePreLearner)
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, MultiAgentBatch]:

        # If we directly read in episodes we just convert to list.
        if self.input_read_episodes:
            episodes = batch["item"].tolist()
        # Otherwise we ap the batch to episodes.
        else:
            # Map the batch to episodes.
            episodes = episodes = self._map_to_episodes(
                self._is_multi_agent,
                batch,
                schema=SCHEMA | self.config.input_read_schema,
                finalize=True,
                input_compress_columns=self.config.input_compress_columns,
                observation_space=self.observation_space,
                action_space=self.action_space,
            )["episodes"]

        # Compute the GAE via the Learner's method.
        batch, episodes = self._compute_gae_from_episodes(episodes=episodes)

        # Run the `Learner`'s connector pipeline.
        batch = self._learner_connector(
            rl_module=self._module,
            data=batch,
            episodes=episodes,
            shared_data={},
        )

        # Convert to `MultiAgentBatch`.
        batch = MultiAgentBatch(
            {
                module_id: SampleBatch(module_data)
                for module_id, module_data in batch.items()
            },
            # TODO (simon): This can be run once for the batch and the
            # metrics, but we run it twice: here and later in the learner.
            env_steps=sum(e.env_steps() for e in episodes),
        )
        # Remove all data from modules that should not be trained. We do
        # not want to pass around more data than necessaty.
        for module_id in list(batch.policy_batches.keys()):
            if not self._should_module_be_updated(module_id, batch):
                del batch.policy_batches[module_id]

        # TODO (simon): Log steps trained for metrics (how?). At best in learner
        # and not here. But we could precompute metrics here and pass it to the learner
        # for logging. Like this we do not have to pass around episode lists.

        # TODO (simon): episodes are only needed for logging here.
        return {"batch": [batch]}

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
            rl_module=self._module,
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

    def _compute_values(
        self,
        batch_for_vf: Dict[str, Any],
    ) -> Union[TensorType, Dict[str, Any]]:
        """Computes the value function predictions for the module being optimized.

        This method must be overridden by multiagent-specific algorithm learners to
        specify the specific value computation logic. If the algorithm is single agent
        (or independent multi-agent), there should be no need to override this method.

        Args:
            batch_for_vf: The multi-agent batch (mapping ModuleIDs to module data) to
                be used for value function predictions.

        Returns:
            A dictionary mapping module IDs to individual value function prediction
            tensors.
        """
        return {
            module_id: self._module[module_id].unwrapped().compute_values(module_batch)
            for module_id, module_batch in batch_for_vf.items()
            if self._should_module_be_updated(module_id, batch_for_vf)
        }
