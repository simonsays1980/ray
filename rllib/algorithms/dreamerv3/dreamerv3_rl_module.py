"""
This file holds framework-agnostic components for DreamerV3's RLModule.
"""

import abc
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from ray.rllib.algorithms.dreamerv3.utils import do_symlog_obs
from ray.rllib.algorithms.dreamerv3.tf.models.actor_network import ActorNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.critic_network import CriticNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.dreamer_model import DreamerModel
from ray.rllib.algorithms.dreamerv3.tf.models.world_model import WorldModel
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.policy.eager_tf_policy import _convert_to_tf
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import one_hot


_, tf, _ = try_import_tf()


@ExperimentalAPI
class DreamerV3RLModule(RLModule, abc.ABC):
    @override(RLModule)
    def setup(self):
        super().setup()

        # Gather model-relevant settings.
        B = 1
        T = self.config.model_config_dict["batch_length_T"]
        horizon_H = self.config.model_config_dict["horizon_H"]
        gamma = self.config.model_config_dict["gamma"]

        # Build encoder and decoder from catalog.
        catalog = self.config.get_catalog()
        self.encoder = catalog.build_encoder(framework=self.framework)
        self.decoder = catalog.build_decoder(framework=self.framework)

        # Build the world model (containing encoder and decoder).
        self.world_model = catalog.build_world_model(
            framework=self.framework,
            encoder=self.encoder,
            decoder=self.decoder,
        )
        self.actor = catalog.build_actor(framework=self.framework)
        self.critic = catalog.build_critic(framework=self.framework)

        # Build the final dreamer model (containing the world model).
        self.dreamer_model = catalog.build_dreamer_model(
            framework=self.framework,
            world_model=self.world_model,
            actor=self.actor,
            critic=self.critic,
            horizon=horizon_H,
            gamma=gamma,
        )
        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)

        # Perform a test `call()` to force building the dreamer model's variables.
        if self.framework == "tf2":
            test_obs = np.tile(
                np.expand_dims(self.config.observation_space.sample(), (0, 1)),
                reps=(B, T) + (1,) * len(self.config.observation_space.shape),
            )
            test_actions = np.tile(
                np.expand_dims(
                    one_hot(
                        self.config.action_space.sample(),
                        depth=self.config.action_space.n,
                    ),
                    (0, 1),
                ),
                reps=(B, T, 1),
            )
            self.dreamer_model(
                inputs=_convert_to_tf(test_obs),
                actions=_convert_to_tf(test_actions.astype(np.float32)),
                is_first=_convert_to_tf(np.ones((B, T), np.float32)),
                start_is_terminated_BxT=_convert_to_tf(np.zeros((B * T,), np.float32)),
                horizon_H=horizon_H,
                gamma=gamma,
            )

        # Initialize the critic EMA net:
        self.critic.init_ema()

    @override(RLModule)
    def get_initial_state(self) -> NestedDict:
        # Use `DreamerModel`'s `get_initial_state` method.
        return self.dreamer_model.get_initial_state()

    @override(RLModule)
    def input_specs_inference(self) -> SpecDict:
        return [Columns.OBS, Columns.STATE_IN, "is_first"]

    @override(RLModule)
    def output_specs_inference(self) -> SpecDict:
        return [Columns.ACTIONS, Columns.STATE_OUT]

    @override(RLModule)
    def input_specs_exploration(self):
        return self.input_specs_inference()

    @override(RLModule)
    def output_specs_exploration(self) -> SpecDict:
        return self.output_specs_inference()

    @override(RLModule)
    def input_specs_train(self) -> SpecDict:
        return [Columns.OBS, Columns.ACTIONS, "is_first"]

    @override(RLModule)
    def output_specs_train(self) -> SpecDict:
        return [
            "sampled_obs_symlog_BxT",
            "obs_distribution_means_BxT",
            "reward_logits_BxT",
            "rewards_BxT",
            "continue_distribution_BxT",
            "continues_BxT",
            # Sampled, discrete posterior z-states (t1 to T).
            "z_posterior_states_BxT",
            "z_posterior_probs_BxT",
            "z_prior_probs_BxT",
            # Deterministic, continuous h-states (t1 to T).
            "h_states_BxT",
        ]

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        # Call the Dreamer-Model's forward_inference method and return a dict.
        actions, next_state = self.dreamer_model.forward_inference(
            observations=batch[SampleBatch.OBS],
            previous_states=batch[STATE_IN],
            is_first=batch["is_first"],
        )
        return {SampleBatch.ACTIONS: actions, STATE_OUT: next_state}

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        # Call the Dreamer-Model's forward_exploration method and return a dict.
        actions, next_state = self.dreamer_model.forward_exploration(
            observations=batch[SampleBatch.OBS],
            previous_states=batch[STATE_IN],
            is_first=batch["is_first"],
        )
        return {SampleBatch.ACTIONS: actions, STATE_OUT: next_state}

    @override(RLModule)
    def _forward_train(self, batch: NestedDict):
        # Call the Dreamer-Model's forward_train method and return its outputs as-is.
        return self.dreamer_model.forward_train(
            observations=batch[SampleBatch.OBS],
            actions=batch[SampleBatch.ACTIONS],
            is_first=batch["is_first"],
        )
