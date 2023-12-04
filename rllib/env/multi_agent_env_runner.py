import gymnasium as gym
import numpy as np
import tree

from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module.marl_module import (
    ModuleID,
    MultiAgentRLModule,
    MultiAgentRLModuleSpec,
)
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import (
    ModelWeights,
    MultiAgentDict,
    TensorStructType,
    TensorType,
)
from ray.tune.registry import ENV_CREATOR, _global_registry

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

    # TODO (sven): This gives a tricky circular import that goes
    #  deep into the library. We have to see, where to dissolve it.
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

_, tf, _ = try_import_tf()
torch, nn = try_import_torch()


class MultiAgentEnvRunner(EnvRunner):
    """The genetic environment runner for the multi agent case."""

    @override(EnvRunner)
    def __init__(self, config: "AlgorithmConfig", **kwargs):
        """Initializes a `MultiAgentEnvRunner` instance.

        Args:
            config: An `AlgorithmConfig` object containing all parameters
                needed to build this `EnvRunner` class.
        """
        super().__init__(config=config)

        # Get the worker index on which this instance is running.
        self.worker_index: int = kwargs.get("worker_index")

        # TODO (simon): Instantiate an error in `Algorithm.validate()`
        # if `num_envs_per_worker > 1` and `is_multi_agent==True`.

        # Register env for the local context.
        # Note, `gym.register` has to be called on each worker.
        if isinstance(self.config.env, str) and _global_registry.contains(
            ENV_CREATOR, self.config.env
        ):
            entry_point = partial(
                _global_registry.get(ENV_CREATOR, self.config.env),
                self.config.env_config,
            )

        else:
            entry_point = partial(
                _gym_env_creator,
                env_context=self.config.env_config,
                env_descriptor=self.config.env,
            )
        gym.register(
            "rllib-multi-agent-env-runner-v0",
            entry_point=entry_point,
            disable_env_checker=True,
        )

        # Wrap into `VectorListInfo`` wrapper to get infos as lists.
        self.env = gym.make(
            "rllib-multi-agent-env-runner-v0",
        )

        # Create the vectorized gymnasium env.
        assert isinstance(self.env.unwrapped, MultiAgentEnv), (
            "ERROR: When using the `MultiAgentEnvRunner` the environment needs "
            "to inherit from `ray.rllib.env.multi_agent_env.MultiAgentEnv`."
        )

        # Check, if spaces are in preferred format, i.e. `gym.spaces.Dict` with
        # agent ids mapping to agent spaces.
        self._action_space_in_preferred_format = (
            self.env.unwrapped._check_if_action_space_maps_agent_id_to_sub_space()
        )
        self._obs_space_in_preferred_format = (
            self.env.unwrapped._check_if_obs_space_maps_agent_id_to_sub_space()
        )

        # TODO (simon): An env runner needs not to know about what agents are there.
        # We can pull these when needed from the environment or the episode.
        self.agent_ids: List[Union[str, int]] = self.env.get_agent_ids()

        # Create our own instance of the (single-agent) `RLModule` (which
        # the needs to be weight-synched) each iteration.
        # TODO (sven, simon): We have to rebuild the `AlgorithmConfig` to work on
        # `RLModule`s and not `Policy`s. Like here `policies`->`modules`
        try:
            policy_dict, _ = self.config.get_multi_agent_setup(env=self.env)
            module_spec: MultiAgentRLModuleSpec = self.config.get_marl_module_spec(
                policy_dict=policy_dict
            )

            # TODO (simon): The `gym.Wrapper` for `gym.vector.VectorEnv` should
            #  actually hold the spaces for a single env, but for boxes the
            #  shape is (1, 1) which brings a problem with the action dists.
            #  shape=(1,) is expected.
            module_spec.action_space = self.env.action_space
            module_spec.observation_space = self.env.observation_space
            # Set action and observation spaces for all module specs.
            for agent_id, agent_module_spec in module_spec.module_specs.items():
                # Note, `MultiAgentEnv` has a preferred format of spaces, i.e.
                # a mapping from agent ids to spaces.
                # If the action space is a mapping from agent ids to spaces.
                if self._action_space_in_preferred_format:
                    agent_module_spec.action_space = self.env.action_space[agent_id]
                # Otherwise, use the same space for each agent.
                else:
                    agent_module_spec.action_space = self.env.action_space
                # Similar to the action spaces, observation spaces could be described
                # by a mapping.
                if self._obs_space_in_preferred_format:
                    agent_module_spec.observation_space = self.env.observation_space[
                        agent_id
                    ]
                # Otherwise, use the same space for all agents.
                else:
                    agent_module_spec.observation_space = self.env.observation_space

            # Build the module from its spec.
            self.marl_module: MultiAgentRLModule = module_spec.build()
            # Get the policy mapping function b/c we need it to map observations and
            # states to single-agent `RLModule`s.
            # TODO (sven, simon): We have to rebuild the `AlgorithmConfig` to work on
            # `RLModule`s and not `Policy`s. Like here `policies`->`modules`
            self._agent_id_to_module_id = {
                agent_id: self.config.policy_mapping_fn()(agent_id, None, self)
                for agent_id in self.agent_ids
            }
            # Remove as soon as multi-agent connectors are ready.
            self._module_id_to_agent_id = {
                module_id: agent_id
                for agent_id, module_id in self._agent_id_to_module_id.items()
            }
        # This error could be thrown, when only random actions are used.
        except NotImplementedError:
            self.marl_module = None

        # This should be the default.
        self._needs_initial_reset: bool = True
        self._episode: "MultiAgentEpisode" = None

        self._done_episodes_for_metrics: List["MultiAgentEpisode"] = []
        self._ongoing_episode_for_metrics: Dict[
            List["MultiAgentEpisode"]
        ] = defaultdict(list)
        self._ts_since_last_metrics: int = 0
        self._weights_seq_no: int = 0

        # TODO (simon): Following `SingleAgentEpisode`
        # This is a temporary solution. STATE_OUTs
        # will be resolved entirely as `extra_model_outputs` and
        # not be stored separately inside Episodes.
        self._states = {agent_id: None for agent_id in self.marl_module.keys()}

    @override(EnvRunner)
    def sample(
        self,
        *,
        num_timesteps: int = None,
        num_episodes: int = None,
        explore: bool = True,
        random_actions: bool = False,
        with_render_data: bool = False,
    ) -> List["MultiAgentEpisode"]:
        """Runs and returns a sample (n timesteps or m episodes) on the env(s).

        Args:
            num_timesteps: int. Number of timesteps to sample during rollout.
                Note, only one parameter, `num_timetseps` or `num_episodes`
                can be provided.
            num_episodes: int. Number of episodes to sample during rollout.
                Note, only one parameter, `num_timetseps` or `num_episodes`
                    can be provided.
            explore: boolean. If in exploration or inference mode. Exploration
                mode might for some algorithms provide extza model outputs that
                are redundant in inference mode.
            random_actions: boolean. If actions should be sampled from the action
                space. In default mode (i.e. `False`) we sample actions frokm the
                policy.
            with_render_data: If render data from the environment should be collected.
                This is only available when sampling episodes, i.e. `num_episodes` is
                not `None`.
        Returns:
            `Lists of `MultiAgentEpisode` instances, carrying the collected sample data.
        """
        assert not (num_timesteps is not None and num_episodes is not None)

        # If no execution details are provided, use the config.
        if num_timesteps is None and num_episodes is None:
            if self.config.batch_mode == "truncate_episodes":
                num_timesteps = self.config.get_rollout_fragment_length(
                    worker_index=self.worker_index,
                )
            else:
                num_episodes = 1

        # Sample n timesteps
        if num_timesteps is not None:
            return self._sample_timesteps(
                num_timesteps=num_timesteps,
                explore=explore,
                random_actions=random_actions,
                force_reset=False,
            )
        # Sample m episodes.
        else:
            return self._sample_episodes(
                num_episodes=num_episodes,
                explore=explore,
                random_actions=random_actions,
                with_render_data=with_render_data,
            )

    def _sample_timesteps(
        self,
        num_timesteps: int,
        explore: bool = True,
        random_actions: bool = False,
        force_reset: bool = False,
    ) -> List["MultiAgentEpisode"]:
        """Helper method to sample n timesteps.

        Args:
            num_timesteps: int. Number of timesteps to sample during rollout.
            explore: boolean. If in exploration or inference mode. Exploration
                mode might for some algorithms provide extza model outputs that
                are redundant in inference mode.
            random_actions: boolean. If actions should be sampled from the action
                space. In default mode (i.e. `False`) we sample actions frokm the
                policy.
            with_render_data: If render data from the environment should be collected.
                This is only available when sampling episodes, i.e. `num_episodes` is
                not `None`.

        Returns:
            `Lists of `MultiAgentEpisode` instances, carrying the collected sample data.
        """

        # TODO (sven): This gives a tricky circular import that goes
        # deep into the library. We have to see, where to dissolve it.
        from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

        done_episodes_to_return: List["MultiAgentEpisode"] = []

        # Get the initial states for all modules. Note, `get_initial_state()`
        # returns an empty dictionary, if no initial states are defined.
        # TODO (sven, simon): We could simply use `MARLModule._run_forward_pass()`
        # to get also an initial state for all modules. Maybe this can be
        # added to the MARLModule.
        if self.marl_module:
            initial_states = {
                agent_id: self.marl_module[agent_id].get_initial_state()
                for agent_id in self.marl_module.keys()
            }
        else:
            # TODO (sven, simon): Do we even need states when we have no module?
            initial_states = {agent_id: {} for agent_id in self.env.get_agent_ids()}

        # Have to reset the env.
        if force_reset or self._needs_initial_reset:
            # Reset the environment.
            # TODO (simon): CHeck, if we need here the seed from the config.
            obs, infos = self.env.reset()

            # We just reset the environment. We do not have to force this again
            # in the next call so `self._sample_timesteps()`.
            self._needs_initial_reset = False

            self._episode = MultiAgentEpisode(agent_ids=self.agent_ids)

            # Set the initial observations in the episodes.
            # TODO (sven): maybe move this into connector pipeline (even
            # if automated).
            self._episode.add_env_reset(observations=obs, infos=infos)

            # Get the states for all agents.
            states = initial_states
        # Do not reset environments, but instead continue in already started episodes.
        else:
            # Pick up stored observations from previous timesteps.
            obs = self._episode.get_observations(as_list=True)[0]
            # Get the states from the buffer or get the initial states.
            # TODO (simon): Do we need to iterate here over agents? Or can
            # one agent have no buffered state while another one has?
            states = {
                agent_id: initial_states[agent_id]
                if agent_state is None
                else agent_state
                for agent_id, agent_state in self._states.items()
            }

        # Loop through timesteps.
        env_steps = 0
        agent_steps = 0
        ts = 0

        while ts < num_timesteps:
            # Act randomly.
            if random_actions:
                # Note, to get sampled actions from all agents' action
                # spaces we need to call `MultiAgentEnv.action_space_sample()`.
                if self.env.unwrapped._action_space_iin_preferred_format:
                    actions = self.env.action_space.sample()
                # Otherwise, `action_space_sample()` needs to be implemented.
                else:
                    actions = self.env.action_space_sample()
                # Remove all actions for agents that had no observation.
                actions = {
                    agent_id: agent_action
                    for agent_id, agent_action in actions.items()
                    if agent_id in obs
                }

            else:
                # TODO (simon): This is not correct `forward()` expects
                # `SampleBatchType`.
                # Note, `RLModule`'s `forward()` methods expect `NestedDict`s.
                # Note, we only consider for states and obs agents that step.
                batch: MultiAgentDict = {
                    self._agent_id_to_module_id[agent_id]: {
                        STATE_IN: tree.map_structure(
                            lambda s: self._convert_from_numpy(s),
                            states[self._agent_id_to_module_id[agent_id]],
                        ),
                        SampleBatch.OBS: self._convert_from_numpy(
                            np.expand_dims(agent_obs, axis=0)
                        ),
                    }
                    for agent_id, agent_obs in obs.items()
                }
                # TODO (Sven, Simon): The `RLModule` has `SampleBatchType` as input
                # type. Only the _forward_x()` methods have a `NestedDict`. Shall we
                # compile to `SampleBatchType` here and in `SingleAgentEnvRunner`?
                from ray.rllib.utils.nested_dict import NestedDict

                batch = NestedDict(batch)

                # Explore or not.
                if explore:
                    fwd_out = self.marl_module.forward_exploration(batch)
                else:
                    fwd_out = self.marl_module.forward_inference(batch)

                # Sample the actions or draw randomly.
                actions, action_logps = self._sample_actions_if_necessary(
                    fwd_out,
                    explore=explore,
                )

                # Convert to numpy for recording later to the episode.
                fwd_out = tree.map_structure(convert_to_numpy, fwd_out)

                # Assign the new states for the agents that stepped.
                if STATE_OUT in fwd_out:
                    states.update(tree.map_structure(lambda s: s[STATE_OUT], fwd_out))

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

            env_steps += 1
            agent_steps += len(obs)
            # If we count by environment steps.
            # TODO (sven, simon): We have to record these steps somewhere.
            # TODO: Refactor into multiagent-episode sth. like `get_agent_steps()`.
            if self.config.count_steps_by == "env_steps":
                ts = env_steps
            # Or by agent steps.
            else:
                ts = agent_steps

            # TODO (simon): Remove when connector API for multi-agent is ready.
            extra_model_outputs = {
                self._module_id_to_agent_id[module_id]: {
                    k: v for k, v in module_fwd_out.items() if k != SampleBatch.ACTIONS
                }
                for module_id, module_fwd_out in fwd_out.items()
            }
            # TODO (sven, simon): There are algos that do not need ACTION_LOGP.
            for agent_id, agent_extra_model_output in extra_model_outputs.items():
                agent_extra_model_output[SampleBatch.ACTION_LOGP] = action_logps[
                    agent_id
                ]

            # Record the timestep in the episode instance.
            self._episode.add_env_step(
                obs,
                actions,
                rewards,
                infos=infos,
                terminateds=terminateds,
                truncateds=truncateds,
                extra_model_outputs=extra_model_outputs,
            )

            # TODO (sven, simon): We have to check, if we need this elaborate
            # function here or if the `MultiAgentEnv` defines the cases that
            # can happen.
            # Right now we have:
            #   1. Most times only agents that step get `terminated`, `truncated`
            #       i.e. the rest we have to check in the episode.
            #   2. There are edge cases like, some agents terminated, all others
            #       truncated and vice versa.
            # See also `MultiAgentEpisode` for handling the `__all__`.
            if self._all_agents_done(terminateds, truncateds):
                # Reset all h-states to the model's initial ones b/c we are starting
                # a new episode.
                if self.marl_module and self.marl_module.is_stateful():
                    states = initial_states

                # Finish the episode.
                for agent_eps in self._episode.agent_episodes.values():
                    agent_eps.finalize()
                # TODO (simon): Call here the `MAE.finalize()` method when ready.
                done_episodes_to_return.append(self._episode)
                # Create a new episode instance.
                self._episode = MultiAgentEpisode(agent_ids=self.agent_ids)
                # Reset the environment.
                obs, infos = self.env.reset()
                # Add initial observations and infos.
                self._episode.add_env_reset(observations=obs, infos=infos)
                # Reset h-states to the models' initial ones b/c we are starting a new
                # episode.
                if self.marl_module:
                    # TODO (sven, simon): Are there cases where a module overrides its
                    # own `initial_states` while learning?
                    states = initial_states
            else:
                # Buffer the states for an eventual next `sample()` call.
                self._states = states

        # Return done episodes ...
        # TODO (simon): Check, how much memory this attribute uses.
        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        # ... and the ongoing episode chunk. Exclude the ongoing episode if
        # it is only initialized.
        ongoing_episode: List["MultiAgentEpisode"] = (
            [self._episode] if self._episode.t > 0 else []
        )
        # Also make sure, we return a copy and start new chunks so that callers
        # of this function do not alter the ongoing and returned episode object.
        self._episode = self._episode.cut()
        if ongoing_episode:
            for agent_eps in ongoing_episode[0].agent_episodes.values():
                agent_eps.finalize()
            self._ongoing_episode_for_metrics[ongoing_episode[0].id_].append(
                ongoing_episode[0]
            )

        # Record last metrics collection.
        self._ts_since_last_metrics += ts

        # Return collected episode data.
        return done_episodes_to_return + ongoing_episode

    def _sample_episodes(
        self,
        num_episodes: int,
        explore: bool = True,
        random_actions: bool = False,
        with_render_data: bool = False,
    ) -> List["MultiAgentEpisode"]:
        """Helper method to run n episodes.

        Args:
            num_episodes: int. Number of episodes to sample during rollout.
            explore: boolean. If in exploration or inference mode. Exploration
                mode might for some algorithms provide extza model outputs that
                are redundant in inference mode.
            random_actions: boolean. If actions should be sampled from the action
                space. In default mode (i.e. `False`) we sample actions frokm the
                policy.
            with_render_data: If render data from the environment should be collected.
                This is only available when sampling episodes, i.e. `num_episodes` is
                not `None`.

        Returns:
            `Lists of `MultiAgentEpisode` instances, carrying the collected sample data.
        """

        # TODO (sven): This gives a tricky circular import that goes
        # deep into the library. We have to see, where to dissolve it.
        from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

        # If user calls sample(num_timesteps=..) after this, we must reset again
        # at the beginning.
        self._needs_initial_reset = True

        done_episodes_to_return: List["MultiAgentEpisode"] = []

        # Get the initial states for all modules. Note, `get_initial_state()`
        # returns an empty dictionary, if no initial states are defined.
        # TODO (sven, simon): We could simply use `MARLModule._run_forward_pass()`
        # to get also an initial state for all modules. Maybe this can be
        # added to the MARLModule.
        if self.marl_module:
            initial_states = {
                agent_id: self.marl_module[agent_id].get_initial_state()
                for agent_id in self.marl_module.keys()
            }
        else:
            initial_states = {agent_id: {} for agent_id in self.env.get_agent_ids()}

        # Reset the environment.
        # TODO (simon): CHeck, if we need here the seed from the config.
        obs, infos = self.env.reset()

        # Create a new multi-agent episode.
        self._episode = MultiAgentEpisode(agent_ids=self.agent_ids)

        # Initialize image rendering if needed.
        render_image = None
        if with_render_data:
            render_image = self.env.render()

        # Set the initial observations in the episodes.
        # TODO (sven): maybe move this into connector pipeline (even
        # if automated).
        self._episode.add_env_reset(
            observations=obs, infos=infos, render_image=render_image
        )
        # Set states to initial states and start sampling.
        states = initial_states

        # Loop over episodes.
        eps = 0
        while eps < num_episodes:
            # Act randomly.
            if random_actions:
                # Note, to get sampled actions from all agents' action
                # spaces we need to call `MultiAgentEnv.action_space_sample()`.
                if self.env.unwrapped._action_space_iin_preferred_format:
                    actions = self.env.action_space.sample()
                # Otherwise, `action_space_sample()` needs to be implemented.
                else:
                    actions = self.env.action_space_sample()
                # Remove all actions for agents that had no observation.
                actions = {
                    agent_id: agent_action
                    for agent_id, agent_action in actions.items()
                    if agent_id in obs
                }

            else:
                # TODO (sven): This will move entirely into connector logic in
                #  upcoming PR.
                # TODO (simon): This is not correct `forward()` expects
                # `SampleBatchType`.
                # Note, `RLModule`'s `forward()` methods expect `NestedDict`s.
                # Note, we only consider for states and obs of agents that step.
                batch: MultiAgentDict = {
                    agent_id: {
                        STATE_IN: tree.map_structure(
                            lambda s: self._convert_from_numpy(s),
                            states[agent_id],
                        ),
                        SampleBatch.OBS: self._convert_from_numpy(
                            np.expand_dims(agent_obs, axis=0)
                        ),
                    }
                    for agent_id, agent_obs in obs.items()
                }
                # TODO (Sven, Simon): The `RLModule` has `SampleBatchType` as input
                # type. Only the _forward_x()` methods have a `NestedDict`. Shall we
                # compile to `SampleBatchType` here and in `SingleAgentEnvRunner`?
                from ray.rllib.utils.nested_dict import NestedDict

                batch = NestedDict(batch)

                # Explore or not.
                if explore:
                    fwd_out = self.marl_module.forward_exploration(batch)
                else:
                    fwd_out = self.marl_module.forward_inference(batch)

                # Sample the actions or draw randomly.
                actions, action_logps = self._sample_actions_if_necessary(
                    fwd_out,
                    explore=explore,
                )

                # Convert to numpy for recording later to the episode.
                fwd_out = tree.map_structure(convert_to_numpy, fwd_out)

                # Assign the new states for the agents that stepped.
                if STATE_OUT in fwd_out:
                    states.update(tree.map_structure(lambda s: s[STATE_OUT], fwd_out))

            # Step the environment.
            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            # Add render data if needed.
            if with_render_data:
                render_image = self.env.render()

            extra_model_outputs = {
                agent_id: {
                    k: v for k, v in agent_fwd_out.items() if k != SampleBatch.ACTIONS
                }
                for agent_id, agent_fwd_out in fwd_out.items()
            }
            # TODO (sven, simon): There are algos that do not need ACTION_LOGP.
            for agent_id, agent_extra_model_output in extra_model_outputs.items():
                agent_extra_model_output[SampleBatch.ACTION_LOGP] = action_logps[
                    agent_id
                ]

            # Record the timestep in the episode instance.
            self._episode.add_env_step(
                obs,
                actions,
                rewards,
                infos=infos,
                terminateds=terminateds,
                truncateds=truncateds,
                extra_model_outputs=extra_model_outputs,
                render_image=render_image,
            )

            # TODO (sven, simon): We have to check, if we need this elaborate
            # function here or if the `MultiAgentEnv` defines the cases that
            # can happen.
            # Right now we have:
            #   1. Most times only agents that step get `terminated`, `truncated`
            #       i.e. the rest we have to check in the episode.
            #   2. There are edge cases like, some agents terminated, all others
            #       truncated and vice versa.
            # See also `MultiAgentEpisode` for handling the `__all__`.
            if self._all_agents_done(terminateds, truncateds):
                # Increase episode count.
                eps += 1
                # Reset all h-states to the model's initial ones b/c we are starting
                # a new episode.
                if self.marl_module and self.marl_module.is_stateful():
                    states = initial_states

                # Finish the episode.
                # TODO (simon): Call here the `MAE.finalize()` method when ready.
                done_episodes_to_return.append(self._episode)
                # Create a new episode instance.
                self._episode = MultiAgentEpisode(agent_ids=self.agent_ids)
                # Reset the environment.
                obs, infos = self.env.reset()
                # Add initial observations and infos.
                self._episode.add_env_reset(observations=obs, infos=infos)
                # Reset h-states to the models' initial ones b/c we are starting a new
                # episode.
                if self.marl_module:
                    # TODO (sven, simon): Are there cases where a module overrides its
                    # own `initial_states` while learning?
                    states = initial_states

        self._done_episodes_for_metrics.extend(done_episodes_to_return)
        # TODO (sven, simon): It might be appropriate here to replace this attribute
        # with two, one for env steps and one for agent steps.
        self._ts_since_last_metrics += sum(len(eps) for eps in done_episodes_to_return)

        return done_episodes_to_return

    def set_weights(
        self,
        weights: Dict[ModuleID, ModelWeights],
        global_vars: Optional[Dict] = None,
        weights_seq_no: int = 0,
    ) -> None:
        """Writes the weights of our multi-agent `RLModule`

        Args:
            weigths: A dictionary mapping `ModuleID`s to the new weigths to
                be used in the `MultiAgentRLModule` stored in this instance.
            global_vars: An optional global vars dictionary to set this
                worker to. If None, do not update the global_vars.
            weights_seq_no: If needed, a sequence number for the weights version
                can be passed into this method. If not None, will store this seq no
                (in self.weights_seq_no) and in future calls - if the seq no did not
                change wrt. the last call - will ignore the call to save on performance.

        .. testcode::
            :skipif: True

            from ray.rllib.env import MultiAgentEnvRunner
            # Create an `MultiAgentEnvRunner`.
            worker = ...
            weights = worker.get_weights()
            # Set `global_vars` (timestep) as well.
            worker.set_weights(weights, {"timestep": 42})
        """

        self.marl_module.set_state(weights)

    def get_weights(self, modules=None) -> Dict[ModuleID, ModelWeights]:
        """Returns the weights of our multi-agent `RLModule`.

        Args:
            modules: `ModuleID`s for which to return the weights. If `None`
                weigths for all modules are returned. See for details
                `MultiAgentRLModule.get_state()`.

        Returns:
            A dictionary mapping `ModuleID`s to their corresponding weights.
        """

        return self.marl_module.get_state(module_ids=modules)

    @override(EnvRunner)
    def assert_healthy(self):
        """Checks that self.__init__() has been completed properly.

        Ensures that the instances has a `MultiAgentRLModule` and an
        environment defined.

        Raises:
            AssertionError: If the EnvRunner Actor has NOT been properly initialized.
        """

        # Make sure, we have built our gym.vector.Env and RLModule properly.
        assert self.env and self.marl_module

    @override(EnvRunner)
    def stop(self):
        """Closes this `EnvRunner` by running necessary closing operations."""

        # Note, `MultiAgentEnv` inherits `close()`-method from `gym.Env`.
        self.env.close()

    def _convert_from_numpy(self, array: np.array) -> TensorType:
        """Converts a numpy array to a framework-specific tensor.

        This helper function is needed as an environment returns numpy
        arrays and the `RLModule`s need `TensorType`s.

        Args:
            array: numpy.array. An array to be converted to a framework-specific
                tensor.

        Returns:
            A framework-specific tensor containing all values from `array`.
        """

        if self.config.framework_str == "torch":
            return torch.from_numpy(array)
        else:
            return tf.convert_to_tensor(array)

        # TODO (sven): Replace by default "to-env" connector.

    def _sample_actions_if_necessary(
        self, fwd_out: TensorStructType, explore: bool = True
    ) -> Tuple[np.array, np.array]:
        """Samples actions from action distribution if necessary.

        The `RLModule`s return `TensorType`s specific to the framework
        used (i.e. either Torch or TensorFlow). These need to be converted.
        In addition, action log-probabilities are converted if provided
        by an agent module.
        Furthermore, the `MultiAgentRLModule` returns a `MultiAgentDict`
        mapping `ModuleID`s to their corresponding `forward()` outputs.
        The environment needs, however, a `MultiAgentDict` mapping
        agent ids to their corresponding actions, only. This function
        converts the parameters accordingly.

        Args:
            fwd_out: dict. A dictionary mapping `ModuleID`s to the outputs
                of their module`s `forward()` method.

        Returns:
            A `tuple` containing a `MultiAgentDict` mapping agent ids to their
            corresponding actions and a second `MultiAgentDict` mapping agent ids
            to their corresponding action log-probabilities if provided by their
            modules.
        """

        # TODO (sven): Move this into connector pipeline (if no
        # "actions" key in returned dict, sample automatically as
        # the last piece of the connector pipeline; basically do
        # the same thing that the Policy is currently doing, but
        # using connectors)
        actions = {}
        action_logps = {}
        for module_id, module_fwd_out in fwd_out.items():
            # If actions are provided just load them.
            if SampleBatch.ACTIONS in module_fwd_out.keys():
                actions[self._module_id_to_agent_id[module_id]] = convert_to_numpy(
                    module_fwd_out[SampleBatch.ACTIONS]
                )
                # TODO (simon, sven): Some algos do not return logps.
                if SampleBatch.ACTION_LOGP in module_fwd_out:
                    action_logps[
                        self._module_id_to_agent_id[module_id]
                    ] = convert_to_numpy(module_fwd_out[SampleBatch.ACTION_LOGP])
            # If no actions are provided we need to sample them.
            else:
                # Explore or not.
                if explore:
                    action_dist_cls = self.marl_module[
                        module_id
                    ].get_exploration_action_dist_cls()
                else:
                    action_dist_cls = self.marl_module[
                        module_id
                    ].get_inference_action_dist_cls()
                # Generate action distribution and sample actions.
                action_dist = action_dist_cls.from_logits(
                    module_fwd_out[SampleBatch.ACTION_DIST_INPUTS]
                )
                action = action_dist.sample()
                # We need numpy actions for gym environments.
                action_logps[self._module_id_to_agent_id[module_id]] = convert_to_numpy(
                    action_dist.logp(action)
                )
                actions[self._module_id_to_agent_id[module_id]] = convert_to_numpy(
                    action
                ).reshape(-1)
                # Squeeze for the last dimension if necessary.
                # TODO (sven, simon): This is not optimal here. But there seems
                # to be some differences between MultiDiscrete action spaces
                # and Box action spaces for `gym.VectorEnv`.
                # For the former we have to squeeze away the last action
                # dimension delivered from the action_dist and for the latter
                # we should not. This might be connected to the way how the
                # `action_space` is defined for the `RLModule` in the
                # `__init__()` of this class here.
                # if actions.ndim > len(self.env.action_space.shape):
                #    actions = actions.squeeze(axis=-1)

        return actions, action_logps

    def _all_agents_done(self, terminateds, truncateds):
        """Determines, if all agents are either terminated or truncated

        Note, this is not determined by the `__all__` in an `MultiAgentEnv`
        as this does not cover the case, if some agents are truncated and
        all the others are terminated and vice versa.

        Args:
            terminateds: dict. A dictionary mapping an agent id to a
                corresponding boolean indicating if the agent is terminated.
            truncateds: dict. A dictionary mapping an agent id to a
                corresponding boolean indicating if the agent is truncated.

        Returns:
            A boolean indicating if all agents are done.
        """

        # CASE 1: all agents are terminated or all are truncated.
        if terminateds["__all__"] or truncateds["__all__"]:
            return True
        # If not we have two further
        else:
            # TODO (simon): Refactor into `MultiAgentEpisode`.
            # Find all agents that were done at prior timesteps.
            agents_done = [
                agent_id
                for agent_id, agent_eps in self._episode.agent_episodes.items()
                if agent_eps.is_done
            ]
            # Add the agents that are done at the present timestep.
            agents_done += [
                agent_id for agent_id in terminateds if terminateds[agent_id]
            ]
            agents_done += [agent_id for agent_id in truncateds if truncateds[agent_id]]
            # CASE 2: some agents are truncated and the others are terminated.
            if all(agent_id in set(agents_done) for agent_id in self.agent_ids):
                return True
            # CASE 3: there are still some agents alive.
            else:
                return False
