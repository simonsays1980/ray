import numpy as np
import uuid

from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union

from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict


# TODO (simon): Include cases in which the number of agents in an
# episode are shrinking or growing during the episode itself.
# TODO (simon): When creating a successor we do start the global
# timestep mapping at 0 again, even though along the episode this
# is not the case.
# TODO (simon): When creating a successor we copy over the rewards
# and the partial reward timestep mapping, this however might not toally
# match with the global timestep mapping - plays a role when we create
# another successor from that one and concatenate episodes.
class MultiAgentEpisode:
    """Stores multi-agent episode data.

    The central attribute of the class is the timestep mapping
    `global_t_to_local_t` that maps the global (environment)
    timestep to the local (agent) timesteps.

    The `MultiAgentEpisode` is based on the `SingleAgentEpisode`s
    for each agent, stored in `MultiAgentEpisode.agent_episodes`.
    """

    def __init__(
        self,
        id_: Optional[str] = None,
        agent_ids: List[str] = None,
        agent_episode_ids: Optional[Dict[str, str]] = None,
        *,
        observations: Optional[List[MultiAgentDict]] = None,
        actions: Optional[List[MultiAgentDict]] = None,
        rewards: Optional[List[MultiAgentDict]] = None,
        states: Optional[List[MultiAgentDict]] = None,
        infos: Optional[List[MultiAgentDict]] = None,
        t_started: Optional[int] = None,
        is_terminated: Union[List[MultiAgentDict], bool] = False,
        is_truncated: Union[List[MultiAgentDict], bool] = False,
        render_images: Optional[List[np.ndarray]] = None,
        extra_model_outputs: Optional[List[MultiAgentDict]] = None,
        # TODO (simon): Validate terminated/truncated for env/agents.
    ) -> "MultiAgentEpisode":
        """Initializes a `MultiAgentEpisode`.

        Args:
            id_: Optional. Either a string to identify an episode or None.
                If None, a hexadecimal id is created. In case of providing
                a string, make sure that it is unique, as episodes get
                concatenated via this string.
            agent_ids: Obligatory. A list of strings containing the agent ids.
                These have to be provided at initialization.
            agent_episode_ids: Optional. Either a dictionary mapping agent ids
                corresponding `SingleAgentEpisode` or None. If None, each
                `SingleAgentEpisode` in `MultiAgentEpisode.agent_episodes`
                will generate a hexadecimal code. If a dictionary is provided
                make sure that ids are unique as agents'  `SingleAgentEpisode`s
                get concatenated or recreated by it.
            observations: A dictionary mapping from agent ids to observations.
                Can be None. If provided, it should be provided together with
                all other episode data (actions, rewards, etc.)
            actions: A dictionary mapping from agent ids to corresponding
                actions. Can be None. If provided, it should be provided
                together with all other episode data (observations, rewards,
                etc.).
            rewards: A dictionary mapping from agent ids to corresponding rewards.
                Can be None. If provided, it should be provided together with
                all other episode data (observations, rewards, etc.).
            infos: A dictionary mapping from agent ids to corresponding infos.
                Can be None. If provided, it should be provided together with
                all other episode data (observations, rewards, etc.).
            states: A dictionary mapping from agent ids to their corresponding
                modules' hidden states. These will be stored into the
                `SingleAgentEpisode`s in `MultiAgentEpisode.agent_episodes`.
                Can be None.
            t_started: Optional. An unsigned int that defines the starting point
                of the episode. This is only different from zero, if an ongoing
                episode is created.
            is_terminazted: Optional. A boolean defining, if an environment has
                terminated. The default is `False`, i.e. the episode is ongoing.
            is_truncated: Optional. A boolean, defining, if an environment is
                truncated. The default is `False`, i.e. the episode is ongoing.
            render_images: Optional. A list of RGB uint8 images from rendering
                the environment.
            extra_model_outputs: Optional. A dictionary mapping agent ids to their
                corresponding extra model outputs. Each of the latter is a list of
                dictionaries containing specific model outputs for the algorithm
                used (e.g. `vf_preds` and `action_logp` for PPO) from a rollout.
                If data is provided it should be complete (i.e. observations,
                actions, rewards, is_terminated, is_truncated, and all necessary
                `extra_model_outputs`).
        """

        self.id_: str = id_ or uuid.uuid4().hex

        # Agent ids must be provided if data is provided. The Episode cannot
        # know how many agents are in the environment. Also the number of agents
        # can grwo or shrink.
        self._agent_ids: Union[Set[str], Set[object]] = (
            set() if agent_ids is None else set(agent_ids)
        )

        # The global last timestep of the episode and the timesteps when this chunk
        # started.
        self.t = self.t_started = (
            t_started
            if t_started is not None
            else (len(observations) - 1 if observations is not None else 0)
            # t_started if t_started is not None else max(len(observations) - 1, 0)
        )
        # Keeps track of the correspondence between agent steps and environment steps.
        # This is a mapping from agents to `IndexMapping`. The latter keeps
        # track of the global timesteps at which an agent stepped.
        # Note, global (env) timesteps are values, while local (agent) steps are the
        # indices at which these global steps are recorded.
        self.global_t_to_local_t: Dict[str, List[int]] = self._generate_ts_mapping(
            observations
        )

        # In the `MultiAgentEPisode` we need these buffers to keep track of actions,
        # that happen when an agent got observations and acted, but did not receive
        # a next observation, yet. In this case we buffer the action, add the rewards,
        # and record `is_terminated/is_truncated` until the next observation is
        # received.
        self.agent_buffers = {
            agent_id: {
                "actions": Queue(maxsize=1),
                "rewards": Queue(maxsize=1),
                "states": Queue(maxsize=1),
                "extra_model_outputs": Queue(maxsize=1),
            }
            for agent_id in self._agent_ids
        }
        # Initialize buffers.
        for buffer in self.agent_buffers.values():
            # Default reward for accumulation is zero.
            buffer["rewards"].put_nowait(0.0)
            # Default state is None.
            buffer["states"].put_nowait(None)
            # Default extra_mdoel_output is None.
            buffer["extra_model_outputs"].put_nowait(None)

        # These are needed to reconstruct (see `self.get_rewards()`) reward
        # structures if agents received rewards without observations. This
        # is specific to multi-agent environemnts.
        # TODO (simon): implement into `get_rewards()`, `concat_episode`,
        # and `create_successor`. Add there also the buffers.
        self.partial_rewards = {agent_id: [] for agent_id in self._agent_ids}
        self.partial_rewards_t = {
            agent_id: _IndexMapping() for agent_id in self._agent_ids
        }

        # If this is an ongoing episode than the last `__all__` should be `False`
        self.is_terminated: bool = (
            is_terminated
            if isinstance(is_terminated, bool)
            else is_terminated[-1]["__all__"]
        )

        # If this is an ongoing episode than the last `__all__` should be `False`
        self.is_truncated: bool = (
            is_truncated
            if isinstance(is_truncated, bool)
            else is_truncated[-1]["__all__"]
        )

        # Note that all attributes will be recorded along the global timestep
        # in an multi-agent environment. `SingleAgentEpisodes`
        self.agent_episodes: MultiAgentDict = {
            agent_id: self._generate_single_agent_episode(
                agent_id,
                agent_episode_ids,
                observations,
                actions,
                rewards,
                infos,
                is_terminated,
                is_truncated,
                states,
                extra_model_outputs,
            )
            for agent_id in self._agent_ids
        }

        # RGB uint8 images from rendering the env; the images include the corresponding
        # rewards.
        assert render_images is None or observations is not None
        self.render_images: Union[List[np.ndarray], List[object]] = (
            [] if render_images is None else render_images
        )

    def concat_episode(self, episode_chunk: "MultiAgentEpisode") -> None:
        """Adds the given `episode_chunk` to the right side of self.

        For concatenating episodes the following rules hold:
            - IDs are identical.
            - timesteps match (`t` of `self` matches `t_started` of `episode_chunk`).

        Args:
            episode_chunk: `MultiAgentEpsiode` instance that should be concatenated
                to `self`.
        """
        assert episode_chunk.id_ == self.id_
        assert not self.is_done
        # Make sure the timesteps match.
        assert self.t == episode_chunk.t_started

        # TODO (simon): Write `validate()` method.

        # Make sure, end matches `episode_chunk`'s beginning for all agents.
        observations: MultiAgentDict = self.get_observations()
        for agent_id, agent_obs in episode_chunk.get_observations(indices=0):
            # Make sure that the same agents stepped at both timesteps.
            assert agent_id in observations
            assert observations[agent_id] == agent_obs
        # Pop out the end for the agents that stepped.
        for agent_id in observations:
            self.agent_episodes[agent_id].observations.pop()

        # Call the `SingleAgentEpisode`'s `concat_episode()` method for all agents.
        for agent_id, agent_eps in self.agent_episodes:
            agent_eps[agent_id].concat_episode(episode_chunk.agent_episodes[agent_id])
            # Update our timestep mapping.
            # TODO (simon): Check, if we have to cut off here as well.
            self.global_t_to_local_t[agent_id][
                :-1
            ] += episode_chunk.global_t_to_local_t[agent_id]

        self.t = episode_chunk.t
        if episode_chunk.is_terminated:
            self.is_terminated = True
        if episode_chunk.is_truncated:
            self.is_truncated = True

        # Validate
        # TODO (simon): Write validate function.
        # self.validate()

    # TODO (simon): Maybe adding agent axis. We might need only some agent observations.
    # Then also add possibility to get __all__ obs (or None)
    # Write many test cases (numbered obs).
    def get_observations(
        self,
        indices: Union[int, List[int]] = -1,
        global_ts: bool = True,
        as_list: bool = False,
    ) -> Union[MultiAgentDict, List[MultiAgentDict]]:
        """Gets observations for all agents that stepped in the last timesteps.

        Note that observations are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the observations
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to observations (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, observations are returned (i.e. not all agent ids are
            necessarily in the keys).
        """
        return self._getattr_by_index(
            "observations",
            indices,
            has_initial_value=True,
            global_ts=global_ts,
            as_list=as_list,
        )

    def get_actions(
        self,
        indices: Union[int, List[int]] = -1,
        global_ts: bool = True,
        as_list: bool = False,
    ) -> Union[MultiAgentDict, List[MultiAgentDict]]:
        """Gets actions for all agents that stepped in the last timesteps.

        Note that actions are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the actions
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to actions (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, actions are returned (i.e. not all agent ids are
            necessarily in the keys).
        """

        return self._getattr_by_index("actions", indices, global_ts, as_list)

    def get_rewards(
        self,
        indices: Union[int, List[int]] = -1,
        global_ts: bool = True,
        as_list: bool = False,
    ) -> Union[MultiAgentDict, List[MultiAgentDict]]:
        """Gets rewards for all agents that stepped in the last timesteps.

        Note that rewards are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the rewards
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to rewards (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, rewards are returned (i.e. not all agent ids are
            necessarily in the keys).
        """
        # TODO (simon): Add partial rewards.
        return self._getattr_by_index("rewards", indices, global_ts, as_list)

    def get_infos(
        self,
        indices: Union[int, List[int]] = -1,
        global_ts: bool = True,
        as_list: bool = False,
    ) -> Union[MultiAgentDict, List[MultiAgentDict]]:
        """Gets infos for all agents that stepped in the last timesteps.

        Note that infos are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the infos
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to infos (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, infos are returned (i.e. not all agent ids are
            necessarily in the keys).
        """
        return self._getattr_by_index(
            "infos",
            indices,
            has_initial_value=True,
            global_ts=global_ts,
            as_list=as_list,
        )

    def get_terminateds(self) -> MultiAgentDict:
        """Gets the terminateds at given indices."""
        terminateds = {
            agent_id: self.agent_episodes[agent_id].is_terminated
            for agent_id in self._agent_ids
        }
        terminateds.update({"__all__": self.is_terminated})
        return terminateds

    def get_truncateds(self) -> MultiAgentDict:
        truncateds = {
            agent_id: self.agent_episodes[agent_id].is_truncated
            for agent_id in self._agent_ids
        }
        truncateds.update({"__all__": self.is_terminated})
        return truncateds

    def get_states(self) -> MultiAgentDict:
        """Gets all agent states."""
        return {
            agent_id: self.agent_episodes[agent_id].states
            for agent_id in self._agent_ids
        }

    def get_extra_model_outputs(
        self, indices: Union[int, List[int]] = -1, global_ts: bool = True
    ) -> Union[MultiAgentDict, List[MultiAgentDict]]:
        """Gets extra model outputs for all agents that stepped in the last timesteps.

        Note that extra model outputs are only returned for agents that stepped
        during the given index range.

        Args:
            indices: Either a single index or a list of indices. The indices
                can be reversed (e.g. [-1, -2]) or absolute (e.g. [98, 99]).
                This defines the time indices for which the extra model outputs.
                should be returned.
            global_ts: Boolean that defines, if the indices should be considered
                environment (`True`) or agent (`False`) steps.

        Returns: A dictionary mapping agent ids to extra model outputs (of different
            timesteps). Only for agents that have stepped (were ready) at a
            timestep, extra model outputs are returned (i.e. not all agent ids are
            necessarily in the keys).
        """
        return self._getattr_by_index("extra_model_outputs", indices, global_ts)

    def add_initial_observation(
        self,
        *,
        initial_observation: MultiAgentDict,
        initial_info: Optional[MultiAgentDict] = None,
        initial_state: Optional[MultiAgentDict] = None,
        initial_render_image: Optional[np.ndarray] = None,
    ) -> None:
        """Stores initial observation.

        Args:
            initial_observation: Obligatory. A dictionary, mapping agent ids
                to initial observations. Note that not all agents must have
                an initial observation.
            initial_info: Optional. A dictionary, mapping agent ids to initial
                infos. Note that not all agents must have an initial info.
            initial_state: Optional. A dictionary, mapping agent ids to the
                initial hidden states of their corresponding model (`RLModule`).
                Note, this is only available, if the models are stateful. Note
                also that not all agents must have an initial state at `t=0`.
            initial_render_image: An RGB uint8 image from rendering the
                environment.
        """
        assert not self.is_done
        # Assume that this episode is completely empty and has not stepped yet.
        # Leave self.t (and self.t_started) at 0.
        assert self.t == self.t_started == 0

        # TODO (simon): After clearing with sven for initialization of timesteps
        # this might be removed.
        if len(self.global_t_to_local_t) == 0:
            self.global_t_to_local_t = {
                agent_id: _IndexMapping() for agent_id in self._agent_ids
            }

        # Note that we store the render images into the `MultiAgentEpisode`
        # instead into each `SingleAgentEpisode`.
        if initial_render_image is not None:
            self.render_images.append(initial_render_image)

        # Note, all agents will have an initial observation.
        for agent_id in initial_observation.keys():
            # Add initial timestep for each agent to the timestep mapping.
            self.global_t_to_local_t[agent_id].append(self.t)
            # Add initial observations to the agent's episode.
            self.agent_episodes[agent_id].add_initial_observation(
                # Note, initial observation has to be provided.
                initial_observation=initial_observation[agent_id],
                initial_info=None if initial_info is None else initial_info[agent_id],
                initial_state=None
                if initial_state is None
                else initial_state[agent_id],
            )

    def add_timestep(
        self,
        observation: MultiAgentDict,
        action: MultiAgentDict,
        reward: MultiAgentDict,
        *,
        info: Optional[MultiAgentDict] = None,
        state: Optional[MultiAgentDict] = None,
        is_terminated: Optional[bool] = None,
        is_truncated: Optional[bool] = None,
        render_image: Optional[np.ndarray] = None,
        extra_model_output: Optional[MultiAgentDict] = None,
    ) -> None:
        """Adds a timestep to the episode.

        Args:
            observation: Mandatory. A dictionary, mapping agent ids to their
                corresponding observations. Note that not all agents must have stepped
                a this timestep.
            action: Mandatory. A dictionary, mapping agent ids to their
                corresponding actions. Note that not all agents must have stepped
                a this timestep.
            reward: Mandatory. A dictionary, mapping agent ids to their
                corresponding observations. Note that not all agents must have stepped
                a this timestep.
            info: Optional. A dictionary, mapping agent ids to their
                corresponding info. Note that not all agents must have stepped
                a this timestep.
            state: Optional. A dictionary, mapping agent ids to their
                corresponding hidden model state. Note, this is only available for a
                stateful model. Also note that not all agents must have stepped a this
                timestep.
            is_terminated: A boolean indicating, if the environment has been
                terminated.
            is_truncated: A boolean indicating, if the environment has been
                truncated.
            render_image: Optional. An RGB uint8 image from rendering the environment.
            extra_model_output: Optional. A dictionary, mapping agent ids to their
                corresponding specific model outputs (also in a dictionary; e.g.
                `vf_preds` for PPO).
        """
        # Cannot add data to an already done episode.
        assert not self.is_done

        # Environment step.
        self.t += 1

        # TODO (sven, simon): Wilol there still be an `__all__` that is
        # terminated or truncated?
        # TODO (simon): Maybe allow user to not provide this and then all is False?
        self.is_terminated = (
            False if is_terminated is None else is_terminated["__all__"]
        )
        self.is_truncated = False if is_truncated is None else is_truncated["__all__"]

        # Note that we store the render images into the `MultiAgentEpisode`
        # instead of storing them into each `SingleAgentEpisode`.
        if render_image is not None:
            self.render_images.append(render_image)

        # Add data to agent episodes.
        for agent_id in self._agent_ids:
            # Skip agents that have been terminated or truncated.
            if self.agent_episodes[agent_id].is_done:
                continue

            agent_is_terminated = (
                False if agent_id not in is_terminated else is_terminated[agent_id]
            ) or is_terminated["__all__"]
            agent_is_truncated = (
                False if agent_id not in is_truncated else is_truncated[agent_id]
            ) or is_truncated["__all__"]

            # CASE 1: observation, no action.
            # If we have an observation, but no action, we might have a buffered action,
            # or an initial agent observation.
            if agent_id in observation and agent_id not in action:
                # We have a buffered action.
                if self.agent_buffers[agent_id]["actions"].full():
                    # Get the action from the buffer.
                    agent_action = self.agent_buffers[agent_id]["actions"].get_nowait()
                    # Get the extra model output if available.
                    agent_extra_model_output = self.agent_buffers[agent_id][
                        "extra_model_outputs"
                    ].get_nowait()
                    # Reset the buffer to default value.
                    self.agent_buffers[agent_id]["extra_model_outputs"].put_nowait(None)
                    # Get the agent's last state from the buffer. Note this buffer's
                    # default is `None`.
                    agent_state = self.agent_buffers[agent_id]["states"].get_nowait()
                    # Reset the state buffer to its default value.
                    self.agent_buffers[agent_id]["states"].put_nowait(None)

                    # Note, the reward buffer will be always full with at least the
                    # default of zero reward.
                    agent_reward = self.agent_buffers[agent_id]["rewards"].get_nowait()
                    # We might also got some reward in this episode.
                    if agent_id in reward:
                        agent_reward += reward[agent_id]
                        # Also add to the global reward list.
                        self.partial_rewards[agent_id].append(reward[agent_id])
                        # And add to the global reward timestep mapping.
                        self.partial_rewards_t[agent_id].append(self.t)

                    # Refill the reward buffer with the default value of zero.
                    self.agent_buffers[agent_id]["rewards"].put_nowait(0.0)

                    # It could be the last timestep of the agent.
                    # Note, in this case the agent is in `is_terminated` and
                    # `is_truncated` b/c it has an observation.
                    if agent_is_terminated or agent_is_truncated:
                        # Then, flush the buffers.
                        # TODO (simon): You can simply not refill them above.
                        self.agent_buffers[agent_id]["rewards"].get_nowait()
                        self.agent_buffers[agent_id]["states"].get_nowait()
                        self.agent_buffers[agent_id]["extra_model_outputs"].get_nowait()
                    # If the agent stepped we need to keep track in the timestep
                    # mapping.
                    self.global_t_to_local_t[agent_id].append(self.t)
                    # Add data to `SingleAgentEpisode.
                    self.agent_episodes[agent_id].add_timestep(
                        observation=observation[agent_id],
                        action=agent_action,
                        reward=agent_reward,
                        info=None if agent_id not in info else info[agent_id],
                        state=agent_state,
                        is_terminated=agent_is_terminated,
                        is_truncated=agent_is_truncated,
                        extra_model_output=agent_extra_model_output,
                    )
                # We have no buffered action.
                else:
                    # The agent might have been terminated/truncated.
                    # if agent_id in is_terminated or agent_id in is_truncated:
                    if agent_is_terminated or agent_is_truncated:
                        # If the agent has never stepped, we treat it as not being
                        # part of this episode.
                        # Delete all of the agent's registers.
                        # del self._agent_ids[self._agent_ids.index(agent_id)]
                        self._agent_ids.discard(agent_id)
                        del self.agent_episodes[agent_id]
                        del self.agent_buffers[agent_id]
                        del self.global_t_to_local_t[agent_id]
                        # Then continue with the next agent.
                        continue
                    # Then this must be the agent's initial observation.
                    else:
                        # If this was the agent's first step, record the step in the
                        # global timestep mapping.
                        self.global_t_to_local_t[agent_id].append(self.t)
                        # TODO (sven, simon): An agent could even have an accumulated
                        # reward over all the timesteps it did not act or receive
                        # observations.
                        self.agent_episodes[agent_id].add_initial_observation(
                            initial_observation=observation[agent_id],
                            initial_info=None
                            if agent_id not in info
                            else info[agent_id],
                            initial_state=None
                            if agent_id not in state
                            else info[agent_id],
                        )
            # CASE 2: No observation, but action.
            # We have no observation, but we have an action. This must be an orphane
            # action and we need to buffer it.
            elif agent_id not in observation and agent_id in action:
                # Maybe the agent got terminated.
                # if agent_id in is_terminated or agent_id in is_truncated:
                if agent_is_terminated or agent_is_truncated:
                    # If this was indeed the agent's last step, we need to record it
                    # in the timestep mapping.
                    self.global_t_to_local_t[agent_id].append(self.t)
                    # Also flush all default values from the buffers.
                    self.agent_buffers[agent_id]["rewards"].get_nowait()
                    self.agent_buffers[agent_id]["states"].get_nowait()
                    self.agent_buffers[agent_id]["extra_model_outputs"].get_nowait()
                    # If the agent was terminated and no observation is provided,
                    # take the last one.
                    self.agent_episodes[agent_id].add_timestep(
                        observation=self.agent_episodes[agent_id].observations[-1],
                        action=action[agent_id],
                        reward=0.0 if agent_id not in reward else reward[agent_id],
                        state=None if agent_id not in state else state[agent_id],
                        is_terminated=agent_is_terminated,
                        is_truncated=agent_is_truncated,
                        extra_model_output=None
                        if agent_id not in extra_model_output
                        else extra_model_output[agent_id],
                    )
                # Agent is still alive.
                else:
                    # TODO (simon): Maybe add a shift mapping that keeps track on
                    # original action timestep (global one). Right now the
                    # `global_reward_t` might serve here.
                    # Buffer the action.
                    self.agent_buffers[agent_id]["actions"].put_nowait(action[agent_id])
                    # If available, buffer also reward. Note, if the agent is terminated
                    # or truncated, we finish the `SingleAgentEpisode`.
                    if agent_id in reward:
                        # Add the reward to the existing one in the buffer. Note, the
                        # default value is zero.
                        self.agent_buffers[agent_id]["rewards"].put_nowait(
                            self.agent_buffers[agent_id]["rewards"].get_nowait()
                            + reward[agent_id]
                        )
                        # Add to the global reward list.
                        self.partial_rewards[agent_id].append(reward[agent_id])
                        # Add also to the global reward timestep mapping.
                        self.partial_rewards_t[agent_id].append(self.t)
                    # If not available set reward to zero.
                    # TODO (simon): Check, if this is okay for training.
                    # else:
                    #     self.agent_buffers[agent_id]["rewards"].put_nowait(0.0)
                    # If the agent got a state buffer that one, too.
                    if state and agent_id in state:
                        # Flush the default `None` from buffer.
                        self.agent_buffers[agent_id]["states"].get_nowait()
                        # Store the hidden state into the buffer.
                        self.agent_buffers[agent_id]["states"].put_nowait(
                            state[agent_id]
                        )
                    # If the agent got any extra model outputs, buffer them, too.
                    if extra_model_output and agent_id in extra_model_output:
                        # Flush the default `None` from buffer.
                        self.agent_buffers[agent_id]["extra_model_outputs"].get_nowait()
                        # STore the extra model outputs into the buffer.
                        self.agent_buffers[agent_id]["extra_model_outputs"].put_nowait(
                            extra_model_output[agent_id]
                        )

            # CASE 3: No observation and no action.
            # We have neither observation nor action. Then, we could have `reward`,
            # `is_terminated` or `is_truncated` and should record it.
            elif agent_id not in observation and agent_id not in action:
                # The agent could be is_terminated
                if agent_is_terminated or agent_is_truncated:
                    # If the agent has never stepped, we treat it as not being
                    # part of this episode.
                    if len(self.agent_episodes[agent_id].observations) == 0:
                        # Delete all of the agent's registers.
                        # del self._agent_ids[self._agent_ids.index(agent_id)]
                        self._agent_ids.discard(agent_id)
                        del self.agent_episodes[agent_id]
                        del self.agent_buffers[agent_id]
                        del self.global_t_to_local_t[agent_id]
                        # Then continue with the next agent.
                        continue

                    # If no observation and no action is available and the agent had
                    # stepped before the buffer must be full b/c after each
                    # observation the agent does step and if no observation followed
                    # to write the record away, the action gets buffered.
                    agent_action = self.agent_buffers[agent_id]["actions"].get_nowait()
                    # Note, this is initialized as `None` if agents do not have
                    # extra model outputs.
                    agent_extra_model_output = self.agent_buffers[agent_id][
                        "extra_model_outputs"
                    ].get_nowait()
                    # Note, also the state is initialized as `None` and therefore
                    # always available.
                    agent_state = self.agent_buffers[agent_id]["states"].get_nowait()
                    # Get the reward from the buffer. Note, this is always available
                    # as it is initialized as a zero reward.
                    agent_reward = self.agent_buffers[agent_id]["rewards"].get_nowait()
                    # If a reward is received at this timestep record it.
                    if agent_id in reward:
                        agent_reward += reward[agent_id]
                        # Add to the global reward list.
                        self.partial_rewards[agent_id].append(reward[agent_id])
                        # Add also to the global reward timestep mapping.
                        self.partial_rewards_t[agent_id].append(self.t)

                    # If this was indeed the agent's last step, we need to record
                    # it in the timestep mapping.
                    self.global_t_to_local_t[agent_id].append(self.t)
                    # Finish the agent's episode.
                    self.agent_episodes[agent_id].add_timestep(
                        observation=self.agent_episodes[agent_id].observations[-1],
                        action=agent_action,
                        reward=agent_reward,
                        info=None if agent_id not in info else info[agent_id],
                        state=agent_state,
                        is_terminated=agent_is_terminated,
                        is_truncated=agent_is_truncated,
                        extra_model_output=agent_extra_model_output,
                    )
                # The agent is still alive.
                else:
                    # TODO (sven, simon): Check, if this should stay like this. This
                    # enables agents that have never stepped to collect rewards.

                    # First check, if the agent had already an intiial observation.
                    # if len(self.agent_episodes[agent_id].observations) > 0:

                    # If the agent received an reward (triggered by actions of
                    # other agents) we collect it and add it to the one in the
                    # buffer.
                    # TODO (sven, simon): Agents could have had already rewards
                    # before the their initial observation, .e.g. cooperation
                    # games.
                    if agent_id in reward:
                        self.agent_buffers[agent_id]["rewards"].put_nowait(
                            self.agent_buffers[agent_id]["rewards"].get_nowait()
                            + reward[agent_id]
                        )
                        # Add to the global reward list.
                        self.partial_rewards[agent_id].append(reward[agent_id])
                        # Add also to the global reward timestep mapping.
                        self.partial_rewards_t[agent_id].append(self.t)
            # CASE 4: Observation and action.
            # We have an observation and an action. Then we can simply add the
            # complete information to the episode.
            else:
                # In this case the agent id is also in the `is_terminated` and
                # `is_truncated` dictionaries.
                if agent_is_terminated or agent_is_truncated:
                    # If the agent is also done in this timestep, flush the buffers.
                    self.agent_buffers[agent_id]["rewards"].get_nowait()
                    self.agent_buffers[agent_id]["states"].get_nowait()
                    self.agent_buffers[agent_id]["extra_model_outputs"].get_nowait()
                # If the agent stepped we need to keep track in the timestep mapping.
                self.global_t_to_local_t[agent_id].append(self.t)
                # Also add to the global reward list.
                self.partial_rewards[agent_id].append(reward[agent_id])
                # And add to the global reward timestep mapping.
                self.partial_rewards_t[agent_id].append(self.t)
                # Add timestep to `SingleAgentEpisode`.
                self.agent_episodes[agent_id].add_timestep(
                    observation[agent_id],
                    action[agent_id],
                    reward[agent_id],
                    info=None if agent_id not in info else info[agent_id],
                    state=None if state is None else state[agent_id],
                    # TODO (simon): Check, if case checking is necessary here.
                    # This would even insure against user errors in the env.
                    is_terminated=agent_is_terminated,
                    is_truncated=agent_is_truncated,
                    extra_model_output=None
                    if extra_model_output is None
                    else extra_model_output[agent_id],
                )

    @property
    def is_done(self):
        """Whether the episode is actually done (terminated or truncated).

        A done episode cannot be continued via `self.add_timestep()` or being
        concatenated on its right-side with another episode chunk or being
        succeeded via `self.create_successor()`.

        Note that in a multi-agent environment this does not necessarily
        correspond to single agents having terminated or being truncated.

        `self.is_terminated` should be `True`, if all agents are terminated and
        `self.is_truncated` should be `True`, if all agents are truncated. If
        only one or more (but not all!) agents are `terminated/truncated the
        `MultiAgentEpisode.is_terminated/is_truncated` should be `False`. This
        information about single agent's terminated/truncated states can always
        be retrieved from the `SingleAgentEpisode`s inside the 'MultiAgentEpisode`
        one.

        If all agents are either terminated or truncated, but in a mixed fashion,
        i.e. some are terminated and others are truncated: This is currently
        undefined and could potentially be a problem (if a user really implemented
        such a multi-agent env that behaves this way).

        Returns:
            Boolean defining if an episode has either terminated or truncated.
        """
        return self.is_terminated or self.is_truncated

    def create_successor(self) -> "MultiAgentEpisode":
        """Restarts an ongoing episode from its last observation.

        Note, this method is used so far specifically for the case of
        `batch_mode="truncated_episodes"` to ensure that episodes are
        immutable inside the `EnvRunner` when truncated and passed over
        to postprocessing.

        The newly created `MultiAgentEpisode` contains the same id, and
        starts at the timestep where it's predecessor stopped in the last
        rollout. Last observations, infos, rewards, etc. are carried over
        from the predecessor. This also helps to not carry stale data that
        had been collected in the last rollout when rolling out the new
        policy in the next iteration (rollout).

        Returns: A MultiAgentEpisode starting at the timepoint where
            its predecessor stopped.
        """
        assert not self.is_done

        # Get the last multi-agent observation and info.
        observations = self.get_observations(as_list=True)
        infos = self.get_infos(as_list=True)
        states = self.get_states()
        is_terminateds = self.get_terminateds()
        is_truncateds = self.get_truncateds()
        # It is more safe to use here a list of episode ids instead of
        # calling `create_successor()` as we need as the single source
        # of truth always the `global_t_to_local_t` timestep mapping.
        # TODO (sven, simon): Maybe we better create an episode with only the
        # agent that are still alive.
        successor = MultiAgentEpisode(
            id_=self.id_,
            agent_ids=self._agent_ids,
            agent_episode_ids={
                agent_id: agent_eps.id_
                for agent_id, agent_eps in self.agent_episodes.items()
            },
            observations=observations,
            infos=infos,
            is_terminated=[is_terminateds],
            is_truncated=[is_truncateds],
            states=states,
            t_started=self.t,
        )
        # Now add the buffers.
        # TODO (simon): Refactor to helper function
        successor.agent_buffers = self.agent_buffers.copy()
        # for agent_id, agent_rewards in self.global_rewards.items():
        #     indices_to_keep = self.global_rewards_t[
        #         agent_id
        #     ].find_indices_with_greater_values(self.global_t_to_local_t[agent_id][-1])
        #     # indices_to_keep = [
        #     #     self.global_rewards_t[agent_id].index(i)
        #     #     for i in self.global_t_to_local_t[agent_id]
        #     #     if i > self.global_t_to_local_t[agent_id][-1]
        #     # ]
        #     successor.global_rewards_t[agent_id] = _IndexMapping(
        #         map(self.global_rewards_t[agent_id].__getitem__, indices_to_keep)
        #     )
        #     successor.global_rewards[agent_id] = list(
        #         map(agent_rewards.__getitem__, indices_to_keep)
        #     )
        (
            successor.partial_rewards_t,
            successor.partial_rewards,
        ) = self._generate_agent_rewards()

        return successor

    def _generate_agent_rewards(self):
        successor_global_rewards_t = {}
        successor_global_rewards = {}
        for agent_id, agent_global_rewards in self.partial_rewards.items():
            if self.global_t_to_local_t[agent_id]:
                indices_to_keep = self.partial_rewards_t[
                    agent_id
                ].find_indices_with_greater_values(
                    self.global_t_to_local_t[agent_id][-1]
                )
            else:
                if self.partial_rewards_t[agent_id]:
                    indices_to_keep = list(range(len(self.partial_rewards_t[agent_id])))
                else:
                    indices_to_keep = []

            successor_global_rewards_t[agent_id] = _IndexMapping(
                map(self.partial_rewards_t[agent_id].__getitem__, indices_to_keep)
            )
            successor_global_rewards[agent_id] = list(
                map(agent_global_rewards.__getitem__, indices_to_keep)
            )

        return successor_global_rewards_t, successor_global_rewards

    def get_state(self) -> Dict[str, Any]:
        """Returns the state of a multi-agent episode.

        Note that from an episode's state the episode itself can
        be recreated.

        Returns: A dicitonary containing pickable data fro a
            `MultiAgentEpisode`.
        """
        return list(
            {
                "id_": self.id_,
                "agent_ids": self._agent_ids,
                "global_t_to_local_t": self.global_t_to_local_t,
                "agent_episodes": list(
                    {
                        agent_id: agent_eps.get_state()
                        for agent_id, agent_eps in self.agent_episodes.items()
                    }.items()
                ),
                "t_started": self.t_started,
                "t": self.t,
                "is_terminated": self.is_terminated,
                "is_truncated": self.is_truncated,
            }.items()
        )

    @staticmethod
    def from_state(state) -> None:
        """Creates a multi-agent episode from a state dictionary.

        See `MultiAgentEpisode.get_state()` for creating a state for
        a `MultiAgentEpisode` pickable state. For recreating a
        `MultiAgentEpisode` from a state, this state has to be complete,
        i.e. all data must have been stored in the state.
        """
        eps = MultiAgentEpisode(id=state[0][1])
        eps._agent_ids = state[1][1]
        eps.global_t_to_local_t = state[2][1]
        eps.agent_episodes = {
            agent_id: SingleAgentEpisode.from_state(agent_state)
            for agent_id, agent_state in state[3][1]
        }
        eps.t_started = state[3][1]
        eps.t = state[4][1]
        eps.is_terminated = state[5][1]
        eps.is_trcunated = state[6][1]
        return eps

    def to_sample_batch(self) -> MultiAgentBatch:
        """Converts a `MultiAgentEpisode` into a `MultiAgentBatch`.

        Each `SingleAgentEpisode` instances in
        `MultiAgentEpisode.agent_epiosdes` will be converted into
        a `SampleBatch` and the environment timestep will be passed
        towards the `MultiAgentBatch`'s `count`.

        Returns: A `MultiAgentBatch` instance.
        """
        # TODO (simon): Check, if timesteps should be converted into global
        # timesteps instead of agent steps.
        return MultiAgentBatch(
            policy_batches={
                agent_id: agent_eps.to_sample_batch()
                for agent_id, agent_eps in self.agent_episodes.items()
            },
            env_steps=self.t,
        )

    def get_return(self) -> float:
        """Get the all-agent return.

        Returns: A float. The aggregate return from all agents.
        """
        # TODO (simon): Also include the partial rewards.
        return sum(
            [agent_eps.get_return() for agent_eps in self.agent_episodes.values()]
        )

    def _generate_ts_mapping(
        self, observations: List[MultiAgentDict]
    ) -> MultiAgentDict:
        """Generates a timestep mapping to local agent timesteps.

        This helps us to keep track of which agent stepped at
        which global (environment) timestep.
        Note that the local (agent) timestep is given by the index
        of the list for each agent.

        Args:
            observations: A list of observations.Each observations maps agent
                ids to their corresponding observation.

        Returns: A dictionary mapping agents to time index lists. The latter
            contain the global (environment) timesteps at which the agent
            stepped (was ready).
        """
        # Only if agent ids have been provided we can build the timestep mapping.
        if len(self._agent_ids) > 0:
            global_t_to_local_t = {agent: _IndexMapping() for agent in self._agent_ids}

            # We need the observations to create the timestep mapping.
            if observations:
                for t, agent_map in enumerate(observations):
                    for agent_id in agent_map:
                        # If agent stepped add the timestep to the timestep mapping.
                        global_t_to_local_t[agent_id].append(t)
            # Otherwise, set to an empoty dict (when creating an empty episode).
            else:
                global_t_to_local_t = {}
        # Otherwise, set to an empoty dict (when creating an empty episode).
        else:
            # TODO (sven, simon): Shall we return an empty dict or an agent dict with
            # empty lists?
            global_t_to_local_t = {}
        # Return the index mapping.
        return global_t_to_local_t

    # TODO (sven, simon): This function can only deal with data if it does not contain
    # terminated or truncated agents (i.e. you have to provide ONLY alive agents in the
    # agent_ids in the constructor - the episode does not deduce the agents).
    def _generate_single_agent_episode(
        self,
        agent_id: str,
        agent_episode_ids: Optional[Dict[str, str]] = None,
        observations: Optional[List[MultiAgentDict]] = None,
        actions: Optional[List[MultiAgentDict]] = None,
        rewards: Optional[List[MultiAgentDict]] = None,
        infos: Optional[List[MultiAgentDict]] = None,
        is_terminateds: Union[MultiAgentDict, bool] = False,
        is_truncateds: Union[MultiAgentDict, bool] = False,
        states: Optional[MultiAgentDict] = None,
        extra_model_outputs: Optional[MultiAgentDict] = None,
    ) -> SingleAgentEpisode:
        """Generates a `SingleAgentEpisode` from multi-agent data.

        Note, if no data is provided an empty `SingleAgentEpiosde`
        will be returned that starts at `SIngleAgentEpisode.t_started=0`.

        Args:
            agent_id: String, idnetifying the agent for which the data should
                be extracted.
            agent_episode_ids: Optional. A dictionary mapping agents to
                corresponding episode ids. If `None` the `SingleAgentEpisode`
                creates a hexadecimal code.
            observations: Optional. A list of dictionaries, each mapping
                from agent ids to observations. When data is provided
                it should be complete, i.e. observations, actions, rewards,
                etc. should be provided.
            actions: Optional. A list of dictionaries, each mapping
                from agent ids to actions. When data is provided
                it should be complete, i.e. observations, actions, rewards,
                etc. should be provided.
            rewards: Optional. A list of dictionaries, each mapping
                from agent ids to rewards. When data is provided
                it should be complete, i.e. observations, actions, rewards,
                etc. should be provided.
            infos: Optional. A list of dictionaries, each mapping
                from agent ids to infos. When data is provided
                it should be complete, i.e. observations, actions, rewards,
                etc. should be provided.
            states: Optional. A dicitionary mapping each agent to it's
                module's hidden model state (if the model is stateful).
            extra_model_outputs: Optional. A list of agent mappings for every
                timestep. Each of these dictionaries maps an agent to its
                corresponding `extra_model_outputs`, which a re specific model
                outputs needed by the algorithm used (e.g. `vf_preds` and
                `action_logp` for PPO). f data is provided it should be complete
                (i.e. observations, actions, rewards, is_terminated, is_truncated,
                and all necessary `extra_model_outputs`).

        Returns: An instance of `SingleAgentEpisode` containing the agent's
            extracted episode data.
        """

        # If an episode id for an agent episode was provided assign it.
        episode_id = None if agent_episode_ids is None else agent_episode_ids[agent_id]
        # We need the timestep mapping to create single agent's episode.
        if len(self.global_t_to_local_t) > 0:
            # Set to None if not provided.
            agent_observations = (
                None
                if observations is None
                else self._get_single_agent_data(agent_id, observations)
            )

            # Note, the timestep mapping is deduced from observations and starts one
            # timestep earlier. Therefore all other data is missing the last index.
            agent_actions = (
                None
                if actions is None
                else self._get_single_agent_data(
                    #    agent_id, actions, start_index=1, shift=-1
                    agent_id,
                    actions,
                    use_global_to_to_local_t=False,
                )
            )

            # Rewards are special in multi-agent scenarios, as agents could receive a
            # reward even though they did not get an observation at a specific timestep.
            agent_rewards = (
                None
                if rewards is None
                else self._get_single_agent_data(
                    agent_id,
                    rewards,
                    use_global_to_to_local_t=False,
                    start_index=1,
                    shift=-1,
                )
            )
            # Like observations, infos start at timestep `t=0`, so we do not need to
            # shift or start later.
            agent_infos = (
                None if infos is None else self._get_single_agent_data(agent_id, infos)
            )
            agent_states = (
                None if states is None or agent_id not in states else states[agent_id]
            )
            agent_extra_model_outputs = (
                None
                if extra_model_outputs is None
                else self._get_single_agent_data(
                    agent_id,
                    extra_model_outputs,
                    use_global_to_to_local_t=False,
                    end_index=-1,
                    shift=0,
                )
            )

            agent_is_terminated = (
                [False]
                if is_terminateds is None
                else self._get_single_agent_data(
                    agent_id, is_terminateds, start_index=1, shift=-1
                )
            )
            # If a list the list could be empty, if the agent never stepped.
            agent_is_terminated = (
                False if not agent_is_terminated else agent_is_terminated[-1]
            )

            agent_is_truncated = (
                [False]
                if is_truncateds is None
                else self._get_single_agent_data(
                    agent_id, is_truncateds, start_index=1, shift=-1
                )
            )
            # If a list the list could be empty, if the agent never stepped.
            agent_is_truncated = (
                False if not agent_is_truncated else agent_is_truncated[-1]
            )

            # If there are as many actions as observations we have to buffer.
            if (
                agent_actions
                and agent_observations
                and len(agent_observations) == len(agent_actions)
            ):
                # Assert then that the other data is in order.
                if agent_states:
                    # Put the last states into the buffer.
                    self.agent_buffers[agent_id]["states"].get_nowait()
                    self.agent_buffers[agent_id]["states"].put_nowait(agent_states)
                if agent_extra_model_outputs:
                    assert len(agent_extra_model_outputs) == len(
                        agent_actions
                    ), f"Agent {agent_id} has not as many extra model outputs as "
                    "actions."
                    # Put the last extra model outputs into the buffer.
                    self.agent_buffers[agent_id]["extra_model_outputs"].get_nowait()
                    self.agent_buffers[agent_id]["extra_model_outputs"].put_nowait(
                        agent_extra_model_outputs.pop()
                    )

                # Put the last action into the buffer.
                self.agent_buffers[agent_id]["actions"].put_nowait(agent_actions.pop())

            # Now check, if there had been rewards for the agent without him acting
            # or receiving observations.
            if (agent_rewards and not observations) or (
                agent_rewards
                and observations
                and len(agent_rewards) >= len(agent_observations)
            ):
                # In this case we have to make use of the timestep mapping.
                # Note, this mapping does then only contain the rewards that are not
                # stored in the agent's `SingleAgentEpisode`, i.e. the rewards
                # received in between to observations, but not with the observations.
                global_rewards_t = _IndexMapping()
                global_agent_rewards = []
                agent_rewards = []
                agent_reward = 0.0
                for t, reward in enumerate(rewards):
                    if agent_id in reward:
                        # TODO (simon): This has to be put into the
                        # if clause below. There is some inconsistency as here
                        # we add all rewards not only partial ones and in
                        # add_timestep and create_successor we use only
                        # partial ones.
                        global_agent_rewards.append(reward[agent_id])
                        # Then add the reward.
                        agent_reward += reward[agent_id]
                        global_rewards_t.append(t)
                        if t in self.global_t_to_local_t[agent_id]:
                            agent_rewards.append(agent_reward)
                            agent_reward = 0.0
                            continue
                # If the agent reward is not zero, we must have rewards that came
                # after the last observation. Then we buffer this reward.
                self.agent_buffers[agent_id]["rewards"].put_nowait(
                    self.agent_buffers[agent_id]["rewards"].get_nowait() + agent_reward
                )
                # Now save away the original rewards and the reward timesteps.
                self.partial_rewards_t[agent_id] = global_rewards_t
                self.partial_rewards[agent_id] = global_agent_rewards

            return SingleAgentEpisode(
                id_=episode_id,
                observations=agent_observations,
                actions=agent_actions,
                rewards=agent_rewards,
                infos=agent_infos,
                is_terminated=agent_is_terminated,
                is_truncated=agent_is_truncated,
                states=agent_states,
                extra_model_outputs=agent_extra_model_outputs,
            )
        # Otherwise return empty `SingleAgentEpisosde`.
        else:
            return SingleAgentEpisode(id_=episode_id)

    def _getattr_by_index(
        self,
        attr: str = "observations",
        indices: Union[int, List[int]] = -1,
        has_initial_value=False,
        global_ts: bool = True,
        as_list: bool = False,
    ) -> MultiAgentDict:
        # First for global_ts = True:
        if global_ts:
            # Check, if the indices are iterable.
            if isinstance(indices, list):
                indices = [
                    self.t + (idx + int(has_initial_value) if idx < 0 else idx)
                    for idx in indices
                ]
            else:
                indices = (
                    [self.t + indices + int(has_initial_value)]
                    if indices < 0
                    else [indices]
                )

            # If a list should be returned.
            if as_list:
                return [
                    {
                        agent_id: getattr(agent_eps, attr)[
                            self.global_t_to_local_t[agent_id].find_indices([idx])[0]
                        ]
                        for agent_id, agent_eps in self.agent_episodes.items()
                        if self.global_t_to_local_t[agent_id].find_indices([idx])
                    }
                    for idx in indices
                ]
            # Otherwise we return a dictionary.
            else:
                {
                    agent_id: list(
                        map(
                            getattr(agent_eps, attr).__getitem__,
                            self.global_t_to_local_t[agent_id].find_indices(indices),
                        )
                    )
                    for agent_id, agent_eps in self.agent_episodes.items()
                    # Only include agent data for agents that stepped.
                    if len(self.global_t_to_local_t[agent_id].find_indices(indices)) > 0
                }
        # Otherwise just look for the timesteps in the `SingleAgentEpisode`s
        # directly.
        else:
            # Check, if the indices are iterable.
            if not isinstance(indices, list):
                indices = [indices]

            return {
                agent_id: list(map(getattr(agent_eps, attr).__getitem__, indices))
                for agent_id, agent_eps in self.agent_episodes.items()
                # Only include agent data for agents that stepped so far at least once.
                # TODO (sven, simon): This will not include initial observations. Should
                # we?
                if self.agent_episodes[agent_id].t > 0
            }

    def _get_single_agent_data(
        self,
        agent_id: str,
        ma_data: List[MultiAgentDict],
        use_global_to_to_local_t: bool = True,
        start_index: int = 0,
        end_index: Optional[int] = None,
        shift: int = 0,
    ) -> List[Any]:
        """Returns single agent data from multi-agent data.

        Args:
            agent_id: A string identifying the agent for which the
                data should be extracted.
            ma_data: A List of dictionaries, each containing multi-agent
                data, i.e. mapping from agent ids to timestep data.
            start_index: An integer defining the start point of the
                extration window. The default starts at the beginning of the
                the `ma_data` list.
            end_index: Optional. An integer defining the end point of the
                extraction window. If `None`, the extraction window will be
                until the end of the `ma_data` list.g
            shift: An integer that defines by which amount to shift the
                running index for extraction. This is for example needed
                when we extract data that started at index 1.

        Returns: A list containing single-agent data from the multi-agent
            data provided.
        """
        # Should we search along the global timestep mapping, e.g. for observations,
        # or infos.
        if use_global_to_to_local_t:
            # Return all single agent data along the global timestep.
            return [
                singleton[agent_id]
                for singleton in list(
                    map(
                        ma_data.__getitem__,
                        [
                            i + shift
                            for i in self.global_t_to_local_t[agent_id][
                                start_index:end_index
                            ]
                        ],
                    )
                )
                if agent_id in singleton.keys()
            ]
        # Use all. This makes sense in multi-agent games where rewards could be given,
        # even to agents that receive no observation in a timestep.
        else:
            return [
                singleton[agent_id] for singleton in ma_data if agent_id in singleton
            ]

    def __len__(self):
        """Returns the length of an `MultiAgentEpisode`.

        Note that the length of an episode is defined by the difference
        between its actual timestep and the starting point.

        Returns: An integer defining the length of the episode or an
            error if the episode has not yet started.
        """
        assert self.t_started < self.t, (
            "ERROR: Cannot determine length of episode that hasn't started, yet!"
            "Call `MultiAgentEpisode.add_initial_observation(initial_observation=)` "
            "first (after which `len(MultiAgentEpisode)` will be 0)."
        )
        return self.t - self.t_started


class _IndexMapping(list):
    """Provides lists with a method to find multiple elements.

    This class is used for the timestep mapping which is central to
    the multi-agent episode. For each agent the timestep mapping is
    implemented with an `IndexMapping`.

    The `IndexMapping.find_indices` method simplifies the search for
    multiple environment timesteps at which some agents have stepped.
    See for example `MultiAgentEpisode.get_observations()`.
    """

    def find_indices(self, indices_to_find: List[int]):
        """Returns global timesteps at which an agent stepped.

        The function returns for a given list of indices the ones
        that are stored in the `IndexMapping`.

        Args:
            indices_to_find: A list of indices that should be
                found in the `IndexMapping`.

        Returns:
            A list of indices at which to find the `indices_to_find`
            in `self`. This could be empty if none of the given
            indices are in `IndexMapping`.
        """
        indices = []
        for num in indices_to_find:
            if num in self:
                indices.append(self.index(num))
        return indices

    def find_indices_with_greater_values(self, threshold):
        indices = []
        for num in reversed(self):
            if num <= threshold:
                break
            indices.append(self.index(num))
        return reversed(indices)
