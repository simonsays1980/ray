from gym.spaces import Space
from typing import Dict, List, Optional, Union, TYPE_CHECKING

from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_torch, TensorType
from ray.rllib.utils.typing import AlgorithmConfigDict, LocalOptimizer, ModelWeights

if TYPE_CHECKING:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
    from ray.rllib.policy.policy import Policy
    from ray.rllib.utils import try_import_tf

    _, tf, _ = try_import_tf()

_, nn = try_import_torch()


@DeveloperAPI
class Exploration:
    """Implements an exploration strategy for Policies.

    An Exploration takes model outputs, a distribution, and a timestep from
    the agent and computes an action to apply to the environment using an
    implemented exploration schema.
    """

    def __init__(
        self,
        action_space: Space,
        *,
        framework: str,
        policy_config: AlgorithmConfigDict,
        model: ModelV2,
        num_workers: int,
        worker_index: int,
        **kwargs,
    ):
        """
        Args:
            action_space: The action space in which to explore.
            framework: One of "tf" or "torch".
            policy_config: The Policy's config dict.
            model: The Policy's model.
            num_workers: The overall number of workers used.
            worker_index: The index of the worker using this class.
        """
        self.action_space = action_space
        self.policy_config = policy_config
        self.model = model
        self.num_workers = num_workers
        self.worker_index = worker_index
        self.framework = framework
        # If the exploration module needs to update its parameters globally
        # i.e. during the policy training iteration.
        self.global_update = False
        # The device on which the Model has been placed.
        # This Exploration will be on the same device.
        self.device = None
        if isinstance(self.model, nn.Module):
            params = list(self.model.parameters())
            if params:
                self.device = params[0].device

    @DeveloperAPI
    def before_compute_actions(
        self,
        *,
        timestep: Optional[Union[TensorType, int]] = None,
        explore: Optional[Union[TensorType, bool]] = None,
        tf_sess: Optional["tf.Session"] = None,
        **kwargs,
    ):
        """Hook for preparations before policy.compute_actions() is called.

        Args:
            timestep: An optional timestep tensor.
            explore: An optional explore boolean flag.
            tf_sess: The tf-session object to use.
            **kwargs: Forward compatibility kwargs.
        """
        pass

    # fmt: off
    # __sphinx_doc_begin_get_exploration_action__

    @DeveloperAPI
    def get_exploration_action(
        self,
        *,
        action_distribution: ActionDistribution,
        timestep: Union[TensorType, int],
        explore: bool = True
    ):
        """Returns a (possibly) exploratory action and its log-likelihood.

        Given the Model's logits outputs and action distribution, returns an
        exploratory action.

        Args:
            action_distribution: The instantiated
                ActionDistribution object to work with when creating
                exploration actions.
            timestep: The current sampling time step. It can be a tensor
                for TF graph mode, otherwise an integer.
            explore: True: "Normal" exploration behavior.
                False: Suppress all exploratory behavior and return
                a deterministic action.

        Returns:
            A tuple consisting of 1) the chosen exploration action or a
            tf-op to fetch the exploration action from the graph and
            2) the log-likelihood of the exploration action.
        """
        pass

    # __sphinx_doc_end_get_exploration_action__
    # fmt: on

    @DeveloperAPI
    def on_episode_start(
        self,
        policy: "Policy",
        *,
        environment: BaseEnv = None,
        episode: int = None,
        tf_sess: Optional["tf.Session"] = None,
    ):
        """Handles necessary exploration logic at the beginning of an episode.

        Args:
            policy: The Policy object that holds this Exploration.
            environment: The environment object we are acting in.
            episode: The number of the episode that is starting.
            tf_sess: In case of tf, the session object.
        """
        pass

    @DeveloperAPI
    def on_episode_end(
        self,
        policy: "Policy",
        *,
        environment: BaseEnv = None,
        episode: int = None,
        tf_sess: Optional["tf.Session"] = None,
    ):
        """Handles necessary exploration logic at the end of an episode.

        Args:
            policy: The Policy object that holds this Exploration.
            environment: The environment object we are acting in.
            episode: The number of the episode that is starting.
            tf_sess: In case of tf, the session object.
        """
        pass

    @DeveloperAPI
    def postprocess_trajectory(
        self,
        policy: "Policy",
        sample_batch: SampleBatch,
        tf_sess: Optional["tf.Session"] = None,
    ):
        """Handles post-processing of done episode trajectories.

        Changes the given batch in place. This callback is invoked by the
        sampler after policy.postprocess_trajectory() is called.

        Args:
            policy: The owning policy object.
            sample_batch: The SampleBatch object to post-process.
            tf_sess: An optional tf.Session object.
        """
        return sample_batch

    @DeveloperAPI
    def get_exploration_optimizer(
        self, optimizers: List[LocalOptimizer]
    ) -> List[LocalOptimizer]:
        """May add optimizer(s) to the Policy's own `optimizers`.

        The number of optimizers (Policy's plus Exploration's optimizers) must
        match the number of loss terms produced by the Policy's loss function
        and the Exploration component's loss terms.

        Args:
            optimizers: The list of the Policy's local optimizers.

        Returns:
            The updated list of local optimizers to use on the different
            loss terms.
        """
        return optimizers

    @DeveloperAPI
    def compute_loss_and_update(
        self,
        sample_batch: SampleBatch,
        policy: "Policy",
    ):
        """May compute a loss term and update parameters.

        Exploration modules might contain one or more models that need to be
        trained during each iteration. Note, this function is only activated,
        if `global_update` is set to `True`.

        Args:
            sample_batch: The sample batch with the examples to be trained on.
            policy: The `Policy` object that called this function.
        """
        pass

    @DeveloperAPI
    def extra_action_out_fn(
        self,
        policy: "Policy",
    ) -> Dict[str, TensorType]:
        """May compute and initialize some extra action output fetches.

        This function is explicitly for TensorFlow. See `extra_action_out()`
        above for the equivalent for PyTorch.
        Exploration modules might need to compute some extra output during
        action computation. This is especially important when using TensorFlow
        with a static graph as values can be initialized, placeholders created
        and values cached. Note, this function is only activated, if
        `global_update` is set to `True`.

        Args:
            policy: The `Policy` object that calls this function.

        Returns:
            Extra outputs to return in a `compute_actions_from_input_dict()`
            call (3rd return value).
        """
        return policy.policy_extra_action_out_fn()

    @DeveloperAPI
    def extra_action_out(
        self,
        input_dict: Dict[str, TensorType],
        state_batches: List[TensorType],
        model: "TorchModelV2",
        action_dist: "TorchDistributionWrapper",
        policy: "Policy",
    ) -> Dict[str, TensorType]:
        """May compute and initialize some extra action output fetches.

        This function is explicitly for PyTorch. See `extra_action_out_fn()`
        above for the equivalent for TensorFlow.
        Exploration modules might need to compute some extra output during
        action computation. Note, this function is only activated, if
        `global_update` is set to `True`. As Torch needs different parameters
        the function is overloaded.

        Args:
            Args:
            input_dict: Dict of model input tensors.
            state_batches: List of state tensors.
            model: Reference to the model object.
            action_dist: Torch action dist object
                to get log-probs (e.g. for already sampled actions).
            policy: The `Policy` object that calls this function.

        Returns:
            Extra outputs to return in a `compute_actions_from_input_dict()`
            call (3rd return value).
        """
        return policy.policy_action_out(
            input_dict,
            state_batches,
            model,
            action_dist,
        )

    @DeveloperAPI
    def get_weights(self) -> ModelWeights:
        """Returns the current exploration model's weights.

        Returns:
            The model weights if existent.
        """
        pass

    @DeveloperAPI
    def set_weights(self, weights: ModelWeights):
        """Sets the exploration model's weights.

        Args:
            weights: Model weights.
        """
        pass

    @DeveloperAPI
    def get_state(self, sess: Optional["tf.Session"] = None) -> Dict[str, TensorType]:
        """Returns the current exploration state.

        Args:
            sess: An optional tf Session object to use.

        Returns:
            The Exploration object's current state.
        """
        return {}

    @DeveloperAPI
    def set_state(self, state: object, sess: Optional["tf.Session"] = None) -> None:
        """Sets the Exploration object's state to the given values.

        Note that some exploration components are stateless, even though they
        decay some values over time (e.g. EpsilonGreedy). However the decay is
        only dependent on the current global timestep of the policy and we
        therefore don't need to keep track of it.

        Args:
            state: The state to set this Exploration to.
            sess: An optional tf Session object to use.
        """
        pass

    @Deprecated(new="get_state", error=False)
    def get_info(self, sess: Optional["tf.Session"] = None):
        return self.get_state(sess)
