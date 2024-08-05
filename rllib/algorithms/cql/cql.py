import logging
from typing import Optional, Type, Union

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.core.learner.learner import Learner
from ray.rllib.algorithms.cql.cql_tf_policy import CQLTFPolicy
from ray.rllib.algorithms.cql.cql_torch_policy import CQLTorchPolicy
from ray.rllib.algorithms.sac.sac import (
    SAC,
    SACConfig,
)
from ray.rllib.execution.rollout_ops import (
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    multi_gpu_train_one_step,
    train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.metrics import (
    LAST_TARGET_UPDATE_TS,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_TRAINED,
    NUM_TARGET_UPDATES,
    TARGET_NET_UPDATE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
)
from ray.rllib.utils.typing import ResultDict

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()
logger = logging.getLogger(__name__)


class CQLConfig(SACConfig):
    """Defines a configuration class from which a CQL can be built.

    .. testcode::
        :skipif: True

        from ray.rllib.algorithms.cql import CQLConfig
        config = CQLConfig().training(gamma=0.9, lr=0.01)
        config = config.resources(num_gpus=0)
        config = config.env_runners(num_env_runners=4)
        print(config.to_dict())
        # Build a Algorithm object from the config and run 1 training iteration.
        algo = config.build(env="CartPole-v1")
        algo.train()
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CQL)

        # fmt: off
        # __sphinx_doc_begin__
        # CQL-specific config settings:
        self.bc_iters = 20000
        self.temperature = 1.0
        self.num_actions = 10
        self.lagrangian = False
        self.lagrangian_thresh = 5.0
        self.min_q_weight = 5.0

        # Changes to Algorithm's/SACConfig's default:

        # .reporting()
        self.min_sample_timesteps_per_iteration = 0
        self.min_train_timesteps_per_iteration = 100
        # fmt: on
        # __sphinx_doc_end__

        self.timesteps_per_iteration = DEPRECATED_VALUE

    @override(SACConfig)
    def training(
        self,
        *,
        bc_iters: Optional[int] = NotProvided,
        temperature: Optional[float] = NotProvided,
        num_actions: Optional[int] = NotProvided,
        lagrangian: Optional[bool] = NotProvided,
        lagrangian_thresh: Optional[float] = NotProvided,
        min_q_weight: Optional[float] = NotProvided,
        **kwargs,
    ) -> "CQLConfig":
        """Sets the training-related configuration.

        Args:
            bc_iters: Number of iterations with Behavior Cloning pretraining.
            temperature: CQL loss temperature.
            num_actions: Number of actions to sample for CQL loss
            lagrangian: Whether to use the Lagrangian for Alpha Prime (in CQL loss).
            lagrangian_thresh: Lagrangian threshold.
            min_q_weight: in Q weight multiplier.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if bc_iters is not NotProvided:
            self.bc_iters = bc_iters
        if temperature is not NotProvided:
            self.temperature = temperature
        if num_actions is not NotProvided:
            self.num_actions = num_actions
        if lagrangian is not NotProvided:
            self.lagrangian = lagrangian
        if lagrangian_thresh is not NotProvided:
            self.lagrangian_thresh = lagrangian_thresh
        if min_q_weight is not NotProvided:
            self.min_q_weight = min_q_weight

        return self

    @override(SACConfig)
    def validate(self) -> None:
        # First check, whether old `timesteps_per_iteration` is used.
        if self.timesteps_per_iteration != DEPRECATED_VALUE:
            deprecation_warning(
                old="timesteps_per_iteration",
                new="min_train_timesteps_per_iteration",
                error=True,
            )

        # Call super's validation method.
        super().validate()

        # CQL-torch performs the optimizer steps inside the loss function.
        # Using the multi-GPU optimizer will therefore not work (see multi-GPU
        # check above) and we must use the simple optimizer for now.
        if self.simple_optimizer is not True and self.framework_str == "torch":
            self.simple_optimizer = True

        if self.framework_str in ["tf", "tf2"] and tfp is None:
            logger.warning(
                "You need `tensorflow_probability` in order to run CQL! "
                "Install it via `pip install tensorflow_probability`. Your "
                f"tf.__version__={tf.__version__ if tf else None}."
                "Trying to import tfp results in the following error:"
            )
            try_import_tfp(error=True)


class CQL(SAC):
    """CQL (derived from SAC)."""

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfig:
        return CQLConfig()

    @classmethod
    @override(SAC)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return CQLTorchPolicy
        else:
            return CQLTFPolicy


    def get_default_learner_class(self) -> Union[Type["Learner"], str]:
        if self.framework_str == "torch":
            from ray.rllib.algorithms.cql.torch.cql_torch_learner import CQLTorchLearner

            return CQLTorchLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use `'torch'` instead."
            )

    @override(SAC)
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers.
        with self._timers[SAMPLE_TIMER]:
            train_batch = synchronous_parallel_sample(worker_set=self.env_runner_group)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Postprocess batch before we learn on it.
        post_fn = self.config.get("before_learn_on_batch") or (lambda b, *a: b)
        train_batch = post_fn(train_batch, self.env_runner_group, self.config)

        # Learn on training batch.
        # Use simple optimizer (only for multi-agent or tf-eager; all other
        # cases should use the multi-GPU optimizer, even if only using 1 GPU)
        if self.config.get("simple_optimizer") is True:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        # Update target network every `target_network_update_freq` training steps.
        cur_ts = self._counters[
            NUM_AGENT_STEPS_TRAINED
            if self.config.count_steps_by == "agent_steps"
            else NUM_ENV_STEPS_TRAINED
        ]
        last_update = self._counters[LAST_TARGET_UPDATE_TS]
        if cur_ts - last_update >= self.config.target_network_update_freq:
            with self._timers[TARGET_NET_UPDATE_TIMER]:
                to_update = self.env_runner.get_policies_to_train()
                self.env_runner.foreach_policy_to_train(
                    lambda p, pid: pid in to_update and p.update_target()
                )
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = cur_ts

        # Update remote workers's weights after learning on local worker
        # (only those policies that were actually trained).
        if self.env_runner_group.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.env_runner_group.sync_weights(policies=list(train_results.keys()))

        # Return all collected metrics for the iteration.
        return train_results


    def _training_step_new_api_stack(self) -> ResultDict:

        with self.metrics.log_time((TIMERS, OFFLINE_SAMPLING_TIMER)):
            # Sampling from offline data.
            batch = self.offline_data.sample(
                num_samples=self.config.train_batch_size_per_learner,
                num_shards=self.config.num_learners,
                return_iterator=True if self.config.num_learners > 1 else False,
            )
        
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            # Updating the policy.
            # TODO (simon, sven): Check, if we should execute directly s.th. like
            # update_from_iterator.
            learner_results = self.learner_group.update_from_batch(
                batch,
                minibatch_size=self.config.train_batch_size_per_learner,
                num_iters=self.config.dataset_num_iters_per_learner,
            )

            # Log training results.
            self.metrics.merge_and_log_n_dicts(learner_results, key=LEARNER_RESULTS)
            self.metrics.log_value(
                NUM_ENV_STEPS_TRAINED_LIFETIME,
                self.metrics.peek(
                    (LEARNER_RESULTS, ALL_MODULES, NUM_ENV_STEPS_TRAINED)
                ),
                reduce="sum",
            )
            self.metrics.log_dict(
                {
                    (LEARNER_RESULTS, mid, NUM_MODULE_STEPS_TRAINED_LIFETIME): (
                        stats[NUM_MODULE_STEPS_TRAINED]
                    )
                    for mid, stats in self.metrics.peek(LEARNER_RESULTS).items()
                },
                reduce="sum",
            )

        # Synchronize weights.
        # As the results contain for each policy the loss and in addition the
        # total loss over all policies is returned, this total loss has to be
        # removed.
        modules_to_update = set(learner_results[0].keys()) - {ALL_MODULES}

        # Update weights - after learning on the local worker -
        # on all remote workers. Note, we only have the local `EnvRunner`,
        # but from this `EnvRunner` the evaulation `EnvRunner`s get updated.
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            self.env_runner_group.sync_weights(
                # Sync weights from learner_group to all EnvRunners.
                from_worker_or_learner_group=self.learner_group,
                policies=modules_to_update,
                inference_only=True,
            )

        return self.metrics.reduce()