from gymnasium.spaces import Space
import copy
from typing import Any, Dict, List, Optional, Type, Tuple, Union

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import (
    AlgorithmConfig,
    _check_rl_module_spec,
    NotProvided,
)
from ray.rllib.algorithms.marwil import MARWIL, MARWILConfig
from ray.rllib.connectors.learner import AddEpisodeLengthsToTrainBatch
from ray.rllib.core import ALL_MODULES, DEFAULT_MODULE_ID, DEFAULT_POLICY_ID
from ray.rllib.core.learner import Learner
from ray.rllib.core.learner.training_data import TrainingData
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    OFFLINE_SAMPLING_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EnvType, PolicyID


class BCIRLPPOConfig(MARWILConfig):
    def __init__(self, algo_class=None):
        """Initializes a PPOConfig instance."""
        self.exploration_config = {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": "StochasticSampling",
            # Add constructor kwargs here (if any).
        }

        super().__init__(algo_class=algo_class or BCIRLPPO)

        # fmt: off
        # __sphinx_doc_begin__

        self.policies ={DEFAULT_POLICY_ID: PolicySpec()}
        self.lr = 5e-5
        self.rollout_fragment_length = "auto"
        self.train_batch_size = 4000

        # PPO specific settings:
        self.use_critic = True
        self.use_gae = True
        self.num_epochs = 30
        self.minibatch_size = 128
        self.shuffle_batch_per_epoch = True
        self.lambda_ = 1.0
        self.use_kl_loss = True
        self.kl_coeff = 0.2
        self.kl_target = 0.01
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.0
        self.clip_param = 0.3
        self.vf_clip_param = 10.0
        self.grad_clip = None

        # Override some of AlgorithmConfig's default values with PPO-specific values.
        self.num_env_runners = 2

        self.reward_update_freq = 1

        # __sphinx_doc_end__
        # fmt: on

    @override(MARWILConfig)
    def get_default_rl_module_spec(self) -> RLModuleSpec:
        # return super().get_default_rl_module_spec()
        default_ppo_rl_module_spec = super().get_default_rl_module_spec()

        if self.framework_str == "torch":
            from ray.rllib.algorithms.bc_irl_ppo.torch.default_bc_irl_ppo_torch_rl_module import (
                DefaultBCIRLRewardTorchRLModule,
            )

            return MultiRLModuleSpec(
                rl_module_specs={
                    DEFAULT_MODULE_ID: default_ppo_rl_module_spec,
                    "reward_model": RLModuleSpec(
                        module_class=DefaultBCIRLRewardTorchRLModule,
                        learner_only=True,
                    ),
                }
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use either 'torch' or 'tf2'."
            )

    @override(MARWILConfig)
    def get_default_learner_class(self) -> Union[Type["Learner"], str]:
        if self.framework_str == "torch":
            from ray.rllib.algorithms.bc_irl_ppo.torch.bc_irl_ppo_torch_learner import (
                BCIRLPPOTorchLearner,
            )

            return BCIRLPPOTorchLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use `framework='torch'`."
            )

    @override(MARWILConfig)
    def build_learner_connector(
        self, input_observation_space, input_action_space, device=None
    ):
        pipeline = super().build_learner_connector(
            input_observation_space, input_action_space, device
        )

        pipeline.remove("TensorToNumpy")
        pipeline.remove("GeneralAdvantageEstimation")
        pipeline.insert_after(
            "AddNextObservationsFromEpisodesToTrainBatch",
            AddEpisodeLengthsToTrainBatch(),
        )
        return pipeline

    @override(MARWILConfig)
    def training(
        self,
        *,
        use_critic: Optional[bool] = NotProvided,
        use_gae: Optional[bool] = NotProvided,
        lambda_: Optional[float] = NotProvided,
        use_kl_loss: Optional[bool] = NotProvided,
        kl_coeff: Optional[float] = NotProvided,
        kl_target: Optional[float] = NotProvided,
        vf_loss_coeff: Optional[float] = NotProvided,
        entropy_coeff: Optional[float] = NotProvided,
        entropy_coeff_schedule: Optional[List[List[Union[int, float]]]] = NotProvided,
        clip_param: Optional[float] = NotProvided,
        vf_clip_param: Optional[float] = NotProvided,
        grad_clip: Optional[float] = NotProvided,
        **kwargs,
    ) -> "BCIRLPPOConfig":
        """Sets the training related configuration.

        Args:
            use_critic: Should use a critic as a baseline (otherwise don't use value
                baseline; required for using GAE).
            use_gae: If true, use the Generalized Advantage Estimator (GAE)
                with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            lambda_: The lambda parameter for General Advantage Estimation (GAE).
                Defines the exponential weight used between actually measured rewards
                vs value function estimates over multiple time steps. Specifically,
                `lambda_` balances short-term, low-variance estimates against long-term,
                high-variance returns. A `lambda_` of 0.0 makes the GAE rely only on
                immediate rewards (and vf predictions from there on, reducing variance,
                but increasing bias), while a `lambda_` of 1.0 only incorporates vf
                predictions at the truncation points of the given episodes or episode
                chunks (reducing bias but increasing variance).
            use_kl_loss: Whether to use the KL-term in the loss function.
            kl_coeff: Initial coefficient for KL divergence.
            kl_target: Target value for KL divergence.
            vf_loss_coeff: Coefficient of the value function loss. IMPORTANT: you must
                tune this if you set vf_share_layers=True inside your model's config.
            entropy_coeff: The entropy coefficient (float) or entropy coefficient
                schedule in the format of
                [[timestep, coeff-value], [timestep, coeff-value], ...]
                In case of a schedule, intermediary timesteps will be assigned to
                linearly interpolated coefficient values. A schedule config's first
                entry must start with timestep 0, i.e.: [[0, initial_value], [...]].
            clip_param: The PPO clip parameter.
            vf_clip_param: Clip param for the value function. Note that this is
                sensitive to the scale of the rewards. If your expected V is large,
                increase this.
            grad_clip: If specified, clip the global norm of gradients by this amount.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if use_critic is not NotProvided:
            self.use_critic = use_critic
            # TODO (Kourosh) This is experimental.
            #  Don't forget to remove .use_critic from algorithm config.
        if use_gae is not NotProvided:
            self.use_gae = use_gae
        if lambda_ is not NotProvided:
            self.lambda_ = lambda_
        if use_kl_loss is not NotProvided:
            self.use_kl_loss = use_kl_loss
        if kl_coeff is not NotProvided:
            self.kl_coeff = kl_coeff
        if kl_target is not NotProvided:
            self.kl_target = kl_target
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if clip_param is not NotProvided:
            self.clip_param = clip_param
        if vf_clip_param is not NotProvided:
            self.vf_clip_param = vf_clip_param
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip

        return self

    @property
    @override(MARWILConfig)
    def _model_config_auto_includes(self) -> Dict[str, Any]:
        return super()._model_config_auto_includes | {"vf_share_layers": False}

    @property
    @override(AlgorithmConfig)
    def rl_module_spec(self):
        default_rl_module_spec: MultiRLModuleSpec = self.get_default_rl_module_spec()
        _check_rl_module_spec(default_rl_module_spec)

        if self._rl_module_spec:
            _check_rl_module_spec(self._rl_module_spec)

            if isinstance(self._rl_module_spec, RLModuleSpec):
                # Merge with the default RLModuleSpec for the policy.
                default_rl_module_spec.remove_modules(DEFAULT_MODULE_ID)
                default_rl_module_spec.add_modules(
                    {DEFAULT_MODULE_ID: self._rl_module_spec}
                )
            if isinstance(self._rl_module_spec, MultiRLModuleSpec):
                multi_rl_module_spec = copy.deepcopy(self._rl)
                default_rl_module_spec.update(multi_rl_module_spec)
                return multi_rl_module_spec
        else:
            return default_rl_module_spec

    @override(AlgorithmConfig)
    def get_multi_rl_module_spec(
        self,
        *,
        env: Optional[EnvType] = None,
        spaces: Optional[Dict[PolicyID, Tuple[Space, Space]]] = None,
        inference_only: bool = False,
        # @HybridAPIStack
        policy_dict: Optional[Dict[str, PolicySpec]] = None,
        single_agent_rl_module_spec: Optional[RLModuleSpec] = None,
    ) -> MultiRLModuleSpec:
        # if single_agent_rl_module_spec is None:
        #     single_agent_rl_module_spec = super().get_default_rl_module_spec()
        # return super().get_multi_rl_module_spec(
        #     env=env,
        #     spaces=spaces,
        #     inference_only=inference_only,
        #     policy_dict=policy_dict,
        #     single_agent_rl_module_spec=single_agent_rl_module_spec
        # )
        # TODO (simon): `get_multi_rl_module_spec` is overriding an already
        # defined `MultiRLModule`. This should be avoided or the `policy_dict`
        # has to be enriched with the reward module.
        multi_rl_module_spec = self.rl_module_spec

        if policy_dict is None:
            policy_dict, _ = self.get_multi_agent_setup(env=env, spaces=spaces)
        # Fill in the missing values from the specs that we already have. By combining
        # PolicySpecs and the default RLModuleSpec.
        for module_id in multi_rl_module_spec.rl_module_specs:

            # Remove/skip `learner_only=True` RLModules if `inference_only` is True.
            module_spec = multi_rl_module_spec.rl_module_specs[module_id]
            if inference_only and module_spec.learner_only:
                multi_rl_module_spec.remove_modules(module_id)
                continue

            policy_spec = policy_dict.get(module_id)
            if policy_spec is None:
                policy_spec = policy_dict[DEFAULT_MODULE_ID]

            # if module_spec.catalog_class is None:
            #     if isinstance(default_rl_module_spec, RLModuleSpec):
            #         module_spec.catalog_class = default_rl_module_spec.catalog_class
            #     elif isinstance(default_rl_module_spec.rl_module_specs, RLModuleSpec):
            #         catalog_class = default_rl_module_spec.rl_module_specs.catalog_class
            #         module_spec.catalog_class = catalog_class
            #     elif module_id in default_rl_module_spec.rl_module_specs:
            #         module_spec.catalog_class = default_rl_module_spec.rl_module_specs[
            #             module_id
            #         ].catalog_class
            #     else:
            #         raise ValueError(
            #             f"Catalog class for module {module_id} cannot be inferred. "
            #             f"It is neither provided in the rl_module_spec that "
            #             "is passed in nor in the default module spec used in "
            #             "the algorithm."
            #         )
            # TODO (sven): Find a good way to pack module specific parameters from
            # the algorithms into the `model_config_dict`.
            if module_spec.observation_space is None:
                module_spec.observation_space = policy_spec.observation_space
            if module_spec.action_space is None:
                module_spec.action_space = policy_spec.action_space
            # In case the `RLModuleSpec` does not have a model config dict, we use the
            # the one defined by the auto keys and the `model_config_dict` arguments in
            # `self.rl_module()`.
            if module_spec.model_config is None:
                module_spec.model_config = self.model_config
            # Otherwise we combine the two dictionaries where settings from the
            # `RLModuleSpec` have higher priority.
            else:
                module_spec.model_config = (
                    self.model_config | module_spec._get_model_config()
                )

        return multi_rl_module_spec

    @property
    def is_online(self) -> bool:
        """Defines, if this config is for online RL."""
        return True


class BCIRLPPO(MARWIL):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return BCIRLPPOConfig()

    @override(MARWIL)
    def training_step(self) -> None:

        # Collect batches from sample workers until we have a full batch.
        with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
            # Sample in parallel from the workers.
            if self.config.count_steps_by == "agent_steps":
                episodes, env_runner_results = synchronous_parallel_sample(
                    worker_set=self.env_runner_group,
                    max_agent_steps=self.config.total_train_batch_size,
                    sample_timeout_s=self.config.sample_timeout_s,
                    _uses_new_env_runners=(
                        self.config.enable_env_runner_and_connector_v2
                    ),
                    _return_metrics=True,
                )
            else:
                episodes, env_runner_results = synchronous_parallel_sample(
                    worker_set=self.env_runner_group,
                    max_env_steps=self.config.total_train_batch_size,
                    sample_timeout_s=self.config.sample_timeout_s,
                    _uses_new_env_runners=(
                        self.config.enable_env_runner_and_connector_v2
                    ),
                    _return_metrics=True,
                )

            # Return early if all our workers failed.
            if not episodes:
                return

            # Reduce EnvRunner metrics over the n EnvRunners.
            self.metrics.merge_and_log_n_dicts(
                env_runner_results, key=ENV_RUNNER_RESULTS
            )

        # TODO (simon): Take care of sampler metrics: right
        #  now all rewards are `nan`, which possibly confuses
        #  the user that sth. is not right, although it is as
        #  we do not step the env.
        with self.metrics.log_time((TIMERS, OFFLINE_SAMPLING_TIMER)):
            # Sampling from offline data.
            iterator = self.offline_data.sample(
                num_samples=self.config.train_batch_size_per_learner,
                num_shards=self.config.num_learners,
                # Return an iterator, if a `Learner` should update
                # multiple times per RLlib iteration.
                return_iterator=True,
            )
            training_data = TrainingData(data_iterators=iterator, episodes=episodes)

        # Perform a learner update step on the collected episodes.
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update(
                # episodes=episodes,
                training_data=training_data,
                timesteps={
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: (
                        self.metrics.peek(
                            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME)
                        )
                    ),
                },
                num_epochs=self.config.num_epochs,
                minibatch_size=self.config.minibatch_size,
                shuffle_batch_per_epoch=self.config.shuffle_batch_per_epoch,
                num_iters=1,
            )
            self.metrics.merge_and_log_n_dicts(learner_results, key=LEARNER_RESULTS)

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            # The train results's loss keys are ModuleIDs to their loss values.
            # But we also return a total_loss key at the same level as the ModuleID
            # keys. So we need to subtract that to get the correct set of ModuleIDs to
            # update.
            # TODO (sven): We should also not be using `learner_results` as a messenger
            #  to infer which modules to update. `policies_to_train` might also NOT work
            #  as it might be a very large set (100s of Modules) vs a smaller Modules
            #  set that's present in the current train batch.
            modules_to_update = set(learner_results[0].keys()) - {ALL_MODULES}
            self.env_runner_group.sync_weights(
                # Sync weights from learner_group to all EnvRunners.
                from_worker_or_learner_group=self.learner_group,
                policies=modules_to_update,
                inference_only=True,
            )
