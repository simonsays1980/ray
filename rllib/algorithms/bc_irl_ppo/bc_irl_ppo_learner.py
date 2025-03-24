import higher
import numpy

from typing import Dict

from ray.rllib.algorithms.marwil.marwil_learner import MARWILLearner
from ray.rllib.core.learner.learner import Learner
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils import unflatten_dict
from ray.rllib.utils.annotations import override
from ray.rllib.utils.minibatch_utils import (
    MiniBatchCyclicIterator,
    MiniBatchRayDataIterator,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.schedules.scheduler import Scheduler


class BCIRLPPOLearner(MARWILLearner):
    @override(MARWILLearner)
    def build(self) -> None:
        super().build()

        self._n_updates = 0

        # Dict mapping module IDs to the respective entropy Scheduler instance.
        self.entropy_coeff_schedulers_per_module: Dict[
            ModuleID, Scheduler
        ] = LambdaDefaultDict(
            lambda module_id: Scheduler(
                fixed_value_or_schedule=(
                    self.config.get_config_for_module(module_id).entropy_coeff
                ),
                framework=self.framework,
                device=self._device,
            )
        )

        # Set up KL coefficient variables (per module).
        # Note that the KL coeff is not controlled by a Scheduler, but seeks
        # to stay close to a given kl_target value.
        self.curr_kl_coeffs_per_module: Dict[ModuleID, TensorType] = LambdaDefaultDict(
            lambda module_id: self._get_tensor_variable(
                self.config.get_config_for_module(module_id).kl_coeff
            )
        )

    @override(Learner)
    def update(
        self,
        batch=None,
        batches=None,
        batch_refs=None,
        episodes=None,
        episodes_refs=None,
        data_iterators=None,
        training_data=None,
        *,
        timesteps=None,
        num_total_minibatches=0,
        num_epochs=1,
        minibatch_size=None,
        shuffle_batch_per_epoch=False,
        _no_metrics_reduce=False,
        **kwargs,
    ):

        self._check_is_built()

        # Call `before_gradient_based_update` to allow for non-gradient based
        # preparations-, logging-, and update logic to happen.
        self.before_gradient_based_update(timesteps=timesteps or {})

        if not self.iterator:
            num_iters = kwargs.pop("num_iters", None)
            if num_iters is None:
                raise ValueError(
                    "Learner.update(data_iterators=..) requires `num_iters` kwarg!"
                )

            def _collate_fn(_batch: Dict[str, numpy.ndarray]) -> MultiAgentBatch:
                _batch = unflatten_dict(_batch)
                _batch = MultiAgentBatch(
                    {
                        module_id: SampleBatch(module_data)
                        for module_id, module_data in _batch.items()
                    },
                    env_steps=sum(
                        len(next(iter(module_data.values())))
                        for module_data in _batch.values()
                    ),
                )
                _batch = self._convert_batch_type(_batch, to_device=False)
                return self._set_slicing_by_batch_id(_batch, value=True)

            def _finalize_fn(batch: MultiAgentBatch) -> MultiAgentBatch:
                return self._convert_batch_type(batch, to_device=True, use_stream=True)

            # This iterator holds a `ray.data.DataIterator` and manages it state.
            self.iterator = MiniBatchRayDataIterator(
                iterator=training_data.data_iterators[0],
                collate_fn=_collate_fn,
                finalize_fn=_finalize_fn,
                minibatch_size=minibatch_size,
                num_iters=num_iters,
                **kwargs,
            )

        offline_batch_iter = self.iterator

        batch = self._make_batch_if_necessary(training_data=training_data)
        assert batch is not None
        # TODO: Move this into LearnerConnector pipeline?
        # Filter out those RLModules from the final train batch that should not be
        # updated.
        for module_id in list(batch.policy_batches.keys()):
            if not self.should_module_be_updated(module_id, batch):
                del batch.policy_batches[module_id]
        if not batch.policy_batches:
            return {}

        batch_iter = MiniBatchCyclicIterator(
            batch,
            num_epochs=num_epochs,
            minibatch_size=minibatch_size,
            shuffle_batch_per_epoch=shuffle_batch_per_epoch and (num_epochs > 1),
            num_total_minibatches=num_total_minibatches,
        )

        if (
            self.config.reward_update_freq != -1
            and self._n_updates % self.config.reward_update_freq == 0
        ):
            offline_batch = next(iter(offline_batch_iter))
        else:
            offline_batch = None

        # TODO: Clone parameters for inner loop.
        policy_params = {
            name: param.clone().requires_grad_(True)
            for name, param in self.module["default_policy"].named_parameters()
        }

        # Run PPO policy optimization, but do not apply gradient.
        for iteration, tensor_minibatch in enumerate(batch_iter):
            # Check the MultiAgentBatch, whether our RLModule contains all ModuleIDs
            # found in this batch. If not, throw an error.
            unknown_module_ids = set(tensor_minibatch.policy_batches.keys()) - set(
                self.module.keys()
            )
            if unknown_module_ids:
                raise ValueError(
                    f"Batch contains one or more ModuleIDs ({unknown_module_ids}) that "
                    f"are not in this Learner!"
                )

            # Make the actual in-graph/traced `_update` call. This should return
            # all tensor values (no numpy).
            # TODO: Use `torch.func` and pass in cloned parameters.
            fwd_out, loss_per_module, tensor_metrics = self._update(
                tensor_minibatch.policy_batches,
                params=policy_params,
                module_id="default_policy",
            )

            # Convert logged tensor metrics (logged during tensor-mode of MetricsLogger)
            # to actual (numpy) values.
            self.metrics.tensors_to_numpy(tensor_metrics)

            # TODO (sven): Maybe move this into loop above to get metrics more accuratcely
            #  cover the minibatch/epoch logic.
            # Log all timesteps (env, agent, modules) based on given episodes/batch.
            self._log_steps_trained_metrics(tensor_minibatch)

            self._set_slicing_by_batch_id(tensor_minibatch, value=False)

        # Log all individual RLModules' loss terms and its registered optimizers'
        # current learning rates.
        # Note: We do this only once for the last of the minibatch updates, b/c the
        # window is only 1 anyways.
        for mid, loss in convert_to_numpy(loss_per_module).items():
            self.metrics.log_value(
                key=(mid, self.TOTAL_LOSS_KEY),
                value=loss,
                window=1,
            )

        # Call `after_gradient_based_update` to allow for non-gradient based
        # cleanups-, logging-, and update logic to happen.
        # TODO (simon): Check, if this should stay here, when running multiple
        # gradient steps inside the iterator loop above (could be a complete epoch)
        # the target networks might need to be updated earlier.
        self.after_gradient_based_update(timesteps=timesteps or {})

        # Reduce results across all minibatch update steps.
        if not _no_metrics_reduce:
            return self.metrics.reduce()
