import logging
from datasets import load_dataset

import ray
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.offline.offline_data import OfflineData
from ray.rllib.tuned_examples.rlhf.tldr_rlhf_offline_prelearner import (
    TLDRRLHFOfflinePreLearner,
)

logger = logging.getLogger(__name__)


class TLDRRLHFOfflineData(OfflineData):
    def __init__(self, config: AlgorithmConfig):

        self.config: AlgorithmConfig = config
        self.is_multi_agent: bool = False
        self.path: str = self.config.input_

        # SFT model ID to load a corresponding tokenizer.
        self.sft_model_id = self.config.sft_model_id

        # Load the dataset.
        try:
            # Enable streaming for the huggingface data.
            hf_dataset_stream = load_dataset(self.path, streaming=True)
            # Wrap the huggingface data stream by a Ray Data datasource.
            # TODO (simon): Check, later, how to stream the validation data.
            self.data = ray.data.from_huggingface(hf_dataset_stream["train"])
        except Exception as e:
            logger.error(e)

        # Avoids reinstantiating the batch iterator each time we sample.
        self.batch_iterator = None

        self.data_mapped = False
        self.map_batches_kwargs = (
            self.default_map_batches_kwargs | self.config.map_batches_kwargs
        )
        self.iter_batches_kwargs = (
            self.default_iter_batches_kwargs | self.config.iter_batches_kwargs
        )
        # Defines the prelearner class. Note, this could be user-defined.
        # TODO (simon): Add the TLDR prelearner here.
        self.prelearner_class = (
            self.config.prelearner_class or TLDRRLHFOfflinePreLearner
        )

    def sample(
        self,
        num_samples: int,
        return_iterator: bool = False,
        num_shards: int = 1,
    ):
        if num_shards > 1:
            raise ValueError("`RLHFDataOffline` class allows no multi-learner setups.")

        # If the data is not mapped, yet, map it here and apply the filter.
        if not self.data_mapped:
            self.data = self.data.map_batches(
                self.prelearner_class,
                fn_constructor_kwargs={
                    "config": self.config,
                    "sft_model_id": self.sft_model_id,
                },
                batch_size=num_samples,
                **self.map_batches_kwargs,
            )

        # Build an data iterator, if necessary.
        if not self.batch_iterator:
            self.batch_iterator = iter(
                self.data.iter_batches(
                    batch_size=1,
                    **self.iter_batches_kwargs,
                )
            )

        # Return a single batch from the iterator.
        try:
            return next(self.batch_iterator)
        except StopIteration:
            # If the batch iterator is exhausted, reinstantiate a new one.
            logger.debug(
                "===> [OfflineData]: Batch iterator exhausted. Reinstantiating ..."
            )
            self.batch_iterator = None
            return self.sample(
                num_samples=num_samples,
                return_iterator=False,
                num_shards=1,
            )

    @property
    def default_map_batches_kwargs(self):
        kwargs = super().default_map_batches_kwargs
        del kwargs["zero_copy_batch"]
        return kwargs
