import numpy as np

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.offline.offline_prelearner import OfflinePreLearner
from ray.rllib.utils.typing import EpisodeType

from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Union

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}{{message['role'].capitalize() + ': ' + "
    "message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}"
    "{{ 'Assistant:' }}{% endif %}"
)


class TLDRRLHFOfflinePreLearner(OfflinePreLearner):
    def __init__(
        self,
        *,
        config: "AlgorithmConfig",
        sft_model_id: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ):

        self.config = config

        # Check, if an SFT Model ID had beend
        if not sft_model_id:
            raise ValueError(
                "===> [TLDRRLHFOfflinePreLearner] - no SFT model ID provided."
            )
        else:
            self.sft_model_id = sft_model_id

        # Set up the tokenizer for on-the-fly tokenizing and padding.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sft_model_id,
            # TODO (simon): Add tokenizer_args.
            padding_side="left",
            trust_remote_code=True,
        )

        # Add the chat template for the specific TL;DR data.
        # TODO (simon): For RLHFPreLearner bring this into Tokenizer kwargs.
        self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def __call__(
        self, batch: Union[Dict[str, np.ndarray], List[Dict[str, Any]]]
    ) -> Union[Dict[str, List[EpisodeType]], List[Dict[str, Any]]]:

        batch = [dict(zip(batch, values)) for values in zip(*batch.values())]
        tokenized_inputs = []
        for row in batch:
            tokenized_input = self._tokenize(row)
            if tokenized_input if tokenized_input["lengths"] <= 512 else []:
                tokenized_inputs.append(tokenized_input)

        return {"batch": [self._pad(tokenized_inputs)]}

    def _tokenize(self, element) -> Dict[str, Any]:
        input_ids = self.tokenizer.apply_chat_template(
            element["messages"][:1],
            padding=False,
            add_generation_prompt=True,
        )
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    def _pad(self, elements) -> Dict[str, Any]:
        return self.tokenizer.pad(
            elements,
            max_length=None,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
