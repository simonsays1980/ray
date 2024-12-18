import numpy as np
import uuid

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.offline.offline_prelearner import OfflinePreLearner
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import EpisodeType

from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Union

torch, nn = try_import_torch()

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

    def postprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        # TODO (simon): Put this probably into a connector.
        postprocessed_responses = batch[Columns.ACTIONS]

        if self.tokenizer.eos_token_id is not None:
            # Get the positions of the EOS token.
            stop_token_locations = postprocessed_responses == self.tokenizer.eos_token_id
            first_stop_tokens = self._match_first_token(stop_token_locations)
            # Mask now the EOS token IDs with the PAD token ID.
            view_shape = [1] * (len(postprocessed_responses.size()) - 1) + [postprocessed_responses.shape[1]]
            batch_indices = torch.arange(
                postprocessed_responses.shape[1], device=postprocessed_responses.device
            ).view(view_shape)
            # Store these for reward calculations later.
            batch["postprocessed_actions"] = torch.masked_fill(
                postprocessed_responses,
                batch_indices > first_stop_tokens,
                self.tokenizer.pad_token_id,
            )
        
        # Concatenate postprocessed responses with the queries.
        batch["postprocessed_query_responses"] = torch.cat([batch["input_ids"], batch["postprocessed_actions"]], dim=-1)
        # Receive the sequence lengths by finding the first right padding.
        pad_token_locations = postprocessed_responses == self.tokenizer.pad_token_id
        first_pad_tokens = self._match_first_token(pad_token_locations)
        # Note, the sequence ends before the first right PAD token.
        batch["sequence_lengths"] = first_pad_tokens - 1

        return batch

    def postprocess_rewards(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        # TODO (simon): Put this probably into a connector.

        # If we need to penalize missing EOS tokens in responses.
        if self.config.missing_eos_penalty:
            contain_eos_token = torch.any(batch["postprocessed_actions"] == self.tokenizer.eos_token_id)
            batch["scores"][~contain_eos_token] -= self.config.missing_eos_penalty
        
        # Mask the log-probabilities of responses with the `invalid_logprob_value`,
        # i.e. where responses are shorter than the sequence.
        response_idxs = torch.arange(batch[Columns.ACTIONS].shape[1], device=batch[Columns.ACTIONS].device).repeat(
            batch[Columns.ACTIONS].shape[0], 1,
        )
        # TODO (simon): Maybe rename to "response_lengths", b/c the sequence lengths are actually
        # all the same for the batch.
        padding_mask = response_idxs > batch["sequence_lengths"]
        batch[Columns.ACTION_LOGP] = torch.masked_fill(batch[Columns.ACTION_LOGP], padding_mask, self.config.invalid_logprob_value)
        batch["reference_action_logp"] = torch.masked_fill(batch["reference_action_logp"], padding_mask, self.config.invalid_logprob_value)
        # Mask the values with 0 on the right side, but include the EOS token ID.
        sequence_lengths_plus_eos = batch["sequence_lengths"] + 1
        # Note, the values are single-dimensioned.
        padding_mask_plus_eos = response_idxs > sequence_lengths_plus_eos.unsqueeze(-1)
        batch[Columns.VF_PREDS] = torch.masked_fill(batch[Columns.VF_PREDS], padding_mask_plus_eos, 0)

        # Compute now the rewards. Each token gets a KL reward for being near in probability to the
        # reference policy.
        kl_reward = batch[Columns.ACTION_LOGP] - batch["reference_action_logp"]
        kl_reward *= -self.config.kl_coeff
        # Then, add the reward for the full response (i.e. until the EOS token; if no EOS token the
        # full response is used and might have been penalized before).
        rewards = kl_reward.clone()
        actual_end_of_response = torch.where(
            sequence_lengths_plus_eos < rewards.size(1), 
            sequence_lengths_plus_eos, 
            batch["sequence_lengths"],
        )
        rewards[
            [
                torch.arange(rewards.size(0), device=rewards.device),
                actual_end_of_response,
            ]
        ] += batch[Columns.VF_PREDS]
        batch[Columns.REWARDS] = rewards

        lastgaelam = 0
        advantages_reversed = []
        gen_length = batch[Columns.ACTIONS].shape[1]
        for t in reversed(range(gen_length)):
            nextvalues = batch[Columns.VF_PREDS][:, t + 1] if t < gen_length else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - batch[Columns.VF_PREDS][:, t]
            lastgaelam = delta + self.config.gamma * self.config.lambda_ * lastgaelam
            advantages_reversed.append(lastgaelam)

        # Whiten advantages.
        

        batch = MultiAgentBatch(
            {
                "default_policy": SampleBatch(batch),
            },
            env_steps=batch["input_ids"].shape[0],
        )

        return batch


    def _match_first_token(self, locations):
        first_matches = torch.max(locations.type(torch.long), dim=-1).indices.unsqueeze(-1)
        # If the locations show no `True` `torch.max` assigns the index 0 to it.
        first_matches[first_matches == 0] = locations.shape[-1]

        return first_matches

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
