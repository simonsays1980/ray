from typing import Any, Dict, Optional

from ray.rllib.algorithms.rlhf.rlhf_rl_module import RLHFRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from torch.nn import attention

torch, nn = try_import_torch()


class RLHFTorchRLModule(TorchRLModule, RLHFRLModule):
    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:

        output = self.pi(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            return_dict=True,
            output_hidden_states=True,
        )

        logits = output.sequences.logits[:, batch["context_length"] - 1 : -1]
        logits /= self.temperature + 1e-7
        return {Columns.ACTION_DIST_INPUTS: logits}

    def generate(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:

        generation_config = kwargs.get("generation_config") or self.generation_config
        output = self.pi.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Return a dictionary instead of the output object of `transformers`.
        return {
            # Sequences. Return only responses not queries.
            Columns.ACTIONS: output.sequences[:, batch["input_ids"].shape[1] :],
            # Logits.
            Columns.ACTION_DIST_INPUTS: torch.stack(output.scores, dim=-1),
        }

    def reference_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        attention_mask = batch["attention_mask"]
        # Generate position IDs.
        position_ids = attention_mask.cumsum(dim=1) - attention_mask.long()
        # Mask padding with zero.
        masked_input_ids = torch.masked_fill(
            batch["query_responses"], ~attention_mask, value=0
        )
        output = self.ref_pi.forward(
            masked_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )

        logits = output[:, batch["input_ids"].shape[1] - 1 : -1]
        # Scale the logits by the desired temperature.
        logits /= self.temperature + 1e-7

        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:

        # Get the value model LLM backbone.
        vf_llm_backbone = getattr(self.vf, self.vf.base_model_prefix)

        output = vf_llm_backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            return_dict=True,
            output_hidden_states=True,
            # We need output for new sequences.
            use_cache=False,
        )

        # Compute values by using the score function with the last hidden
        # state. Note, this gives a shape (b, context + response, 1)
        values = self.vf.score(output.hidden_state[-1])

        return values[:, batch["context_length"] - 1 : -1].squeeze(-1)

    def compute_rewards(self, batch: Dict[str, Any]) -> TensorType:

        # Get the value model LLM backbone.
        rm_llm_backbone = getattr(self.rm, self.rm.base_model_prefix)

        output = rm_llm_backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            return_dict=True,
            output_hidden_states=True,
            # We need output for new sequences.
            use_cache=False,
        )

        # Compute values by using the score function with the last hidden
        # state. Note, this gives a shape (b, context + response, 1)
        reward_logits = self.rm.score(output.hidden_state[-1])

        return reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            batch["sequence_lengths"],
        ].squeeze(-1)
