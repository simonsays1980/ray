## Nemotron
# SFT: nvidia/nemotron-3-8b-chat-4k-sft

## OpenRLHF
# SFT: OpenRLHF/Llama-3-8b-sft-mixture
# RM: OpenRLHF/Llama-3-8b-rm-mixture
# DS: OpenRLHF/prompt-collection-v0.1

## EleutherAI
# SFT: cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr
# SFT2: EleutherAI/pythia-1b-deduped
# RM: cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr
# RM2: EleutherAI/pythia-160m
# https://huggingface.co/docs/trl/main/ppo_trainer

# SFT Dataset TL;DR
# https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered
#import pandas as pd
import ray

from ray.rllib.utils.framework import try_import_torch
#from ray.rllib.utils.torch_utils import convert_to_torch_tensor

# ctx = ray.data.DataContext.get_current()
# ctx.log_internal_stack_trace_to_stdout = True

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, GenerationConfig
from ray.rllib.utils.framework import try_import_torch
from transformers.generation.stopping_criteria import STOP_STRING_EMBEDDING_CACHE

torch, nn = try_import_torch()

TEMPERATURE = 0.7
SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
STOP_TOKEN_ID = 0
MISSING_EOS_PENALTY = 1.0
INVALID_LOGPROB = 1.0
KL_COEFF = 0.05

# This is used in the PPO TL;DR example.
dataset_name = "trl-internal-testing/tldr-preference-sft-trl-style"

splits = {
    "train": "data/train-*.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
hf_dataset_stream = load_dataset(dataset_name, streaming=True)
ds_train = ray.data.from_huggingface(hf_dataset_stream["train"])

# ds_train = ray.data.from_pandas(df_train).map(prepare_prompts).drop_columns(["id", "subreddit", "title", "post", "summary"])

# row = ds_train.take(1)
# print(row)
# ds2 = load_dataset(dataset_name)["train"]
# reward_model_id = "Ray2333/GRM-gemma2-2B-rewardmodel-ft"

sft_model_id = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
reward_model_id = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"

tokenizer = AutoTokenizer.from_pretrained(
    sft_model_id,
    padding_side="left",
    trust_remote_code=True,
)
# tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE


def tokenize(element):
    #print(f"===> messages: {element['messages']}")
    input_ids = tokenizer.apply_chat_template(
        element["messages"][:1],
        padding=False,
        add_generation_prompt=True,
    )
    return [{"input_ids": input_ids, "lengths": len(input_ids)}] if len(input_ids) <= 512 else []


#ds_train_mapped = ds_train.map(tokenize).filter(lambda x: x["lengths"] <= 512)
ds_train_mapped = ds_train.flat_map(tokenize)

policy = AutoModelForCausalLM.from_pretrained(
    sft_model_id,
    trust_remote_code=True,
)
reference_policy = AutoModelForCausalLM.from_pretrained(
    sft_model_id,
    trust_remote_code=True,
)

value_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_id,
    trust_remote_code=True,
    num_labels=1,
)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_id,
    trust_remote_code=True,
    num_labels=1,
)

generation_config = GenerationConfig(
    max_new_tokens=53, # response_length
    temperature=(TEMPERATURE + 1e-7),
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
)
# generation_config.eos_token_id = (None,)
# generation_config.pad_token_id = None
train_batch = ds_train_mapped.take(4)
print(train_batch)
#input_ids = data_collator.return_tensors(train_batch["input_ids"])
# input_ids = torch.Tensor(train_batch["input_ids"].tolist())
tokenized_inputs = tokenizer.pad(
    train_batch, 
    max_length=None, 
    padding=True, 
    pad_to_multiple_of=None, 
    return_tensors="pt"
)

# TODO (simon): This happens in a loop. Try out, if this works as well
# in batched form.
# -------------- Generation ------------------------------------------
# print(f"Type of input_ids: {type(input_ids)}")
# attention_mask = input_ids != tokenizer.pad_token_id
# input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)
output = policy.generate(
    tokenized_inputs["input_ids"],
    attention_mask=tokenized_inputs["attention_mask"],
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
)
print(output)

logits = torch.stack(output.scores, dim=1)
queries = tokenized_inputs["input_ids"]
context_length = queries.shape[1]
# The output contains masked values. We need however the [PAD] token
# instead.
query_responses = torch.cat((queries, output.sequences[:, context_length:]), dim=1)

# TODO (simon): Some models might need here a posterior padding b/c
# they output answers that are of various length.
# query_responses = tokenizer.pad()
# Same holds true for the logits.
# ---------------------------------------------------------------------
# Log probabilities for all logits (each logits stands for a token).
all_logprobs = nn.functional.softmax(logits, dim=-1)
# Get only the response token IDs.
responses = query_responses[:, context_length:]
# Gather over the tokens in the responses and collect their log probabilities.
logprobs = torch.gather(all_logprobs, dim=2, index=responses.unsqueeze(-1)).squeeze(-1)
# Clean up as fast as possible.
# del all_logrops, logits
# torch.cuda.empty()

# NOTE (simon): Processing class is the tokenizer.
# Run the reference policy.
# Create an attention mask for context + response.
attention_mask = query_responses != tokenizer.pad_token_id
# Generate position ids in the context + response that define at which
# position in the whole (unpadded) sequence a certain token (not padded)
# is located. Start at zero index (subtract 1).
position_ids = attention_mask.cumsum(dim=1) - attention_mask.long()
# Generate a masked torch tensor and mask with 0.
masked_input_ids = torch.masked_fill(query_responses, ~attention_mask, value=0)
reference_output = reference_policy.forward(
    masked_input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    return_dict=True,
    output_hidden_states=True,
)
# Get the logits from the response only.
reference_logits = reference_output.logits[:, context_length - 1 : -1]
# Scale the logits by the temperature.
reference_logits /= TEMPERATURE + 1e-7
# Log probabilities for all logits.
reference_all_logprobs = nn.functional.softmax(reference_logits, dim=-1)
# Gather only the log-probabilities for the tokens in the response.
reference_logprobs = torch.gather(reference_all_logprobs, dim=2, index=responses.unsqueeze(-1)).squeeze(-1)
# Clean up as fast as possible.
del reference_output, reference_logits, reference_all_logprobs
torch.cuda.empty_cache()

postprocessed_responses = responses
# Handle edge cases where the stop token ID is zero (which would be
# the same value like the mask).
if STOP_TOKEN_ID is not None:
    # In this case we need to identify the first location of a zero
    # after the initial token. We do this by creating a boolean
    # tensor and add the sequence length to all non-stop tokens (non-zero).
    # Then we add an increasing sequence and take the mininum for each
    # response.

    # Get the indices at which the stop token occurred the first time
    # in the response and pad the responses from thereon.
    stop_token_locations = postprocessed_responses == STOP_TOKEN_ID
    # Now get the indices of the first `True` (zero).
    first_zero = torch.max(stop_token_locations.type(torch.long), dim=-1).indices.unsqueeze(-1)
    first_zero[first_zero == 0] = postprocessed_responses.shape[-1]
    # Mask now the zero tokens in the responses with the pad token ID.
    view_shape = [1] * (len(postprocessed_responses.size()) - 1) + [postprocessed_responses.shape[1]]
    batch_indices = torch.arange(postprocessed_responses.shape[1], device=postprocessed_responses.device).view(*view_shape)
    postprocessed_responses = torch.masked_fill(postprocessed_responses, batch_indices > first_zero, tokenizer.pad_token_id)


# Concatenate the postprocessed responses with the queries.
postprocessed_query_responses = torch.cat([queries, postprocessed_responses], dim=1)
# Get the sequence lengths.
pad_token_locations = (postprocessed_responses == tokenizer.pad_token_id)
first_pad_tokens = torch.max(pad_token_locations.type(torch.long), dim=-1).indices.unsqueeze(-1)
# Note, the sequence ends before the padding token.
sequence_lengths = first_pad_tokens - 1

# ---------------------VALUE MODEL -----------------------------------------
# Run the value model.
attention_mask = query_responses != tokenizer.pad_token_id
position_ids = attention_mask.cumsum(dim=1) - attention_mask.long()
input_ids = torch.masked_fill(query_responses, ~attention_mask, value=0)
# Get the language model backbone.
value_model_backbone = getattr(value_model, value_model.base_model_prefix)
output = value_model_backbone(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    return_dict=True,
    output_hidden_states=True,
    # We need output for new sequences.^
    use_cache=False,
)
# Compute values by using the score function with the last hidden state.
# Note, this gives a shape (b, context + response, 1)
values = value_model.score(output.hidden_states[-1])
# Only use the value logits of the responses.
values = values[:, context_length - 1 : -1].squeeze(-1)

# --------------------- REWARD MODEL -------------------------------------
# Run the reward model.
attention_mask = postprocessed_query_responses != tokenizer.pad_token_id
position_ids = attention_mask.cumsum(dim=1) - attention_mask.long()
input_ids = torch.masked_fill(postprocessed_query_responses, ~attention_mask, value=0)
reward_model_backbone = getattr(reward_model, reward_model.base_model_prefix)
output = reward_model_backbone(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    return_dict=True,
    output_hidden_states=True,
    # We need output for new sequences.
    use_cache=False,
)
# Compute values by using the score function with the last hidden state.
# Note, this gives a shape (b, context + response, 1)
reward_logits = reward_model.score(output.hidden_states[-1])
# Use only rewards from the last token.
scores = reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1)



# Next processing: Ensure that all responses contain the stop token and
# penalize, if not.
contain_eos_token = torch.any(postprocessed_responses == STOP_TOKEN_ID, dim=-1)
if MISSING_EOS_PENALTY:
    scores[~contain_eos_token] -= MISSING_EOS_PENALTY

# Pad the logprobs with the INVALID_LOGPROB (i.e. where responses are 
# shorter than the sequence).
response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
padding_mask = response_idxs > sequence_lengths
logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
reference_logprobs = torch.masked_fill(reference_logprobs, padding_mask, INVALID_LOGPROB)
# Now the same padding for the values, but including the EOS token.
sequence_lengths_p1 = sequence_lengths + 1
padding_mask_p1 = response_idxs > sequence_lengths_p1.unsqueeze(-1)
values = torch.masked_fill(values, padding_mask_p1, 0)


# --------------------- Rewards ----------------------------------
# Now compute RL rewards.
# Each token gets a KL reward for being near in probability to the reference policy.
kl = logprobs - reference_logprobs
kl_reward = -KL_COEFF * kl
# Now add the reward for the whole sequence to the one with the EOS Token (or last token 
# when no EOS token (this might got penalized before)).
rewards = kl_reward.clone()
actual_end_of_response = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
rewards[[torch.arange(rewards.size(0), device=rewards.device), actual_end_of_response]] += scores
print(rewards)
# # Create SingleAgentEpisode with entries.