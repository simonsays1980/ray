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
# import pandas as pd
import ray

from ray.rllib.utils.framework import try_import_torch

# from ray.rllib.utils.torch_utils import convert_to_torch_tensor

# ctx = ray.data.DataContext.get_current()
# ctx.log_internal_stack_trace_to_stdout = True

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    GenerationConfig,
)
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

TEMPERATURE = 0.7
SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

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

tokenizer = AutoTokenizer.from_pretrained(
    sft_model_id,
    padding_side="left",
    trust_remote_code=True,
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE


def tokenize(element):
    # print(f"===> messages: {element['messages']}")
    input_ids = tokenizer.apply_chat_template(
        element["messages"][:1],
        padding=False,
        add_generation_prompt=True,
    )
    return {"input_ids": input_ids, "lengths": len(input_ids)}


ds_train_mapped = ds_train.map(tokenize).filter(lambda x: x["lengths"] <= 512)

policy = AutoModelForCausalLM.from_pretrained(
    sft_model_id,
    trust_remote_code=True,
)

generation_config = GenerationConfig(
    max_new_tokens=53,
    temperature=(TEMPERATURE + 1e-7),
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
)
generation_config.eos_token_id = (None,)
generation_config.pad_token_id = None
train_batch = ds_train_mapped.take(4)
print(train_batch)
# input_ids = data_collator.return_tensors(train_batch["input_ids"])
# input_ids = torch.Tensor(train_batch["input_ids"].tolist())
tokenized_inputs = tokenizer.pad(
    train_batch,
    max_length=None,
    padding=True,
    pad_to_multiple_of=None,
    return_tensors="pt",
)
input_ids = train_batch["input_ids"]
print(f"Type of input_ids: {type(input_ids)}")
attention_mask = input_ids != tokenizer.pad_token_id
input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)
output = policy.generate(
    input_ids,
    attention_mask=attention_mask,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
)
print(output)
# policy.foward()
