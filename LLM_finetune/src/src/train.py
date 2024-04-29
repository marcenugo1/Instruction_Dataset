from datasets import load_dataset,load_from_disk
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="deepspeed")
# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load() # needed for now, will be fixed in the future

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="codellama/CodeLlama-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dataset = load_from_disk(data_args.data_path)
    dataset = dataset['train']
    dataset = dataset.remove_columns("label")

    def tokenize_function(example_batch):
        return tokenizer(example_batch['prompt'], truncation=True, max_length=512, padding='max_length')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_cache=False
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False
    )

    tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    tokenizer.add_special_tokens({'eos_token': DEFAULT_EOS_TOKEN})
    tokenizer.add_special_tokens({'bos_token': DEFAULT_BOS_TOKEN})
    tokenizer.add_special_tokens({'unk_token': DEFAULT_UNK_TOKEN})

    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args, train_dataset=tokenized_dataset,
                      data_collator=data_collator)

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
