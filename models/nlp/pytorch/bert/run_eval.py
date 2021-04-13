import logging
import os
import sys
from dataclasses import dataclass, field
import math
import json

from datasets import load_dataset
import torch # to load dataset

import transformers
from transformers import (
    BertConfig,
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:

    train_dataset: str = field(
        default="phase1_data.pt",
        metadata={
            "help": "Where preprocess_data.py saved the dataset"
        }
    )

    validation_split_percentage: float = field(
        default=0.15,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    path_to_checkpoints: str = field (
        default="./bert_phase1/"
    )


def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load preprocessed datasets
    datasets = torch.load(data_args.train_dataset)
    datasets = datasets.train_test_split(test_size=data_args.validation_split_percentage)
    eval_dataset = datasets["test"]

    # set config, tokenizer, data collator
    config = BertConfig()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    # get all checkpoints
    content = os.listdir(data_args.path_to_checkpoints)
    checkpoints = [os.path.join(data_args.path_to_checkpoints, path) \
        for path in content \
            if 'checkpoint' in path and os.path.isdir(os.path.join(data_args.path_to_checkpoints, path))]

    # run evaluation on checkpoints
    log_metrics = []
    for checkpoint in checkpoints:
        logger.info(f"Loading model from {checkpoint}")
        model = BertForMaskedLM(config=config)
        model_state = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
        model.load_state_dict(model_state)
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        metrics = trainer.evaluate()
        metrics["step"] = int(checkpoint.split("-")[-1])

        logger.info(metrics)
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
        log_metrics.append(metrics)
    eval_log = {"log_history": log_metrics}

    filename = os.path.join(training_args.output_dir, "offload_eval_results.json")
    with open(filename, 'w') as f:
        json.dump(eval_log , f, indent=4)

if __name__ == "__main__":
    main()
