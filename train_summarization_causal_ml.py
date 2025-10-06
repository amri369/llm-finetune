import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import evaluate
import nltk
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, TaskType

from dotenv import load_dotenv

load_dotenv()

check_min_version("4.57.0.dev0")

logger = logging.getLogger(__name__)

nltk.download('punkt_tab', quiet=True)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub."},
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout probability."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the metrics (rouge) on."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (rouge) on."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "The maximum total sequence length for target text after tokenization."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of evaluation examples."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of prediction examples."},
    )
    source_prefix: Optional[str] = field(
        default="Summarize: ", metadata={"help": "A prefix to add before every source text."}
    )

    def __post_init__(self):
        if (
                self.dataset_name is None
                and self.train_file is None
                and self.validation_file is None
                and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


summarization_name_mapping = {
    "cnn_dailymail": ("article", "highlights"),
    "xsum": ("document", "summary"),
    "samsum": ("dialogue", "summary"),
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Load datasets
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Set pad token for causal LM if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )

    if model_args.use_lora and training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if model_args.use_lora:
        logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Resize token embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Get column names
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    max_seq_length = data_args.max_source_length + data_args.max_target_length

    def preprocess_function(examples):
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        # Format for causal LM: "<prefix><input>\nSummary: <target><eos>"
        texts = []
        for inp, tgt in zip(inputs, targets):
            text = f"{prefix}{inp}\nSummary: {tgt}{tokenizer.eos_token}"
            texts.append(text)

        model_inputs = tokenizer(
            texts,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
        )

        # Create labels: mask the input portion, only compute loss on summary
        labels = []
        for i in range(len(texts)):
            # Tokenize input portion to find where summary starts
            input_text = f"{prefix}{inputs[i]}\nSummary: "
            input_tokens = tokenizer(input_text, add_special_tokens=False)["input_ids"]
            input_length = len(input_tokens)

            # Get the full input_ids for this example
            full_ids = model_inputs["input_ids"][i]

            # Create label: -100 for input tokens and padding, actual ids for summary
            label = []
            for j, token_id in enumerate(full_ids):
                if j < input_length:
                    label.append(-100)  # Mask input
                elif token_id == tokenizer.pad_token_id:
                    label.append(-100)  # Mask padding
                else:
                    label.append(token_id)  # Keep summary tokens

            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    # Preprocess datasets
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("rouge", cache_dir=model_args.cache_dir)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Convert logits to token IDs
        preds = np.argmax(preds, axis=-1)  # <-- Add this line

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Extract only the summary part (after "Summary: ")
        decoded_preds = [pred.split("Summary: ")[-1] if "Summary: " in pred else pred for pred in decoded_preds]

        # Decode labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [len(pred.split()) for pred in decoded_preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluate on training set (subset to avoid OOM)
        logger.info("*** Evaluate on Training Set ***")
        train_eval_subset = train_dataset.select(range(min(200, len(train_dataset))))
        train_metrics = trainer.evaluate(
            eval_dataset=train_eval_subset,
            metric_key_prefix="train_eval"
        )
        trainer.log_metrics("train_eval", train_metrics)
        trainer.save_metrics("train_eval", train_metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        # First get metrics using trainer.predict
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Now do PROPER generation from raw test data
        logger.info("*** Generating predictions with model.generate() ***")

        # Load raw test data (before preprocessing)
        if data_args.dataset_name:
            raw_test = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split="test"
            )
        else:
            raw_test = load_dataset("json", data_files={"test": data_args.test_file}, split="test")

        if data_args.max_predict_samples:
            raw_test = raw_test.select(range(min(data_args.max_predict_samples, len(raw_test))))

        # Get column names
        dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
        text_column = data_args.text_column or (dataset_columns[0] if dataset_columns else "article")
        summary_column = data_args.summary_column or (dataset_columns[1] if dataset_columns else "highlights")

        model.eval()
        predictions = []
        references = []

        batch_size = training_args.per_device_eval_batch_size
        for i in range(0, len(raw_test), batch_size):
            batch = raw_test[i:i + batch_size]

            # Format inputs
            input_texts = [f"{prefix}{text}\nSummary: " for text in batch[text_column]]

            # Tokenize
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=data_args.max_source_length
            ).to(training_args.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=data_args.max_target_length,
                    num_beams=4,  # Use beam search
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 🔥 NEW: Slice off the input tokens, keep only generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_length:]

            # 🔥 MODIFIED: Decode only the generated tokens
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # 🔥 SIMPLIFIED: No need to split anymore
            for pred in decoded:
                predictions.append(pred.strip())

            references.extend(batch[summary_column])

            if (i // batch_size) % 10 == 0:
                logger.info(f"Generated {len(predictions)}/{len(raw_test)} predictions")

        # Save predictions
        output_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(predictions))

        # Save sample comparisons
        sample_file = os.path.join(training_args.output_dir, "sample_predictions.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            for i in range(min(10, len(predictions))):
                f.write(f"=== Example {i + 1} ===\n")
                f.write(f"Input: {raw_test[text_column][i][:200]}...\n")
                f.write(f"Reference: {references[i]}\n")
                f.write(f"Prediction: {predictions[i]}\n\n")

        logger.info(f"Predictions saved to {output_file}")

    return {}


if __name__ == "__main__":
    main()