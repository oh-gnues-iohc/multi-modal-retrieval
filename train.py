from transformers import AutoImageProcessor, ResNetForImageClassification, AutoTokenizer
import torch
from models.model import ImageTextRetrieval, ImageTextRetrievalConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import TrainingArguments, Trainer, HfArgumentParser
import logging
import os
from dataclasses import dataclass, field
import transformers
from typing import Union

@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        default="ohgnues/ImageTextRetrieval"
    )
    use_auth_token: str = field(
        default=None, metadata={"help": "Authentication token required for private model usage"}
    )

@dataclass
class DataArguments:
    path: str = field(
        default="poloclub/diffusiondb", metadata={"help": "Path or name of the dataset"}
    )
    name: str = field(
        default=None, metadata={"help": "Subset name"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Location to store cache files"}
    )
    train_split: str = field(
        default="train", metadata={"help": "Name of the training data"}
    )
    eval_split: str = field(
        default=None, metadata={"help": "Name of the evaluation data"}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the data"}
    )
    text_column_name: str = field(
        default="prompt", metadata={"help": "Column name for text data"}
    )
    image_column_name: str = field(
        default="image", metadata={"help": "Column name for image data"}
    )
    max_length: int = field(
        default=512, metadata={"help": "Maximum token length"}
    )


@dataclass
class TrainArguments(TrainingArguments):
    output_dir: str = "runs/"
    do_train: bool = True
    do_eval: bool = False
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 8
    num_train_epochs: float = 5.0
    learning_rate: float = 5e-5
    save_strategy: Union[transformers.trainer_utils.IntervalStrategy, str] = 'epoch'
    



if __name__ == "__main__":
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    model = ImageTextRetrieval.from_pretrained(**vars(model_args))
    tokenizer = AutoTokenizer.from_pretrained(**vars(model_args))
    processor = AutoImageProcessor.from_pretrained(**vars(model_args))

    if os.path.isdir(data_args.path):
        dataset = load_from_disk(data_args.path)
    else:
        dataset = load_dataset(data_args.path, data_args.name, cache_dir=data_args.cache_dir)

    if data_args.shuffle:
        dataset = dataset.shuffle()


    def example_function(examples):

        tokenized_text = tokenizer(
            examples[data_args.text_column_name],
            truncation=True,
            padding="max_length",
            max_length=data_args.max_length,
            return_tensors="pt"
        )

        processed_image = processor(examples[data_args.image_column_name], return_tensors="pt")

        tokenized_text.update(processed_image)

        return tokenized_text

    dataset = dataset.map(example_function, batched=True, batch_size=10000, remove_columns=dataset[data_args.train_split].column_names)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.eval_split] if data_args.eval_split else None,
    )

    trainer.train()
