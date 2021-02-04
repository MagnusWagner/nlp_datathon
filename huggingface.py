from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, GPT2Tokenizer
import numpy as np

# def evaluate(train_path, test_path):
train_path = "data/train_en_500.csv"
test_path = "data/train_en_100.csv"

model_checkpoint = "roberta-base" # "distilbert-base-uncased"
batch_size = 1
dataset = load_dataset('csv', data_files={"train": train_path, "validation": test_path})
metric = load_metric("accuracy")
dataset["train"][0]


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    print(df)

show_random_elements(dataset["train"])


fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
sentence1_key = "Narrative"

def preprocess_function(examples):

    return tokenizer(examples[sentence1_key], truncation=True)

print(preprocess_function(dataset['train'][:5]))

encoded_dataset = dataset.map(preprocess_function, batched=True)

num_labels = 7
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"

args = TrainingArguments(
    "huggingface",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
evaluation = trainer.evaluate()


### HYPERPARAM OPTIMISATION
#
# def model_init():
#     return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
#
# trainer = Trainer(
#     model_init=model_init,
#     args=args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset["validation"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize")
#
# for n, v in best_run.hyperparameters.items():
#     setattr(trainer.args, n, v)
#
# trainer.train()

# return evaluation