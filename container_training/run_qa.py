# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import os
import json
import argparse
from tqdm import tqdm
import datasets
from datasets import load_dataset, load_metric, Dataset, Features
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoTokenizer, default_data_collator

data_collator = default_data_collator

def prepare_train_features(examples, tokenizer, max_length, doc_stride):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    pad_on_right = tokenizer.padding_side == "right"

    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def create_squad_dict(actual_squad):
    titles = []
    contexts = []
    ids = []
    questions = []
    answers = []
    for example in tqdm(actual_squad["data"]):
        title = example.get("title", "").strip()
        for paragraph in example["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                question = qa["question"].strip()
                id_ = qa["id"]

                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                answer_list = [answer["text"].strip() for answer in qa["answers"]]
                titles.append(title)
                contexts.append(context)
                questions.append(question)
                ids.append(id_)
                answers.append({
                        "answer_start": answer_starts,
                        "text": answer_list,
                    })

    dataset_dict = {
        "answers":answers,
        "context":contexts,
        "id":ids,
        "question":questions,
        "title":titles,
    }
    return dataset_dict


if __name__ == "__main__":
    
    # Sagemaker configuration
    print('Starting training...')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=None) # os.environ["SM_CHANNEL_TEST"]

    args, _ = parser.parse_known_args()
    
    print('model directory:', args.model_dir)
    print('train directory:', args.training_dir)
    print('output data directory', args.output_data_dir)
    
    os.system('echo training directory contents:')
    os.system(f'ls {args.training_dir}')
    
    hf_args = TrainingArguments(
    args.model_dir,
    evaluation_strategy = "epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    )
    
    with open(args.training_dir+'/v2.0/dev-v2.0.json', 'r') as f:
        squad_dev = json.load(f)
    with open(args.training_dir+'/augmented_squad.json', 'r') as f:
        actual_squad = json.load(f)
        
    #datasets = load_dataset("squad_v2") ## NEED TO COMBINE WITH OUR LABELS
    dataset_dict = create_squad_dict(actual_squad)
    test_dataset_dict = create_squad_dict(squad_dev)
    
    squad_dataset = Dataset.from_dict(dataset_dict,
                                 features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    # These are the features of your dataset like images, labels ...
                }
            ))
    squad_test = Dataset.from_dict(test_dataset_dict,
                                 features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    
#     features = prepare_train_features(datasets['train'][:5], tokenizer)
    tokenized_train = squad_dataset.map(prepare_train_features, batched=True, remove_columns=squad_dataset.column_names, fn_kwargs = {'tokenizer':tokenizer, 'max_length':args.max_length, 'doc_stride':args.doc_stride})
    tokenized_test = squad_test.map(prepare_train_features, batched=True, remove_columns=squad_test.column_names, fn_kwargs = {'tokenizer':tokenizer, 'max_length':args.max_length, 'doc_stride':args.doc_stride})

    trainer = Trainer(
        model,
        hf_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    if args.test_dir:
        # evaluate model
        eval_result = trainer.evaluate(eval_dataset=test_dataset)

        # writes eval result to file which can be accessed later in s3 ouput
        with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
            print(f"***** Eval results *****")
            for key, value in sorted(eval_result.items()):
                writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.model_dir)
    os.system(f'ls {args.model_dir}')
    
    
