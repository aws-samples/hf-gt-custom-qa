# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import logging
import sagemaker_containers
import requests

import os
import json
import io
import time
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
CONTENT_TYPE = 'text/plain'


def embed_tformer(model, tokenizer, sentences):
    # encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=256, return_tensors='pt')
    try:
        question, context = sentences.split('|')
    except:
        question, context = sentences[0].split('|')
        print('Had to remove list!!!')
    inputs = tokenizer(question, context, return_tensors='pt')
    input_ids = inputs["input_ids"].tolist()[0]
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])

    with torch.no_grad():
        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    outputs = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return outputs

def model_fn(model_dir):
    logger.info('model_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir) # "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    nlp_model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    nlp_model.to(device)
    model = {'model':nlp_model, 'tokenizer':tokenizer}

#     logger.info(model)
    return model

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(serialized_input_data, content_type=CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    try:
#         if content_type == CONTENT_TYPE:
        data = [serialized_input_data.decode('utf-8')]
        print(data)
        return data
    except:
        raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    response = embed_tformer(model['model'], model['tokenizer'], input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
#     response = sentence_embeddings[0].tolist()
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept):
    logger.info('Serializing the generated output.')
    if accept == 'application/json':
        output = json.dumps(prediction)
        return output
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(content_type))
