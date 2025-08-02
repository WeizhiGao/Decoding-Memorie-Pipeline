# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import MODEL_PATH


@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    if model_name == "microsoft/deberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "llama2-7b-chat":
        # download the model for offline usage
        # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", max_memory={0: '80GIB'})
        # model.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-7b-chat")
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), device_map="auto", max_memory={0: '80GIB'})
    elif model_name == 'llama2-13b-chat':
        # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto", max_memory={0: '80GIB'})
        # model.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-13b-chat")
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), device_map="auto", max_memory={0: '80GIB'})
    elif model_name == "mistral-7b-instruct":
        # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto', max_memory={0: '80GIB'})
        # model.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/mistral-7b-instruct")
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), device_map="auto", max_memory={0: '80GIB'})
    elif model_name == "falcon-7b-instruct":
        # model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", device_map='auto', max_memory={0: '80GIB'})
        # model.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/falcon-7b-instruct")
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), device_map="auto", max_memory={0: '80GIB'})
    elif model_name == 'roberta-large-mnli':
         model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name == "microsoft/deberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    elif model_name == "llama2-7b-chat":
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", token_type_ids=None)
        # tokenizer.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-7b-chat")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name))
    elif model_name == "llama2-13b-chat":
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto", token_type_ids=None)
        # tokenizer.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-13b-chat")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name))
    elif model_name == "mistral-7b-instruct":
    #     tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
    #     tokenizer.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/mistral-7b-instruct")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name))
    elif model_name == "falcon-7b-instruct":
        # tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", device_map="auto", token_type_ids=None)
        # tokenizer.save_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/falcon-7b-instruct")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name))
        
    return tokenizer