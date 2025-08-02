import functools
import json
import os

import datasets
import pandas as pd
from datasets import Dataset, load_dataset

# import _settings

# DATA_FOLDER = "/proj/csc266/scratch/wgao23/.cache/huggingface/offline/datasets"

# def _save_dataset():
#     # https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
#     save_path = f'{DATA_FOLDER}/SQuAD'
#     if not os.path.exists(save_path):
#         # https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json
#         with open('{}/dev-v2.0.json'.format(DATA_FOLDER), 'r') as infile:
#             data = json.load(infile)['data']

#         dataset = {}

#         dataset['story'] = []
#         dataset['question'] = []
#         dataset['answer'] = []
#         dataset['additional_answers'] = []
#         dataset['id'] = []

#         for _data in data:
#             paragraphs = _data["paragraphs"]
#             for sample_id, sample in enumerate(paragraphs):
#                 print(sample)
#                 story = sample['context']
#                 questions = sample['qas']
#                 # answers = sample['answers']
#                 # additional_answers = sample['additional_answers']
#                 for question_index, question in enumerate(questions):
#                     if question["is_impossible"]:
#                         continue
#                     dataset['story'].append(story)
#                     dataset['question'].append(question['question'])
#                     dataset['answer'].append({
#                         'text': question["answers"][0]['text'],
#                         'answer_start': question["answers"][0]['answer_start']
#                     })
#                     dataset['id'].append(question['id'])
#                     additional_answers_list = []
#                     for i in range(len(question["answers"])):
#                         additional_answers_list.append(question["answers"][i]['text'])
#                     dataset['additional_answers'].append(additional_answers_list)

#         dataset_df = pd.DataFrame.from_dict(dataset)

#         dataset = Dataset.from_pandas(dataset_df)

#         dataset.save_to_disk(save_path)
#     return save_path

# @functools.lru_cache(1)
# def read_all_contexts():
#     dataset = datasets.load_from_disk(_save_dataset())
#     return {_['id']: _['story'] for _ in dataset}



# def get_dataset(tokenizer, split='validation'):
#     # from https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_coqa.py
#     dataset = datasets.load_from_disk(_save_dataset())
#     dataset = datasets.load_dataset("squad_v2")
#     id_to_question_mapping = dict(zip(dataset['id'], dataset['question']))

#     def encode_coqa(example):
#         example['answer'] = example['answer']['text']
#         example['prompt'] = prompt = example['story'] + ' Q: ' + example['question'] + ' A:'
#         return tokenizer(prompt, truncation=False, padding=False)
#     dataset = dataset.map(encode_coqa, batched=False, load_from_cache_file=False)
#     dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
#     return dataset

# def _generate_config(tokenizer):
#     if tokenizer.__class__.__name__ in ['LlamaTokenizer', 'LlamaTokenizerFast']:
#         eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
#         #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
#     elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
#         eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', ',', '.']]
#     elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
#         eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
#     else:
#         raise NotImplementedError
#     eos_token_id += [tokenizer.eos_token_id]
#     #bad_words_ids = [tokenizer(_)['input_ids'][1] for _ in ['Q:']] # only "Q"
#     bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']] # only "Q"
#     return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

# if __name__ == '__main__':
#     import models
#     dataset = get_dataset(models.load_tokenizer())

def _generate_config(tokenizer):
    if tokenizer.__class__.__name__ in ['LlamaTokenizer', 'LlamaTokenizerFast']:
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
        #eos_token_id = [tokenizer(_)['input_ids'] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == 'GPT2Tokenizer':
        eos_token_id = [tokenizer.encode(_)[1] for _ in ['\n', ',', '.']]
    elif tokenizer.__class__.__name__ == "PreTrainedTokenizerFast":
        eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    else:
        raise NotImplementedError
    eos_token_id += [tokenizer.eos_token_id]
    #bad_words_ids = [tokenizer(_)['input_ids'][1] for _ in ['Q:']] # only "Q"
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']] # only "Q"
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

def sample_to_prompt(sample, prompt_mode='brief', **kwargs):
    if isinstance(sample['question'], list):
        return [sample_to_prompt({'question': _}, prompt_mode, **kwargs) for _ in sample['question']]
    if prompt_mode == 'brief':
        return f"""Answer these questions:
    Q: In Scotland a bothy/bothie is a?
    A: House
    Q: {sample['question']}
    A:"""
    elif prompt_mode == 'sentence':
        return f"""Answer the following question in a single brief but complete sentence.\nQuestion: {sample['question']}\nAnswer:"""
    else:
        raise ValueError(f"Invalid prompt mode: {prompt_mode}")

def get_dataset(tokenizer, prompt_mode='brief'):
    dataset = load_dataset("squad_v2")['validation']
    def process_instance(example, prompt_mode=prompt_mode):
        example['answer'] = [example['answers']["text"]]
        batch_with_prompt = sample_to_prompt(example, prompt_mode)

        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        # outputs = tokenizer(example['answer'], padding=False, truncation=False)
        example["input_ids"] = inputs.input_ids
        example["attention_mask"] = inputs.attention_mask
        # example['prompt'] = batch_with_prompt

        return example
    data = dataset.map(lambda _: process_instance(_, prompt_mode=prompt_mode))
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True)

    return data
