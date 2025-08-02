from datasets import load_dataset

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
    dataset = load_dataset("pminervini/HaluEval", "qa")['data']
    id_map = {_['question']:str(i) for i, _ in enumerate(dataset)}
    def process_instance(example, prompt_mode='brief'):
        example['id'] = id_map[example['question']]
        example['answer'] = [example["right_answer"]]
        batch_with_prompt = sample_to_prompt(example, prompt_mode)

        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
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
