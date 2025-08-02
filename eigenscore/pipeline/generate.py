import argparse
import glob
import json
import os
import copy
import time
import logging
import gc
from typing import Optional, Tuple

import pandas as pd
import torch
import tqdm
import transformers
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore
from sklearn.metrics import roc_auc_score
from transformers.generation.utils import ModelOutput, dataclass, TopKLogitsWarper, TopPLogitsWarper
from torch.nn.functional import softmax

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import dataeval.truthfulqa as truthfulqa
import dataeval.sciq as sciq
import dataeval.halueval as halueval
import models
import utils
from func.metric import *
from models.openai_models import openai_query

from transformers.cache_utils import Cache

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama2-7b-chat')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
parser.add_argument('--prompt_mode', type=str, default='brief', choices=['brief', 'sentence'])
parser.add_argument('--num_samples', type=int, default=400)
parser.add_argument('--experiment_lot', type=str, default='test')
parser.add_argument('--reuse_mode', type=str, default=None, choices=[None, 'hard', 'soft'])
parser.add_argument('--reuse_prob', type=float, default=0.85)
parser.add_argument('--enable_reuse', action='store_true', help='Enable reuse of cached sentences')
parser.add_argument('--enable_reweight', action='store_true', help='Enable reweight of cached sentences')
parser.add_argument('--reweight_scaling', type=float, default=1.1)
parser.add_argument('--reweight_threshold', type=float, default=0.9)
parser.add_argument('--short_answer_length', type=int, default=10)


args = parser.parse_args()

# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'triviaqa':
        return triviaqa.get_dataset
    if data_name == 'coqa':
        return coqa.get_dataset
    if data_name == 'nq_open':
        return nq_open.get_dataset
    if data_name == "SQuAD":
        return SQuAD.get_dataset
    if data_name == "truthfulqa":
        return truthfulqa.get_dataset
    if data_name == "sciq":
        return sciq.get_dataset
    if data_name == "halueval":
        return halueval.get_dataset


def get_generation_config(input_ids, tokenizer, data_name):
    assert len(input_ids.shape) == 2
    max_length_of_generated_sequence = 256
    if data_name == 'triviaqa':
        generation_config = triviaqa._generate_config(tokenizer)
    if data_name == 'coqa':
        generation_config = coqa._generate_config(tokenizer)
    if data_name == 'nq_open':
        generation_config = nq_open._generate_config(tokenizer)
    if data_name == 'SQuAD':
        generation_config = SQuAD._generate_config(tokenizer)
    if data_name == 'truthfulqa':
        generation_config = truthfulqa._generate_config(tokenizer)
    if data_name == 'sciq':
        generation_config = sciq._generate_config(tokenizer)
    if data_name == 'halueval':
        generation_config = halueval._generate_config(tokenizer)
    generation_config['max_new_tokens'] = max_length_of_generated_sequence
    generation_config['early_stopping'] = True
    # https://jaketae.github.io/study/gpt2/#setup
    generation_config['pad_token_id'] = tokenizer.eos_token_id
    return generation_config


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device)
    SenSimModel = SentenceTransformer('nli-roberta-large')
    # bertscore = BERTScore(model_name_or_path="bert-base-uncased", device="cuda")
    bertscore = BERTScore(model_name_or_path="/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/bert-base-uncased", device="cuda")

    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer, prompt_mode=args.prompt_mode)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['id']: _ for _ in old_sequences}

    sequences = []
    total_time = 0
    # for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    total_generated_tokens = 0
    total_reuse_tokens = 0
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=args.num_samples):
        instance_generated_tokens = 0
        instance_reuse_tokens = 0
        total_time_instance = 0
        if batch['id'][0] in old_sequences:
            sequences.append(old_sequences[batch['id'][0]])
            continue

        if batch_idx >= args.num_samples:
            break

        if isinstance(batch['answer'][0], list):
            batch['answer'] = [[a[0] for a in batch['answer'][0]]]

        input_ids = batch['input_ids'].to(device)
        input_length = input_ids.shape[1]
        generation_config = get_generation_config(input_ids, tokenizer, args.dataset)
        generation_config = transformers.GenerationConfig(**generation_config)

        time_start = time.time()
        if args.decoding_method == 'beam_search':
            raise NotImplementedError()
        elif args.decoding_method == 'greedy':
            if args.enable_reuse:
                dict_outputs, cache, (num_generated_tokens, num_reuse_tokens) = generate_with_cache(
                    model, batch, max_new_tokens=100, output_scores=True, output_logits=True, output_hidden_states=True, 
                    do_sample=False, temperature=args.temperature, generation_config=generation_config)
                instance_generated_tokens += num_generated_tokens
                instance_reuse_tokens += num_reuse_tokens
            else:
                dict_outputs = model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                            num_beams=1,
                                            do_sample=False,
                                            generation_config=generation_config,
                                            output_hidden_states=True,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            output_logits=True)

            scores = dict_outputs.scores    #([logits],[logits],[logits])
            perplexity = get_perplexity_score(scores)
            energy_score = get_energy_score(scores)
            most_likely_generations = dict_outputs.sequences.cpu()[0, input_length:]
            print(f"\nMost likely generations: {tokenizer.decode(most_likely_generations, skip_special_tokens=True)}")

        torch.cuda.empty_cache()
        generations = []
        hidden_states_list = []
        scores_list = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            if args.enable_reuse:
                dict_outputs, cache, (num_generated_tokens, num_reuse_tokens) = generate_with_cache(
                    model, batch, max_new_tokens=100, output_scores=True, output_hidden_states=True, 
                    temperature=args.temperature, do_sample=True, generation_config=generation_config, cache=cache)
                instance_generated_tokens += num_generated_tokens
                instance_reuse_tokens += num_reuse_tokens
            else:
                dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                num_beams=1, num_return_sequences=1,
                                do_sample=True, top_p=args.top_p, top_k=args.top_k,
                                temperature=args.temperature, generation_config=generation_config,
                                output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                                )

            generation = dict_outputs.sequences[:, input_length:].cpu()
            generations.append(generation)
            scores_list.append(dict_outputs.scores)
            hidden_states_list.append(dict_outputs.hidden_states)
            num_gens -= len(generation)

            print(f"{len(generations)}th generations: {tokenizer.decode(generation[0], skip_special_tokens=True)}")
        time_end = time.time()
        total_time_instance += time_end - time_start

        if args.enable_reuse:
            total_generated_tokens += instance_generated_tokens
            total_reuse_tokens += instance_reuse_tokens
            instance_reuse_ratio = instance_reuse_tokens / instance_generated_tokens
            logging.info(f"Instance generated tokens: {instance_generated_tokens}, Instance reuse tokens: {instance_reuse_tokens}, Instance reuse ratio: {instance_reuse_ratio}")
        
        predictive_entropy = get_lenghthNormalized_entropy_list(scores_list) 
        eigenIndicator, eigenValue = getEigenIndicator_v0_list(hidden_states_list)
        generations = torch.nested.nested_tensor(generations).to_padded_tensor(tokenizer.eos_token_id)
        generations = generations.reshape(-1, generations.shape[-1])[:args.num_generations_per_prompt]
        best_generated_text = tokenizer.decode(most_likely_generations, skip_special_tokens=True)
        generated_texts = [tokenizer.decode(_, skip_special_tokens=True) for _ in generations]
        lexical_similarity = getLexicalSim(generated_texts)
        sent_bertscore = getAvgBertScore(bertscore, best_generated_text, generated_texts)
        eigenIndicatorOutput, eigenValue_O = getEigenIndicatorOutput(generated_texts, SenSimModel)


        # remember the data
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['id'][0],
            question=batch['question'][0],
            answer=batch['answer'][0],
            additional_answers=[],
        )
        curr_seq.update(
            dict(
                most_likely_generation_ids = most_likely_generations,
                generations_ids=generations,
            )
        )
        curr_seq.update(
            dict(
                most_likely_generation=tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True),
                generations=generated_texts,
            )
        )
        curr_seq.update(
            dict(
                perplexity=perplexity
            )
        )
        curr_seq.update(
            dict(
                energy=energy_score
            )
        )
        curr_seq.update(
            dict(
                lexical_similarity=lexical_similarity
            )
        )
        curr_seq.update(
            dict(
                sent_bertscore=sent_bertscore
            )
        )
        curr_seq.update(
            dict(
                entropy=predictive_entropy
            )
        )
        curr_seq.update(
            dict(
                eigenIndicator=eigenIndicator
            )
        )
        curr_seq.update(
            dict(
                eigenIndicatorOutput=eigenIndicatorOutput
            )
        )
        # if args.dataset == 'coqa' or args.dataset == "TruthfulQA":
        #     curr_seq['additional_answers'] = [x[0] for x in batch['additional_answers']]

        sequences.append(curr_seq)
        torch.cuda.empty_cache()
        ########## 信息打印 #########
        logging.info("Prompt: %s", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        logging.info("Question: %s", batch['question'][0])
        logging.info("GTAns: %s", batch['answer'][0])
        logging.info("BestAns: %s", tokenizer.decode(curr_seq['most_likely_generation_ids'], skip_special_tokens=True))
        logging.info("BatchGenerations: %s", generated_texts)
        logging.info("Perplexity: %s", perplexity)
        logging.info("Energy: %s", energy_score)
        logging.info("NormalizedEntropy: %s", predictive_entropy)
        logging.info("LexicalSimilarity: %s", lexical_similarity)
        logging.info("SentBERTScore: %s", sent_bertscore)
        logging.info("EigenScore: %s", eigenIndicator)
        # logging.info("EigenValue: %s", eigenValue)
        logging.info("EigenScore-Output: %s", eigenIndicatorOutput)
        logging.info("Time taken for generation: %s seconds", total_time_instance)
        logging.info("\n")
        total_time += total_time_instance

    logging.info("#### Label Answers ####")
    for itm in sequences:
        itm['acc'] = get_gpt_label(itm['question'], itm['answer'], itm['most_likely_generation'])
        logging.info("GPTLabel: %s", itm['acc'])

    logging.info("#### Compute AUROC ####")
    accs = [itm['acc'] for itm in sequences]
    uncertainties_entropy = [-itm['entropy'] for itm in sequences]
    auroc_entropy = roc_auc_score(accs, uncertainties_entropy)
    logging.info("AUROC of LN Entropy: %s", auroc_entropy)

    # uncertainties_perplexity = [-itm['perplexity'] for itm in sequences]
    # auroc_perplexity = roc_auc_score(accs, uncertainties_perplexity)
    # logging.info("AUROC of Perplexity: %s", auroc_perplexity)

    # uncertainties_energy = [-itm['energy'] for itm in sequences]
    # auroc_energy = roc_auc_score(accs, uncertainties_energy)
    # logging.info("AUROC of Energy: %s", auroc_energy)

    uncertainties_lexical_similarity = [itm['lexical_similarity'] for itm in sequences]
    auroc_lexical_similarity = roc_auc_score(accs, uncertainties_lexical_similarity)
    logging.info("AUROC of Lexical Similarity: %s", auroc_lexical_similarity)

    uncertainties_sent_bertscore = [-itm['sent_bertscore'] for itm in sequences]
    auroc_sent_bertscore = roc_auc_score(accs, uncertainties_sent_bertscore)
    logging.info("AUROC of BERTScore: %s", auroc_sent_bertscore)

    uncertainties_eigen = [-itm['eigenIndicator'] for itm in sequences]
    auroc_eigen = roc_auc_score(accs, uncertainties_eigen)
    logging.info("AUROC of EigenScore: %s", auroc_eigen)

    uncertainties_eigen_output = [-itm['eigenIndicatorOutput'] for itm in sequences]
    auroc_eigen_output = roc_auc_score(accs, uncertainties_eigen_output)
    logging.info("AUROC of EigenScore-Output: %s", auroc_eigen_output)

    if args.enable_reuse:
        total_reuse_ratio = total_reuse_tokens / total_generated_tokens
        logging.info(f"Total generated tokens: {total_generated_tokens}, Total reuse tokens: {total_reuse_tokens}, Total reuse ratio: {total_reuse_ratio}")
    
    logging.info(f"Total time taken for generation: {total_time} seconds")

    return sequences


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens


def get_gpt_label(question, correct_answers, predicted_answer):
    prompt = f'We are assessing the quality of answers to the following question: {question}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"
    response = openai_query(prompt, model='gpt-4o', attemptd_id=0, max_tries=50, verbose=False)
    # print(prompt)

    if response is None:
        logging.warning('Answer is None. Defaulting to no!')
        return 0.0

    if 'yes' in response.lower():
        return 1.0
    elif 'no' in response.lower():
        return 0.0
    else:
        logging.warning('Answer neither no nor yes. Defaulting to no!')
        return 0.0


@dataclass
class CachedOutput(ModelOutput):
    sequences: torch.LongTensor = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None


class cache_sentences:
    def __init__(self, high_temp=1.0, reweight_scaling=1.1):
        self.pool = []
        self.high_temp = high_temp
        self.reweight_scaling = reweight_scaling

    def update_pool(self, outputs, frequent_indices=None, temp_rescaling=False):
        if temp_rescaling:
            for i, score in enumerate(outputs.scores):
                # default set low_temp to 0.1
                outputs.scores[i] = score * 0.1 / self.high_temp
        if frequent_indices is not None:
            self.pool.append({"outputs": outputs, "frequent_indices": frequent_indices})
        else:
            self.pool.append({"outputs": outputs})
    
    def reweight(self, idx):
        frequent_indices = self.pool[idx]["frequent_indices"]
        logits = torch.stack(self.pool[idx]["outputs"].logits)
        logits[frequent_indices] *= self.reweight_scaling
        self.pool[idx]["outputs"].logits = list(logits.unbind(0))


def detect_frequent_words(dict_outputs, threshold=0.9):
    last_prompt_token = dict_outputs.hidden_states[0][-1][0, -1, :]  # shape: [hidden_dim]

    answer_tokens = [h[-1][0] for h in dict_outputs.hidden_states[1:]]  # list of [seq, hidden]
    if not answer_tokens:
        return torch.tensor([], dtype=torch.long)
    answer_tokens = torch.cat(answer_tokens, dim=0)  # shape: [total_seq, hidden]

    similarities = torch.nn.functional.cosine_similarity(
        last_prompt_token.unsqueeze(0),  # [1, hidden]
        answer_tokens,  # [total_seq, hidden]
        dim=-1
    )

    mean_similarity = similarities.mean()
    mask = similarities > (threshold * mean_similarity)
    frequent_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    return frequent_indices


def default_stopping_criteria(generated_tokens, scores, generation_config):
    # generated_tokens: (batch, seq_len)
    eos_token_id = getattr(generation_config, 'eos_token_id', None)
    bad_words_ids = getattr(generation_config, 'bad_words_ids', None)
    last_token = generated_tokens[0, -1].item()
    # Stop if EOS token is generated
    if eos_token_id is not None and last_token in eos_token_id:
        return True
    # Stop if any bad words are generated
    if bad_words_ids is not None:
        for bad_word_seq in bad_words_ids:
            bad_word_seq = torch.tensor(bad_word_seq, device=generated_tokens.device)
            if bad_word_seq.numel() == 0:
                continue
            if (generated_tokens[0, -bad_word_seq.numel():] == bad_word_seq).all():
                return True
    return False


def generate_with_cache(model, inputs, max_new_tokens, output_scores=True, output_logits=True, output_hidden_states=True, 
        temperature=1.0, do_sample=True, generation_config=None, cache=None):
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs.get("attention_mask", None).cuda()
    generated_tokens = input_ids.clone()
    logits = [] if output_logits else None
    scores = [] if output_scores else None
    hidden_states_list = [] if output_hidden_states else None
    past_key_values = None
    num_generated_tokens = 0
    num_reuse_tokens = 0

    top_k_filter = TopKLogitsWarper(top_k=args.top_k)
    top_p_filter = TopPLogitsWarper(top_p=args.top_p)

    if cache is None:
        cache = cache_sentences(reweight_scaling=args.reweight_scaling)

    for i in range(max_new_tokens):
        num_generated_tokens += 1
        is_sampled = False
        if args.reuse_mode is not None:
            for idx, cached_dict in enumerate(cache.pool):
                cached_outputs = cached_dict["outputs"]
                sentence = cached_outputs.sequences[0]
                if generated_tokens[0].numel() < sentence.numel() and torch.equal(generated_tokens[0], sentence[:generated_tokens[0].numel()]):
                    is_sampled = True
                    num_reuse_tokens += 1
                    cut_off_idx = generated_tokens.shape[1]
                    attention_mask = cached_outputs.attentions[:,:cut_off_idx+1]
                    hidden_states_list = cached_outputs.hidden_states[:i+1]
                    # past_key_values = tuple((k[:, :, :cut_off_idx, :], v[:, :, :cut_off_idx, :]) for k, v in cached_outputs.past_key_values)
                    past_key_values = copy.deepcopy(cached_outputs.past_key_values)
                    past_key_values.crop(max_length=cut_off_idx)
                    logits = cached_outputs.logits[:i+1]
                    scores = cached_outputs.scores[:i]
                    if args.reuse_mode == "hard":
                        next_score = logits[-1] / temperature
                        next_score = top_k_filter(None, next_score)
                        next_score = top_p_filter(None, next_score)
                        scores.append(next_score)
                        probs = softmax(next_score, dim=-1)
                        if probs.max().item() < args.reuse_prob:
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = sentence[cut_off_idx].view(1,1)
                    elif args.reuse_mode == "soft":
                        next_score = logits[-1] / temperature
                        next_score = top_k_filter(None, next_score)
                        next_score = top_p_filter(None, next_score)
                        scores.append(next_score)
                        probs = softmax(next_score, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        raise ValueError
                    generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                    break

        if not is_sampled:
            outputs = model(
                input_ids=generated_tokens if past_key_values is None else generated_tokens[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )

            next_logits = outputs.logits[:, -1, :]
            next_score = next_logits / temperature  # (batch, vocab)
            if do_sample:
                next_score = top_k_filter(None, next_score)
                next_score = top_p_filter(None, next_score)
                probs = softmax(next_score, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            if logits is not None:
                logits.append(next_logits)
            if scores is not None:
                scores.append(next_score)
            if hidden_states_list is not None:
                hidden_states_list.append(outputs.hidden_states)
            past_key_values = outputs.past_key_values
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                    dim=1
                )
        if default_stopping_criteria(generated_tokens, scores, generation_config):
            break

    generation_output = CachedOutput(
        sequences=generated_tokens,
        logits=logits,
        scores=scores,
        hidden_states=hidden_states_list,
        attentions=attention_mask,
        past_key_values=past_key_values
    )

    if not is_sampled:
        if args.enable_reweight:
            frequent_indices = detect_frequent_words(generation_output, threshold=args.reweight_threshold)
        else:
            frequent_indices = None
        cache.update_pool(generation_output, frequent_indices, temp_rescaling=False)
    else:
        if args.enable_reweight and i >= args.short_answer_length:
            cache.reweight(idx)

    return generation_output, cache, (num_generated_tokens, num_reuse_tokens)


def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        if '/' in model_name:
            model_name = model_name.replace('/', '_')
        cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}/{args.dataset}/{args.experiment_lot}')
        os.makedirs(cache_dir, exist_ok=True)
        log_filename = f"{cache_dir}/log.txt"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        ], force=True)
        old_results = glob.glob(os.path.join(cache_dir, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        if len(old_results) > 0 and not overwrite:
            print(f'Found {len(old_results)} generations in {cache_dir}.')
            return
        run_id = len(old_results)
        with open(os.path.join(cache_dir, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    logging.info(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    logging.info(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences)
    logging.info(f'Writing {len(sequences)} generations to {cache_dir}...')
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{args.project_ind}.pkl'))
    return

if __name__ == '__main__':
    task_runner = main(parallel=args.nprocess)
