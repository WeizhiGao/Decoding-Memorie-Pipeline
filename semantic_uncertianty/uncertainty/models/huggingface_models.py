"""Implement HuggingfaceModel models."""
import copy
import logging
import os
from collections import Counter
import gc

import accelerate
import torch
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download
from transformers.generation.utils import TopKLogitsWarper, TopPLogitsWarper

from transformers.generation.utils import ModelOutput, dataclass
from typing import Optional, Tuple
from torch.nn.functional import softmax
from transformers.cache_utils import Cache


from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False
    
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

def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer!
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """HuggingfaceModel."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None,
                 reuse_mode="soft", reuse_prob=0.85, high_temp=1.0, 
                 enable_reweight=False, short_answer_length=10, threshold=0.9, reweight_scaling=1.1):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        print(model_name)
        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            if 'Llama-2' in model_name or 'Llama-3' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf' if 'Llama-2' in model_name else model_name
            else:
                base = 'huggyllama'

            # self.tokenizer = AutoTokenizer.from_pretrained(
            #     f"{base}/{model_name}", device_map="auto",
            #     token_type_ids=None)

            # self.tokenizer = AutoTokenizer.from_pretrained(
            #     "/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-7b-chat", device_map="auto",
            #     token_type_ids=None)

            llama65b = '65b' in model_name.lower() and base == 'huggyllama'
            llama2or3_70b = '70b' in model_name.lower() and base == 'meta-llama'

            if ('7b' in model_name or '13b' in model_name) or eightbit:
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     f"{base}/{model_name}", device_map="auto",
                #     max_memory={0: '80GIB'}, **kwargs,)
                if '7b' in model_name:
                    self.tokenizer = AutoTokenizer.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-7b-chat")
                    self.model = AutoModelForCausalLM.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-7b-chat",
                         device_map="auto", max_memory={0: '80GIB'}, **kwargs)
                if '13b' in model_name:
                    self.tokenizer = AutoTokenizer.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-13b-chat")
                    self.model = AutoModelForCausalLM.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/llama2-13b-chat",
                         device_map="auto", max_memory={0: '80GIB'}, **kwargs)

            elif llama2or3_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()
                if 'chat' in model_name:
                    max_mem = 17.5 * 4686198491
                else:
                    max_mem = 15 * 4686198491
                
                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16', skip_keys='past_key_values')

            else:
                raise ValueError

        elif 'mistral' in model_name.lower():

            # if model_name.endswith('-8bit'):
            #     kwargs = {'quantization_config': BitsAndBytesConfig(
            #         load_in_8bit=True,)}
            #     model_name = model_name[:-len('-8bit')]
            # if model_name.endswith('-4bit'):
            #     kwargs = {'quantization_config': BitsAndBytesConfig(
            #         load_in_4bit=True,)}
            #     model_name = model_name[:-len('-8bit')]
            # else:
            #     kwargs = {}

            # model_id = f'mistralai/{model_name}'
            # self.tokenizer = AutoTokenizer.from_pretrained(
            #     model_id, device_map='auto', token_type_ids=None,
            #     clean_up_tokenization_spaces=False)

            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_id,
            #     device_map='auto',
            #     max_memory={0: '80GIB'},
            #     **kwargs,
            # )
            self.tokenizer = AutoTokenizer.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/mistral-7b-instruct")
            self.model = AutoModelForCausalLM.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/mistral-7b-instruct",
                 device_map="auto", max_memory={0: '80GIB'})

        elif 'falcon' in model_name:
            # model_id = f'tiiuae/{model_name}'
            # self.tokenizer = AutoTokenizer.from_pretrained(
            #     model_id, device_map='auto', token_type_ids=None,
            #     clean_up_tokenization_spaces=False)

            # kwargs = {'quantization_config': BitsAndBytesConfig(
            #     load_in_8bit=True,)}

            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_id,
            #     trust_remote_code=True,
            #     device_map='auto',
            #     **kwargs,
            # )
            self.tokenizer = AutoTokenizer.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/falcon-7b-instruct")
            self.model = AutoModelForCausalLM.from_pretrained("/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models/falcon-7b-instruct",
                 device_map="auto", max_memory={0: '80GIB'})
        elif 'phi' in model_name.lower():
            model_id = f'microsoft/{model_name}'  # e.g. Phi-3-mini-128k-instruct
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
            )
        elif 'gemma' in model_name:
            model_id = f'google/{model_name}'  # e.g. gemma-7b-it
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = 4096 if 'Llama-2' in model_name else 2048
        self.reuse_mode = reuse_mode
        self.reuse_prob = reuse_prob
        self.short_answer_length = short_answer_length
        self.high_temp = high_temp
        self.enable_reweight = enable_reweight
        self.threshold = threshold
        self.reweight_scaling = reweight_scaling

    def predict(self, input_data, temperature, return_full=False, return_latent=False):

        if isinstance(input_data, tuple):
            logging.WARNING("INPUT IS A TUPLE.")
            input_data = input_data[0]

        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # HF models seems has changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # if access idx is larger/equal
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        if return_latent:
            # Stack second last token embeddings from all layers 
            if len(hidden) == 1:  # FIX: runtime error for mistral-7b on bioasq
                sec_last_input = hidden[0]
            elif ((n_generated - 2) >= len(hidden)):
                sec_last_input = hidden[-2]
            else:
                sec_last_input = hidden[n_generated - 2]
            sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
    
            # Get the last input token embeddings (before generated tokens)
            last_tok_bef_gen_input = hidden[0]
            last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        hidden_states = (last_token_embedding,)

        if return_latent:
            hidden_states += (sec_last_token_embedding, last_tok_bef_gen_embedding)
        else:
            hidden_states += (None, None)

        return_values = (sliced_answer, log_likelihoods, hidden_states)

        return return_values
    
    def predict_with_cache(self, input_data, temperature, topk=50, topp=0.9, return_full=False, return_latent=False, cache=None):

        if isinstance(input_data, tuple):
            logging.WARNING("INPUT IS A TUPLE.")
            input_data = input_data[0]

        inputs = self.tokenizer(input_data, return_tensors="pt", return_offsets_mapping=True).to("cuda")
        offsets_len = inputs["offset_mapping"].size(1)

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            if 'token_type_ids' in inputs:  # HF models seems has changed.
                del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = None

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)

        outputs, cache, (num_generated_tokens, num_reuse_tokens) = self.generate_with_cache(
            inputs=inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            do_sample=True,
            output_scores=True,
            output_hidden_states=True,
            stopping_criteria=stopping_criteria,
            cache=cache,
            topk=topk,
            topp=topp
            )

        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        if return_full:
            return full_answer

        # For some models, we need to remove the input_data from the answer.
        if full_answer.startswith(input_data):
            input_data_offset = len(input_data)
        else:
            raise ValueError('Have not tested this in a while.')

        # Remove input from answer.
        answer = full_answer[input_data_offset:]

        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()
        token_stop_index = self.tokenizer(full_answer[:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_input_token = len(inputs['input_ids'][0])
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # if access idx is larger/equal
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'last_token: %s, generation was: %s, slice_answer: %s',
                n_generated, n_input_token, token_stop_index,
                self.tokenizer.decode(outputs['sequences'][0][-1]),
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            last_input = hidden[n_generated - 1]

        # Then access last layer for input
        last_layer = last_input[-1]
        # Then access last token in input.
        last_token_embedding = last_layer[:, -1, :].cpu()

        if return_latent:
            # Stack second last token embeddings from all layers 
            if len(hidden) == 1:  # FIX: runtime error for mistral-7b on bioasq
                sec_last_input = hidden[0]
            elif ((n_generated - 2) >= len(hidden)):
                sec_last_input = hidden[-2]
            else:
                sec_last_input = hidden[n_generated - 2]
            sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
    
            # Get the last input token embeddings (before generated tokens)
            last_tok_bef_gen_input = hidden[0]
            last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = [score.item() for score in transition_scores[0]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]

        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')

        if len(log_likelihoods) == 0:
            raise ValueError

        hidden_states = (last_token_embedding,)

        if return_latent:
            hidden_states += (sec_last_token_embedding, last_tok_bef_gen_embedding)
        else:
            hidden_states += (None, None)

        return_values = (sliced_answer, log_likelihoods, hidden_states, cache, (num_generated_tokens, num_reuse_tokens))

        return return_values
    
    @torch.no_grad()
    def generate_with_cache(
        self,
        inputs=None,
        max_new_tokens=20,
        output_scores=True,
        output_hidden_states=True,
        temperature=1.0,
        do_sample=True,
        stopping_criteria=None,
        cache:cache_sentences = None,
        topk=50,
        topp=0.9
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        generated_tokens = input_ids.clone()
        logits = [] if output_scores else None
        scores = [] if output_scores else None
        hidden_states_list = [] if output_hidden_states else None
        past_key_values = None

        num_generated_tokens = 0
        num_reuse_tokens = 0

        if cache is None:
            cache = cache_sentences(high_temp=self.high_temp, reweight_scaling=self.reweight_scaling)

        top_k_filter = TopKLogitsWarper(top_k=topk)
        top_p_filter = TopPLogitsWarper(top_p=topp)

        for i in range(max_new_tokens):
            num_generated_tokens += 1
            is_sampled = False

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
                    if self.reuse_mode == "hard":
                        next_score = logits[-1] / temperature
                        next_score = top_k_filter(None, next_score)
                        next_score = top_p_filter(None, next_score)
                        scores.append(next_score)
                        probs = softmax(next_score, dim=-1)
                        if probs.max().item() < self.reuse_prob:
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = sentence[cut_off_idx].view(1,1)
                    elif self.reuse_mode == "soft":
                        next_score = logits[-1] / temperature
                        next_score = top_k_filter(None, next_score)
                        next_score = top_p_filter(None, next_score)
                        probs = softmax(next_score, dim=-1)
                        scores.append(next_score)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        raise ValueError
                    generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                    break

            if not is_sampled:
                outputs = self.model(
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
            
            # Check stopping criteria
            if stopping_criteria(generated_tokens, scores):
                break

        # Construct the final output to mimic model.generate()
        generation_output = CachedOutput(
            sequences=generated_tokens,
            logits=logits,
            scores=scores,
            hidden_states=hidden_states_list,
            attentions=attention_mask,
            past_key_values=past_key_values
        )

        if not is_sampled:
            if self.enable_reweight:
                frequent_indices = detect_frequent_words(generation_output, self.threshold)
            else:
                frequent_indices = None
            cache.update_pool(generation_output, frequent_indices, temp_rescaling=False)
        else:
            if self.enable_reweight and i >= self.short_answer_length: 
                cache.reweight(idx)

        return generation_output, cache, (num_generated_tokens, num_reuse_tokens)

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

    def get_perplexity(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        tokenized_data = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        with torch.no_grad():
            model_output_true = self.model(tokenized_data, labels=tokenized_data)

        perplexity = - model_output_true.loss.item()


        return perplexity
    

def make_track_prompt(question, answer):
        prefix = "Extract from the following long answer the short answer, only the relevant tokens. " \
        "If the long answer does not answer the question, output NO ANSWER."

        track_prompt = prefix + "\n" + "Qusetion: " + question + "\n" + "Long Answer: " + answer + "\n" + "Exact answer: "
    
        return track_prompt


def local_measures(embeddings, attentions=None, mode='l2'):
    flat_embeddings = embeddings.squeeze()
    if mode == 'l2':
        correlation = (flat_embeddings@flat_embeddings.transpose(0,1)).abs()
        correlation.fill_diagonal_(0) # set diagonal as 0 for better visualization
    elif mode == 'cosine':
        norm = torch.norm(flat_embeddings, p=2, dim=1, keepdim=True)
        normalized_flat_embeddings = flat_embeddings / norm
        correlation = (normalized_flat_embeddings@normalized_flat_embeddings.transpose(0,1)).abs()
        correlation.fill_diagonal_(0) # set diagonal as 0 for better visualization
    elif mode == 'attn_score':
        assert attentions is not None
        correlation = attentions.max(dim=1)[0].squeeze()
        # correlation[:, 0] = 0
    else:
        NotImplementedError

    return correlation


def global_measures(embeddings, split_index, mode='l2'):
    flat_embeddings = embeddings.squeeze()
    context = flat_embeddings[:split_index, :].mean(dim=0, keepdim=True)
    response = flat_embeddings[split_index:, :]
    if mode == 'l2':
        correlation = (context@response.transpose(0,1).squeeze()).abs()
    elif mode == 'cosine':
        norm1 = torch.norm(context, p=2, dim=1, keepdim=True)
        normalized_context = context / norm1
        norm2 = torch.norm(response, p=2, dim=1, keepdim=True)
        normalized_response = response / norm2
        correlation = (normalized_context@normalized_response.transpose(0,1).squeeze()).abs()
    else:
        NotImplementedError

    return correlation


def locate_exact_answer(hidden_states_list, start_idx):
    embeddings0 = []
    embeddings1 = []

    for token_hidden in hidden_states_list:
        embeddings0.append(token_hidden[0])
        embeddings1.append(token_hidden[-1])

    embeddings0 = torch.cat(embeddings0, dim=1) 
    embeddings1 = torch.cat(embeddings1, dim=1) 

    corr = local_measures(embeddings0, attentions=None, mode='cosine')
    possible_idx = (torch.tril(corr, diagonal=-1)<0.9).all(dim=1).squeeze().nonzero(as_tuple=False)
    possible_idx = possible_idx[possible_idx>=start_idx]
    # print('QA Split Index: ', start_idx)
    # print('Possible Index: ', possible_idx.cpu().numpy())

    corr = local_measures(embeddings1, attentions=None, mode='cosine')
    similarites1 = corr[possible_idx, 1:(start_idx-1)].mean(dim=1)
    # similarites2 = corr[possible_idx, start_idx:].mean(dim=1)

    most_possible_idx = possible_idx[similarites1.argmin()]
    
    # print("All Response Tokens: ", corr[start_idx:-1, 1:(start_idx-1)].mean(dim=1).cpu().numpy())
    # print("Most possible idx: ", possible_idx[similarites1.argmin()].item())

    return most_possible_idx
