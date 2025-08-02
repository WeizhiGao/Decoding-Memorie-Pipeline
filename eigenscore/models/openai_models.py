import json
import logging
import os
import time

# import openai
from openai import AzureOpenAI
import persist_to_disk as ptd
from openai import APIError, RateLimitError

TOTAL_TOKEN = 0
endpoint = os.getenv("ENDPOINT_URL", None)
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", None)


client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# with open(os.path.join(os.path.dirname(__file__), '..', 'keys.json'), 'r') as f:
#     openai.api_key = json.load(f)['openai']['apiKey']


def _openai_query_cached_new(prompt='Hello World', model='ada', attempt_id=0):
    return client.chat.completions.create(model=model,
                                        messages=[{"role": "user", "content": prompt}])

def retry_openai_query(prompt='Hello World', model='ada', attempt_id=0, max_tries=5,):
    for i in range(max_tries):
        try:
            try:
                return _openai_query_cached_new(prompt, model, attempt_id)
            except Exception:
                # logging.warning(f"Sensitive Question. Returning None.")
                return None
        except (RateLimitError, APIError) as e:
            print(e)
            time.sleep(1)
            if i == max_tries - 1:
                raise e

def _token_to_price(model, tokens):
    return tokens // 1000 * {'gpt-3.5-turbo': 0.002}[model]

def openai_query(prompt, model, attemptd_id, max_tries=50, verbose=False):
    # global TOTAL_TOKEN
    completion = retry_openai_query(prompt, model, attemptd_id, max_tries=max_tries)
    if completion is None:
        return None
    else:
        txt_ans = completion.choices[0].message.content
    # prev_milestone = _token_to_price(model, TOTAL_TOKEN) // 0.1
    # TOTAL_TOKEN += completion['usage']['total_tokens']

    # if (_token_to_price(model, TOTAL_TOKEN) // 0.1)  > prev_milestone:
    #     if verbose:
    #         print(f"Total Cost > $ {(_token_to_price(model, TOTAL_TOKEN) // 0.1) * 0.1:.1f}")
        return txt_ans
