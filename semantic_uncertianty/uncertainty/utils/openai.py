import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type

# from openai import OpenAI
from openai import AzureOpenAI


endpoint = os.getenv("ENDPOINT_URL", None)
# deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", None)

# CLIENT = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', False))
CLIENT = AzureOpenAI(azure_endpoint=endpoint, api_key=subscription_key, api_version="2025-01-01-preview",)


class KeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass


@retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=1.0, model='gpt-4'):
    """Predict with GPT models."""

    if not CLIENT.api_key:
        raise KeyError('Need to provide OpenAI API key in environment variable `OPENAI_API_KEY`.')

    if isinstance(prompt, str):
        messages = [
            {'role': 'user', 'content': prompt},
        ]
    else:
        messages = prompt

    # if model == 'gpt-4':
    #     model = 'gpt-4-0613'
    # elif model == 'gpt-4-turbo':
    #     model = 'gpt-4-1106-preview'
    # elif model == 'gpt-3.5':
    #     model = 'gpt-3.5-turbo-1106'
    if model == 'gpt-4':
        model = 'gpt-4o'
    else:
        model = 'gpt-35-turbo'

    try:
        output = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    except Exception as e:
        # print(e)
        return None

    response = output.choices[0].message.content
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
