import os
import json
import time
import datetime
import random
import sys
from typing import List
from collections.abc import Iterable

import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm
from uuid import uuid4
from openai import OpenAI


BASE_URL = os.environ.get('BASE_URL', 'http://localhost:8000/v1')
MODEL_NAME = os.environ.get('MODEL_NAME', 'qwen25_72B_instruct')


# ---------- model processing ---------- #
def qwen_process(prompt,
                 temperature=0,
                 model_name=MODEL_NAME,
                #  base_url="http://10.128.178.2:8000/v1",
                 base_url=BASE_URL,
                 system_prompt=None,
                 history=[],
                 max_retry = 5
    ):
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt}
        ]
    else:
        messages = []

    history_ = [{"role": "user" if i %2 ==0 else 'assistant', "content": h} for i,h in enumerate(history)]
    messages.extend(history_)
    messages.append({"role": "user", "content": prompt})

    while True:
        try:
            client = OpenAI(
                api_key=os.environ.get('OPENAI_API_KEY', 'None'), base_url=base_url # mine
            )

            response =  client.chat.completions.create(temperature=temperature,
                model=model_name,
                messages=messages,
            )
            text = response.choices[0].message.content
            history += [prompt, text]
            if text == "":
                raise ValueError("Empty response for prompt: " + prompt)
            return text, history
        except Exception as e:
            max_retry -= 1
            log_print(f'Error: {e}, retry left {max_retry}')
            if max_retry <= 0:
                raise e


# ---------- misc ---------- #
def mk_pbar(iterable, ncols=80, **kwargs):
    # check if iterable
    if not hasattr(iterable, '__iter__'):
        raise ValueError("Input is not iterable.")
    return tqdm(iterable, ncols=ncols, **kwargs)


def mk_len_pbar(ncols=80, **kwargs):
    return tqdm(ncols=ncols, **kwargs)


def generate_uuid():
    return str(uuid4())


def load_jsonl(path):
    data = []
    with open(path, "rt") as f:
        for line in mk_pbar(f):
            data.append(json.loads(line.strip()))
    return data

def load_json(path):
    with open(path, "rt") as f:
        return json.load(f)


def save_jsonl(data, path, mode='w', use_tqdm=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        if use_tqdm:
            data = mk_pbar(data)
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wt") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def tag_join(valid_tags, sep=','):
    return sep.join([str(tag) for tag in valid_tags])


def mean_list(data_list):
    data_list = [float(d) for d in data_list if d is not None]
    return sum(data_list) / len(data_list)


def log_print(*content, **kwargs):
    # set datetime timezone to Shanghai.
    os.environ['TZ'] = 'Asia/Shanghai'
    time.tzset()
    content = [f'[{datetime.datetime.now()}]'] + list(content)
    print(*content, **kwargs)
    sys.stdout.flush()


def json_print(data):
    print(json.dumps(data, ensure_ascii=False, indent=4))


def multithreading(func, thread_num=8, data=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        executor.map(func, data)

def math_func(data):
    instruction = data['instruction']
    response, _ = qwen_process(instruction)
    data['response'] = response
    return data

if __name__ == '__main__':
    print(qwen_process('"把下面各数写成用“亿”作单位的数．\n$500000000=$\n$96000000000=$\n$9958200000\\approx$\n$7421305678\\approx$\n\n"'))
    if os.environ.get('UTILS_sWORKER_MODE', 0):
        while True:
            data = load_jsonl('.temp/dataset/openai_gsm8k_7000_samples.jsonl')
            multithreading(math_func, data=data)
