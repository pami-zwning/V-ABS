import requests
import json
import base64
import uuid
import mimetypes
import re
import os
import random
import time
import math
import threading
from pathlib import Path

# ==============================================================================
# VLM Call Tracker (E3, E8: efficiency analysis for rebuttal)
# ==============================================================================
class VLMCallTracker:
    """Thread-safe tracker for VLM API call counts and token usage."""
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.call_count = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.logit_calls = 0

    def record(self, prompt_tokens=0, completion_tokens=0, is_logit=False):
        with self._lock:
            self.call_count += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens
            if is_logit:
                self.logit_calls += 1

    def get_stats(self):
        with self._lock:
            return {
                "call_count": self.call_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "logit_calls": self.logit_calls,
            }

# Global tracker instance
vlm_tracker = VLMCallTracker()

# Import configuration [modified: added INTERNVL_URL_DICT]
from config import (
    GEMINI_URL, GEMINI_HEADERS,
    GPT_PROXY_URL, GPT_KEY, GPT_API_MODEL_NAME,
    QWEN_URL_DICT, INTERNVL_URL_DICT
)

# Try to import openai_proxy (for GPT)
try:
    import openai_proxy
    openai_proxy.generate.default_url = GPT_PROXY_URL
    gpt_client = openai_proxy.GptProxy(api_key=GPT_KEY)
except ImportError:
    gpt_client = None

# Try to import openai (for Qwen/InternVL)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: 'openai' module not found. Qwen/InternVL mode will not work.")

def encode_image(image_path):
    """Helper function: encode an image to Base64"""
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return ""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type if mime_type else "image/jpeg"

def parse_prompt_with_images(prompt_text):
    """Parse text, extract <img src='path'> tags and split into a list of (type, content) tuples."""
    if not isinstance(prompt_text, str):
        return [('text', prompt_text)]

    pattern = r"<img src=['\"](.*?)['\"]>"
    parts = re.split(pattern, prompt_text)
    
    parsed_content = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part: 
                parsed_content.append(('text', part))
        else:
            if part:
                parsed_content.append(('image', part))
                
    return parsed_content

# ==============================================================================
# Helper function: parse OpenAI-format logprobs
# ==============================================================================
def _extract_yes_no_probs(top_logprobs_list):
    """Extract yes/no probabilities from an OpenAI-format top_logprobs list."""
    prob_map = {"yes": 0.0, "no": 0.0}
    if not top_logprobs_list:
        return prob_map

    for candidate in top_logprobs_list:
        if isinstance(candidate, dict):
            token_str = candidate.get('token', '')
            logprob = candidate.get('logprob', -100.0)
        else:
            token_str = candidate.token
            logprob = candidate.logprob
        
        clean_token = token_str.strip().lower()
        if clean_token in prob_map:
            prob_map[clean_token] += math.exp(logprob)
            
    return prob_map

# ==============================================================================
# Unified entry point [modified: added InternVL support]
# ==============================================================================
def chat_vlm(prompt, model_name, messages=None, logit=False, **kwargs):
    """
    Unified entry point.
    Supports Qwen, InternVL, GPT-4o, Gemini
    """
    # Route both Qwen and InternVL to the generic vLLM handler
    if "qwen" in model_name.lower() or "intern" in model_name.lower():
        return _chat_vllm_generic(prompt, model_name, messages, logit=logit, **kwargs)
    elif "gpt" in model_name.lower():
        return _chat_gpt4o(prompt, messages, logit=logit, **kwargs)
    else:
        # Default to Gemini
        return _chat_gemini(prompt, messages, logit=logit, **kwargs)

# ==============================================================================
# vLLM generic implementation (Qwen & InternVL) [modified: refactored into generic function]
# ==============================================================================
def _chat_vllm_generic(prompt, model_name, messages=None, logit=False, **kwargs):
    if OpenAI is None:
        raise ImportError("Please install 'openai' package to use vLLM models.")
    
    if messages is None:
        messages = []

    # 1. Determine which URL dictionary to use
    target_urls = []
    if "qwen" in model_name.lower():
        target_urls = QWEN_URL_DICT.get(model_name, [])
        if not target_urls and QWEN_URL_DICT: # If key not found but dict exists, fall back to first entry
             print(f"[Warn] Model {model_name} not found in QWEN_URL_DICT, using fallback.")
             target_urls = list(QWEN_URL_DICT.values())[0]
    elif "intern" in model_name.lower():
        target_urls = INTERNVL_URL_DICT.get(model_name, [])
        if not target_urls and INTERNVL_URL_DICT:
             print(f"[Warn] Model {model_name} not found in INTERNVL_URL_DICT, using fallback.")
             target_urls = list(INTERNVL_URL_DICT.values())[0]
             
    if not target_urls:
        raise ValueError(f"No URLs configured for model: {model_name}")
    
    # 2. Construct messages (vLLM OpenAI-compatible format)
    vllm_messages = []
    input_flow = messages + [{"role": "user", "content": prompt}]
    
    for msg in input_flow:
        role = msg['role']
        if role == 'model': role = 'assistant'
        content_list = []
        raw_content = msg['content']
        parsed_items = []
        if isinstance(raw_content, str):
            parsed_parts = parse_prompt_with_images(raw_content)
            for p_type, p_val in parsed_parts:
                parsed_items.append((p_type, p_val))
        elif isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, str):
                    parsed_items.append(('text', item))
                elif isinstance(item, dict):
                    if 'image' in item: parsed_items.append(('image', item['image']))
                    if 'text' in item: parsed_items.append(('text', item['text']))

        for p_type, p_val in parsed_items:
            if p_type == 'text':
                content_list.append({"type": "text", "text": p_val})
            elif p_type == 'image':
                mime = get_mime_type(p_val)
                b64 = encode_image(p_val)
                if b64:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })
        vllm_messages.append({"role": role, "content": content_list})

    # 3. Send request
    MAX_RETRIES = 5
    RETRY_DELAY = 2
    res_content = f"Error: Failed after {MAX_RETRIES} retries."
    prob_map = {"yes": 0.0, "no": 0.0}

    for attempt in range(MAX_RETRIES):
        current_base_url = random.choice(target_urls)
        try:
            client = OpenAI(api_key="EMPTY", base_url=current_base_url)
            
            # Auto-detect the currently deployed model name
            try:
                model_list = client.models.list()
                api_model_name = model_list.data[0].id
            except Exception:
                # If detection fails, use the provided name as default
                api_model_name = model_name

            req_kwargs = {
                "model": api_model_name,
                "messages": vllm_messages,
                "temperature": kwargs.get('temperature', 0.7), # Lower temperature recommended for InternVL/Qwen
                "max_tokens": kwargs.get('max_tokens', 8 * 1024),
                "timeout": 180
            }
            if logit:
                req_kwargs["logprobs"] = True
                req_kwargs["top_logprobs"] = 20

            response = client.chat.completions.create(**req_kwargs)
            
            if response.choices:
                res_content = response.choices[0].message.content

                # Track usage
                _pt = getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0
                _ct = getattr(response.usage, 'completion_tokens', 0) if response.usage else 0
                vlm_tracker.record(prompt_tokens=_pt, completion_tokens=_ct, is_logit=logit)

                # Extract logits
                if logit and response.choices[0].logprobs:
                    first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                    prob_map = _extract_yes_no_probs(first_token_logprobs)
            else:
                res_content = "Error: Empty response from vLLM"

            break

        except Exception as e:
            print(f"[Warning] vLLM Request Failed (Attempt {attempt+1}) on {current_base_url}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                res_content = f"Error: vLLM API Request Failed - {str(e)}"

    # 4. Update history and return
    new_messages = messages.copy()
    if not (new_messages and new_messages[-1]['role'] == 'user' and new_messages[-1]['content'] == prompt):
        new_messages.append({"role": "user", "content": prompt})
    new_messages.append({"role": "assistant", "content": res_content})
    
    if logit:
        return res_content, new_messages, prob_map
    else:
        return res_content, new_messages

# ==============================================================================
# GPT-4o implementation (unchanged)
# ==============================================================================
def _chat_gpt4o(prompt, messages=None, logit=False, **kwargs):
    if not gpt_client:
        raise RuntimeError("OpenAI Proxy client not initialized.")

    if messages is None: messages = []
    
    gpt_messages = []
    for msg in messages:
        role = msg['role']
        if role == 'model': role = 'assistant'
        gpt_content = []
        raw_content = msg['content']
        
        if isinstance(raw_content, str):
             parsed_parts = parse_prompt_with_images(raw_content)
             for p_type, p_val in parsed_parts:
                 if p_type == 'text': gpt_content.append({"type": "text", "text": p_val})
                 elif p_type == 'image':
                     mime = get_mime_type(p_val)
                     b64 = encode_image(p_val)
                     if b64: gpt_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        elif isinstance(raw_content, list):
             for item in raw_content:
                 if isinstance(item, str): gpt_content.append({"type": "text", "text": item})
                 elif isinstance(item, dict) and 'image' in item:
                     mime = get_mime_type(item['image'])
                     b64 = encode_image(item['image'])
                     if b64: gpt_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        gpt_messages.append({"role": role, "content": gpt_content})

    current_content = []
    if isinstance(prompt, str):
        parsed_parts = parse_prompt_with_images(prompt)
        for p_type, p_val in parsed_parts:
            if p_type == 'text': current_content.append({"type": "text", "text": p_val})
            elif p_type == 'image':
                mime = get_mime_type(p_val)
                b64 = encode_image(p_val)
                if b64: current_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    elif isinstance(prompt, list):
        for item in prompt:
             if isinstance(item, str): current_content.append({"type": "text", "text": item})
             elif isinstance(item, dict):
                 if 'text' in item: current_content.append({"type": "text", "text": item['text']})
                 if 'image' in item:
                     mime = get_mime_type(item['image'])
                     b64 = encode_image(item['image'])
                     if b64: current_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    gpt_messages.append({"role": "user", "content": current_content})

    tid = f"visualbeamsearch_gpt_{uuid.uuid4().hex[:8]}"
    res_content = "Error: GPT Request Failed"
    prob_map = {"yes": 0.0, "no": 0.0}

    gen_kwargs = {
        "messages": gpt_messages,
        "model": GPT_API_MODEL_NAME,
        "transaction_id": tid,
        "extra_body": {"thinking": {"type": "enabled"}},
        # "temperature": kwargs.get('temperature', 1.0)
    }
    if logit:
        gen_kwargs["logprobs"] = True
        gen_kwargs["top_logprobs"] = 20

    for attempt in range(3):
        try:
            rsp = gpt_client.generate(**gen_kwargs)
            if rsp.ok:
                resp_data = rsp.json()
                choices = None

                # Track GPT call
                vlm_tracker.record(is_logit=logit)

                # Try multiple possible response formats
                if 'choices' in resp_data:
                    choices = resp_data['choices']
                elif 'data' in resp_data:
                    data_obj = resp_data['data']
                    if isinstance(data_obj, dict):
                        if 'choices' in data_obj:
                            choices = data_obj['choices']
                        elif 'response_content' in data_obj:
                            if isinstance(data_obj['response_content'], dict) and 'choices' in data_obj['response_content']:
                                choices = data_obj['response_content']['choices']
                        # Added: extract content directly from data
                        elif 'content' in data_obj:
                            res_content = data_obj['content']
                            break
                        elif 'message' in data_obj:
                            if isinstance(data_obj['message'], dict) and 'content' in data_obj['message']:
                                res_content = data_obj['message']['content']
                                break
                            elif isinstance(data_obj['message'], str):
                                res_content = data_obj['message']
                                break
                        # Added: handle response field
                        elif 'response' in data_obj:
                            if isinstance(data_obj['response'], str):
                                res_content = data_obj['response']
                                break
                            elif isinstance(data_obj['response'], dict) and 'content' in data_obj['response']:
                                res_content = data_obj['response']['content']
                                break
                # Added: top-level content field
                elif 'content' in resp_data:
                    res_content = resp_data['content']
                    break
                elif 'response' in resp_data:
                    res_content = resp_data['response']
                    break

                if choices:
                    # Handle different possible formats in choices
                    first_choice = choices[0]
                    if 'message' in first_choice and 'content' in first_choice['message']:
                        res_content = first_choice['message']['content']
                    elif 'content' in first_choice:
                        res_content = first_choice['content']
                    elif 'text' in first_choice:
                        res_content = first_choice['text']
                    else:
                        print(f"[GPT Debug] Unknown choice structure: {first_choice.keys() if isinstance(first_choice, dict) else type(first_choice)}")
                        raise KeyError("Response choice structure unknown")

                    if logit and 'logprobs' in first_choice and first_choice['logprobs']:
                        logprobs_obj = first_choice['logprobs']
                        if 'content' in logprobs_obj and logprobs_obj['content']:
                            first_token_tops = logprobs_obj['content'][0]['top_logprobs']
                            prob_map = _extract_yes_no_probs(first_token_tops)
                elif res_content == "Error: GPT Request Failed":
                    # Still unable to extract content, print detailed debug info
                    print(f"[GPT Debug] Response structure unknown. Keys: {resp_data.keys()}")
                    if 'data' in resp_data:
                        print(f"[GPT Debug] data keys: {resp_data['data'].keys() if isinstance(resp_data['data'], dict) else type(resp_data['data'])}")
                    raise KeyError(f"Response structure unknown: {list(resp_data.keys())}")
                break
            else:
                print(f"GPT API Error (HTTP {rsp.status_code}): {rsp.text[:500]}")
                time.sleep(1)
        except Exception as e:
            print(f"GPT Request Exception (Attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff

    new_messages = messages.copy()
    if not (new_messages and new_messages[-1]['role'] == 'user' and new_messages[-1]['content'] == prompt):
        new_messages.append({"role": "user", "content": prompt})
    new_messages.append({"role": "assistant", "content": res_content})
    
    if logit:
        return res_content, new_messages, prob_map
    else:
        return res_content, new_messages

# ==============================================================================
# Gemini implementation (unchanged)
# ==============================================================================
def _chat_gemini(prompt, messages=None, logit=False, **kwargs):
    if messages is None: messages = []
    temperature = kwargs.get('temperature', 0.0)
    
    gemini_contents = []
    for msg in messages:
        role = "user" if msg['role'] == "user" else "model"
        parts = []
        content = msg['content']
        if isinstance(content, str):
             parsed_parts = parse_prompt_with_images(content)
             for p_type, p_val in parsed_parts:
                 if p_type == 'text': parts.append({"text": p_val})
                 elif p_type == 'image':
                     b64 = encode_image(p_val)
                     mime = get_mime_type(p_val)
                     if b64: parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str): parts.append({"text": item})
        if parts: gemini_contents.append({"role": role, "parts": parts})
    
    current_parts = []
    if isinstance(prompt, str):
        parsed_parts = parse_prompt_with_images(prompt)
        for p_type, p_val in parsed_parts:
            if p_type == 'text': current_parts.append({"text": p_val})
            elif p_type == 'image':
                b64 = encode_image(p_val)
                mime = get_mime_type(p_val)
                if b64: current_parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    elif isinstance(prompt, list):
        for item in prompt:
            if isinstance(item, dict) and 'image' in item:
                b64_img = encode_image(item['image'])
                mime = get_mime_type(item['image'])
                if b64_img: current_parts.append({"inline_data": {"mime_type": mime, "data": b64_img}})
            elif isinstance(item, dict) and 'text' in item: current_parts.append({"text": item['text']})
            elif isinstance(item, str): current_parts.append({"text": item})
    if current_parts: gemini_contents.append({"role": "user", "parts": current_parts})

    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": 16 * 1024
    }
    if logit:
        generation_config["responseLogprobs"] = True
        generation_config["logprobs"] = 5

    data = {
        "contents": gemini_contents,
        "generationConfig": generation_config
    }

    res_content = "Error: Gemini Request Failed"
    prob_map = {"yes": 0.0, "no": 0.0}

    for attempt in range(3):
        try:
            response = requests.post(GEMINI_URL, headers=GEMINI_HEADERS, json=data, timeout=180)
            response.raise_for_status()
            response_json = response.json()
            
            try:
                candidate = response_json['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    texts = [p['text'] for p in candidate['content']['parts'] if 'text' in p]
                    res_content = "".join(texts)
                else:
                     res_content = "Error: Empty or truncated."

                # Track Gemini call
                vlm_tracker.record(is_logit=logit)

                if logit and 'logprobsResult' in candidate:
                    logprobs_result = candidate['logprobsResult']
                    if 'topCandidates' in logprobs_result and len(logprobs_result['topCandidates']) > 0:
                        first_token_candidates = logprobs_result['topCandidates'][0]['candidates']
                        for cand in first_token_candidates:
                            token_str = cand.get('token', '')
                            log_p = cand.get('logProbability', -100.0)
                            clean_token = token_str.strip().lower()
                            if clean_token in prob_map:
                                prob_map[clean_token] += math.exp(log_p)
                break 
            except (KeyError, IndexError) as e:
                res_content = f"Error: Could not parse response. {str(e)}"
                break 
        except Exception as e:
            print(f"Gemini Request Exception (Attempt {attempt+1}): {e}")
            time.sleep(2)

    new_messages = messages.copy()
    if not (new_messages and new_messages[-1]['role'] == 'user' and new_messages[-1]['content'] == prompt):
        new_messages.append({"role": "user", "content": prompt})
    new_messages.append({"role": "assistant", "content": res_content})

    if logit:
        return res_content, new_messages, prob_map
    else:
        return res_content, new_messages