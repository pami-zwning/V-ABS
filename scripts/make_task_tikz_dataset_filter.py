import base64
import requests
import sys
import os
from openai import OpenAI
import json
import shutil

from utils_chat import chat_gpt4o
from utils_prompt import TIKZ_CONVERT_EXAMPLES
from utils_execution import CodeExecutor
from utils_parse import Parser
from demo_jpeg_convert import convert_jpeg_to_png

sys.path.append('./workspace')
from utils import *

PROMPT = """
## TARGET GEOMETRY ##
<img src='{target_image_path}'>

## GENERATED GEOMETRY ##
<img src='{output_image_path}'>

- Pay attention to the shape of the geometry, the ## GENERATED GEOMETRY ## should be equivalent with the ## TARGET GEOMETRY ##.
- Beside the shape, please check the length annotations, the positions of annotations should be strictly equivalent with the ## TARGET GEOMETRY ##.
- Some difference in overall layout is acceptable. Be strict when evaluating the shape and length annotations of the ## GENERATED GEOMETRY ##, any difference of them will be considered as a failure.

Evaluate the ## GENERATED GEOMETRY ## based on the above criteria, give your reply that ends with TERMINATE.
else, reply that ends with GENERATE and give a score out of 10 in Json format:
```json 
{{"score" : <score>}}
```
"""

def process_task(original_task, output_task):
    assert os.path.exists(original_task + '/ex.json')

    ex_json = load_json(original_task + '/ex.json')
    image_path = ex_json['image_path_code']
    image_path_filename = image_path.split('/')[-1]
    image_path_original = image_path_filename.replace('.png', '_original.png')
    original_image_path = os.path.join(original_task, image_path_original)

    assert os.path.exists(image_path), f"Generated image path {image_path} does not exist!"
    assert os.path.exists(original_image_path), f"Original image path {original_image_path} does not exist!"

    prompt = PROMPT.format(
        target_image_path=original_image_path,
        output_image_path=image_path,
    )

    _SAVE_ORNOT = False
    response, history = chat_gpt4o(prompt)
    if 'TERMINATE' in response:
        _SAVE_ORNOT = True
    else:
        # extract the json block out of the response 
        json_block = response.split('```json')[1].split('```')[0]
        json_dict = json.loads(json_block)
        score = json_dict['score']
        if score >= 9:
            _SAVE_ORNOT = True
        else:
            _SAVE_ORNOT = False 
            raise ValueError(f"Score is too low, failed to save the task!\n{response}")
    
    if _SAVE_ORNOT:
        shutil.copytree(original_task, output_task, dirs_exist_ok=True)
        print(f"Saved the task to {output_task}") 
        oai_check_history_path = os.path.join(output_task, 'oai_check_history.json')
        save_json(history, oai_check_history_path)


def process_task_1(original_task, output_task):
    assert os.path.exists(original_task + '/ex.json')

    ex_json = load_json(original_task + '/ex.json')
    image_path = ex_json['image_path_code']
    image_path_filename = image_path.split('/')[-1]
    image_path_original = image_path_filename.replace('.png', '_original.png')
    original_image_path = os.path.join(original_task, image_path_original)

    assert os.path.exists(image_path), f"Generated image path {image_path} does not exist!"
    assert os.path.exists(original_image_path), f"Original image path {original_image_path} does not exist!"

    shutil.rmtree(output_task, ignore_errors=True)
    shutil.copytree(original_task, output_task, dirs_exist_ok=True)
    target_ex = load_json(os.path.join(output_task, 'ex.json'))
    target_image_path = os.path.join(output_task, image_path_original)
    target_ex['image_path_code'] = target_image_path
    assert os.path.exists(target_image_path)
    save_json(target_ex, os.path.join(output_task, 'ex.json'))
    print(f"Saved the task to {output_task}") 


if __name__ == "__main__":
    unfiltered_tasks_dir = sys.argv[1] # .temp/GeomVerse_D2_Convert
    filtered_tasks_dir = sys.argv[2] # .temp/GeomVerse_D2_Convert_filtered

    os.system(f'rm -r {filtered_tasks_dir}')
    os.makedirs(filtered_tasks_dir, exist_ok=True)

    for task_dir in tqdm(sorted(os.listdir(unfiltered_tasks_dir))):
        try:
            source_task = os.path.join(unfiltered_tasks_dir, task_dir)
            if not source_task.endswith('failed'):
                process_task_1(source_task, os.path.join(filtered_tasks_dir, task_dir))
            else:
                raise ValueError(f"Task {source_task} is failed, skip it!")
        except Exception as e:
            os.system(f'rm -r {task_dir}')
            print('\n' * 3)
            print('-' * 50)
            print(f"Error processing task {source_task}: {e}")
            if 'score' in str(e):
                raise e
