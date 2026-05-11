import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('./workspace')
import utils

def make_task(code_path, problem_text, image_path_code=None, diagram_logic_form=None, task_path=None, ext_info=None):
    with open(code_path, 'r') as f:
        code = f.read()

    ex_json = {
        'problem_text': problem_text,
        'code': code,
        'image_path_code': image_path_code,
        "logic_form": {
            "diagram_logic_form": diagram_logic_form
        },
        "ext_info": ext_info,
    }

    os.makedirs(task_path, exist_ok=True)
    with open(task_path + '/ex.json', 'w') as f:
        json.dump(ex_json, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    TEMP_ROOT = '.temp'
    DATA = TEMP_ROOT + '/GeomVerse/TEST/D2_B100/data.jsonl'
    # TASK_PATH = '.temp/test_geomverse/test_geomverse_TEST_D3_data_{idx}'
    TASK_PATH = '.temp/test_geomverse/test_geomverse_TEST_D2_B100_data_{idx}'
    # CODE_PATH = 'workspace/.temp/figure_{idx}.py'
    CODE_PATH = 'workspace/.temp/figure_d2.py'
    DATASET_NAME = 'GeomVerse'
    DATASET_DESCRIPTION = f'This is a {DATASET_NAME} problem. '

    def get_png_name(image_path):
        return image_path.split('/')[-1].replace('.jpeg', '.png')
    
    def execute_png(code_path, target_image_path):
        with open(code_path, 'r') as f:
            code = f.read()
        code = code.replace('plt.show()', f'plt.savefig("{target_image_path}")')
        try:
            exec(code, globals()) # globals() is required to access imported modules
        except Exception as e:
            print(f'Error in executing {code}: {e}')

    data = utils.load_jsonl(DATA) 
    # for idx in [0, 2, 3, 4, 5, 9]:
    for idx in [2]:
        question = DATASET_DESCRIPTION + data[idx]['question']
        code_path = CODE_PATH.format(idx=idx)
        task_path = TASK_PATH.format(idx=idx)
        ext_info = {"label": data[idx]['label'], "cot": data[idx]['cot']}
        image_path = TEMP_ROOT + data[idx]['image_path']

        os.makedirs(task_path, exist_ok=True)
        image_name = get_png_name(image_path)
        target_image_path = f'{task_path}/{image_name}'
        execute_png(code_path=code_path, target_image_path=target_image_path)
        target_image_path = os.path.abspath(target_image_path)
        
        make_task(
            code_path = code_path,
            problem_text = question,
            task_path = task_path,
            ext_info=ext_info,
            image_path_code=target_image_path,
        )
        print(task_path)

