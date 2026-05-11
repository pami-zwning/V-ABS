import json
import os
import sys
import re

sys.path.append('./workspace') # you should run this script in the workspace directory
from utils import *

from pprint import pprint

def process_one_task(task_output_dir):
    try:
        ex_json_path = os.path.join(task_output_dir, 'ex.json')
        output_path = os.path.join(task_output_dir, 'output.json')
        ex_label = float(load_json(ex_json_path)['ext_info']['label'])
        output_content = load_json(output_path)[-1]

        if isinstance(output_content['content'], str):
            output_text = output_content['content']
        else:
            output_text = output_content['content'][0]['text'] 

        assert output_content['role'] == 'assistant' 
        assert isinstance(output_text, str)

        output_answer = output_text.split('ANSWER:')[-1] 
        # extract the last number (integer or float) in the output_answer
        numbers = re.findall(r'-?\d*\.?\d+', output_answer)
        if not numbers:
            raise ValueError(f"No number found in the output_answer: {output_answer}")
        output_answer = float(numbers[-1])

        if abs(output_answer - ex_label) < 2e-2 * ex_label:
            return 1 
        else:
            return 0
    except Exception as e:
        print(f"Error processing task {task_output_dir}: {e}")
        # import pdb; pdb.set_trace()
        return None

if __name__ == "__main__":
    task_output_dir = sys.argv[1]
    tasks = os.listdir(task_output_dir)
    task_pths = [os.path.join(task_output_dir, task) for task in tasks]
    # filter out the directories only
    task_pths = [pth for pth in task_pths if os.path.isdir(pth)]
    task_pths = [pth for pth in task_pths if os.path.exists(os.path.join(pth, 'output.json'))]
    results = {pth: process_one_task(pth) for pth in mk_pbar(task_pths)}

    # print the valid pths
    for pth, res in results.items():
        if res:
            pprint(pth)

    results = [r for r in results.values() if r is not None]
    print(f"Accuracy: {(sum(results) / len(task_pths) if len(results) else 0) * 100:.1f} ({sum(results)}/{len(task_pths)})")

