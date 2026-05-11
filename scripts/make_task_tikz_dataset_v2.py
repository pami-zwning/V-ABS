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
from utils_geometry import *

 
def make_task(code, problem_text, image_path_code=None, diagram_logic_form=None, task_path=None, ext_info=None):

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



def process_data(d, working_dir):
    os.makedirs(working_dir, exist_ok=True)
    question = d['question']
    image_path = d['image_path']
    tikz_code = d['tikz']

    parser = Parser()
    executor = CodeExecutor(working_dir)
    image_path = TEMP_ROOT + image_path

    # move image to working_dir
    os.system(f'cp {image_path} {working_dir}')

    CODE_FORMAT = """
You code need to incorporate the following sections:

1. Import the necessary packages

The code need to import the following packages:
```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon, Rectangle, Wedge
from matplotlib.path import Path
import matplotlib.patches as patches

# import tools
from utils import *
```

2. Point Definition

The points are defined like this:
```python
points = {
    'A': (.., ..),
    'B': (.., ..),
    'C': (.., ..),
    ...
}
```

3. Draw the figure

You need to draw the figure according to the points and the tikz code. There are tools for you to draw the figure, which has been imported already. You need to call the tools to draw the figure.

You can call the tools like this:
```python
draw_angle_sector(ax, points['A'], points['B'], points['C'], label='xx°')
draw_right_angle(ax, points['A'], points['B'], points['C'])
draw_lines(ax, points['A'], points['B'])
draw_pentagon(ax, [points['A'], points['B'], points['C'], points['D'], points['E']])
...
```

4. Annotate the points

(1) You need to annotate the points in the figure like this, not use for or while loop. 
(2) Also you should only annotate the points that are defined in the tikz code.
```python
ax.text(points['A'][0], points['A'][1] - 1, 'A', fontsize=12, ha='center')
ax.text(points['B'][0], points['B'][1] - 1, 'B', fontsize=12, ha='center')
ax.text(points['C'][0], points['C'][1] - 1, 'C', fontsize=12, ha='center')
...
```

5. Figure Format and Figure Show

You need to incorporate the following format:

```python
# Set figure format
max_x = max([p[0] for p in points.values()])
min_x = min([p[0] for p in points.values()])
max_y = max([p[1] for p in points.values()])
min_y = min([p[1] for p in points.values()])
ax.set_xlim(min_x - 1, max_x + 1)
ax.set_ylim(min_y - 1, max_y + 1)

ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.show()
``` 

"""

    with open(f'{os.path.dirname(__file__)}/utils_geometry.py', 'r') as f:
        tool_code = f.read()
    tool_code = [c for c in tool_code.split('\n') if not (c.startswith('import ') or c.startswith('from '))]
    tool_code = '\n'.join(tool_code)

    TOOLS = f"""
```
Here are the tools for you to draw the figure, you can only call the following tools to draw the figure.

Please strictly follow the description of the tool when you call it.

{tool_code}
```
"""

    PROMPT = f"""
Your task is to generate the corresponding python code with matplotlib for the given image and its tikz code.
## Requirements ##

** Here are code format and figure format requirements: **
{CODE_FORMAT}

** Here are some tools for you to Draw the figure (step 3): **
{TOOLS}

## INPUT: tikz code ##
```latex
 {tikz_code}
```

Please generate the Python code according to the requirements strictly and the tikz code.
"""

    print(PROMPT)
    # exit()
    return

    content, messages = chat_gpt4o(PROMPT)

    # print(content)
    # print('-' * 50 + '\n' * 3)

    for i in range(2):
        if 'TERMINATE' in content:
            break 

        result = parser.parse(content)
        assert result['status'], f"{content}"
        exec_result = executor.execute(result['content'])
        
        if exec_result[0] == 0:
            # OK 
            img_str = exec_result[1]
            prompt = f"""
            OBSERVATION: Execution success. The output is as follows:
            {img_str}
            - The python code must cover all the information that appear in the figure
            - The rendered output should geometrically equivalent to the ## TARGET GEOMETRY ##, and avoid to cause obvious confusion for adding, losing or twisting information (including quite wrong length, radius or positions, etc)
            - Pay attention to the shape of the geometry, the output figure should be strictly the same as the ## TARGET GEOMETRY ##.
            - Any element is shown completely in the figure, else adjust x lim and y lim for better presentation.
            **please compare the output with ## TARGET GEOMETRY ## carefully**
            - If the output satisfies, reply with TERMINATE
            - else, reply GENERATE and generate the fixed code.
            """
        else:
            # ERROR 
            prompt = f"""
            OBSERVATION: Execution error. Error message:
            {exec_result[1]}
            Please fix the error and generate the fixed code, in the next THOUGHT and ACTION.
            """
        
        # print(prompt)
        # print('-' * 50 + '\n' * 3)

        content, messages = chat_gpt4o(prompt, messages)

        # print(content)
        # print('-' * 50 + '\n' * 3)
    json.dump(messages, open(os.path.join(working_dir, 'generate_oai.json'), 'w'), indent=4)

    if 'TERMINATE' in content:
        return True, result['content']
    else:
        return False, None



if __name__ == "__main__":
    """Usage: python workspace/scripts/make_task_tikz.py $1"""

    TEMP_ROOT = '.temp/'
    # DATA = TEMP_ROOT + '/GeomVerse/TEST/D3/data.jsonl'
    DATA = TEMP_ROOT + '/GeomVerse/TEST/D2_B100/data.jsonl'
    # DATA = TEMP_ROOT + '/GeomVerse/TEST/D1/data.jsonl'
    # TASK_PATH = '.temp/test_geomverse/test_geomverse_TEST_D3_data_{idx}'
    TASK_PATH = '.temp/GeomVerse_D2_Convert/test_geomverse_TEST_D2_B100_data_{idx}'
    # TASK_PATH = '.temp/test_geomverse/test_geomverse_TEST_D1_data_{idx}'
    DATASET_NAME = 'GeomVerse'
    DATASET_DESCRIPTION = f'This is a {DATASET_NAME} problem. '

    def get_png_name(image_path):
        return image_path.split('/')[-1].replace('.jpeg', '.png')
    
    def execute_png(code, target_image_path):
        code = code.replace('plt.show()', f'plt.savefig("{target_image_path}")')
        try:
            exec(code, globals()) # globals() is required to access imported modules
        except Exception as e:
            print(f'Error in executing {code}: {e}')

    data = load_jsonl(DATA) 

    idxs = list(range(23, 28))
    for idx in mk_pbar(idxs):
        with open(f'./temp_{idx}.txt', 'w') as f:
            sys.stdout = f

            try:
                question = DATASET_DESCRIPTION + data[idx]['question']
                task_path = TASK_PATH.format(idx=idx)
                ext_info = {"label": data[idx]['label'], "cot": data[idx]['cot']}
                image_path = TEMP_ROOT + data[idx]['image_path']
                sys.stderr.write(image_path + '\n')

                os.system(f'rm -r {task_path}')
                os.system(f'rm -r {task_path}_failed')

                results = process_data(data[idx], task_path)
                # code = results[1]
                # if not results[0]:
                #     raise ValueError(f"Failed to generate code for {idx}")

                # os.makedirs(task_path, exist_ok=True)
                # image_name = get_png_name(image_path)
                # convert_jpeg_to_png(image_path, target_dir=task_path)
                # target_image_path = f'{task_path}/{image_name}'
                # execute_png(code, target_image_path=target_image_path)
                # target_image_path = os.path.abspath(target_image_path)
            
                # make_task(
                #     code = code,
                #     problem_text = question,
                #     task_path = task_path,
                #     ext_info=ext_info,
                #     image_path_code=target_image_path,
                # )
            except Exception as e:
                print(e)
                print(f"Failed to generate code for {idx}")
                if 'keyboard' in str(e).lower():
                    print(f"Keyboard interrupt, exit!")
                    exit()
                # rename the task_path to failed
                shutil.rmtree(task_path + '_failed', ignore_errors=True)
                os.rename(task_path, task_path + '_failed')
                continue

