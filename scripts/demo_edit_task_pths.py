import json
import os

def process_task(task_path):
    image_path = os.path.join(task_path, 'image.png')
    ex_path = os.path.join(task_path, 'ex.json')
    with open(ex_path, 'r') as f:
        ex = json.load(f)
    
    # assert os.path.exists(image_path), f"Image path {image_path} does not exist"
    ex["image_path_code"] = os.path.abspath(image_path)
    # exex_path = ex_path + '.json'
    with open(ex_path, 'w') as f:
        json.dump(ex, f, indent=4)


def main(ROOT='VisualSketchpad/tasks/geometry'):
    list_of_tasks = os.listdir(ROOT)
    for task in list_of_tasks:
        task_path = os.path.join(ROOT, task)
        if os.path.isdir(task_path):
            print(task_path)
            process_task(task_path)

if __name__ == "__main__":
    main()
