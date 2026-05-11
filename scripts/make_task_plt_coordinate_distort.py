import re
import sys
import glob
import json

def distort_number(number : float):
    number = float(number)
    return str(number * 1000 + 34.23)

def process_one_line(line : str):
    # extract numbers out of the line
    numbers = re.findall(r'-?\d*\.?\d+', line)
    assert len(numbers) == 2, f"Expected 2 numbers in the line: {line}"

    # avoid bug when number is 0
    line = line.replace(numbers[0], '[num_first]', 1)
    line = line.replace(numbers[1], '[num_second]', 1)
    distorted_numbers = [distort_number(num) for num in numbers]
    return line.replace('[num_first]', distorted_numbers[0], 1).replace('[num_second]', distorted_numbers[1], 1)

def process_radius_line(line : str):
    # extract numbers out of the line
    numbers = re.findall(r'-?\d*\.?\d+', line)
    assert len(numbers) == 1, f"Expected 2 numbers in the line: {line}"

    # avoid bug when number is 0
    line = line.replace(numbers[0], '[num_first]', 1)
    distorted_numbers = [distort_number(num) for num in numbers]
    return line.replace('[num_first]', distorted_numbers[0], 1)


def get_coordinate_lines(code):
    lines = code.split("\n")
    lines = [line.strip() for line in lines]
    ret_lines = None
    for i, line in enumerate(lines):
        if "#" in line and "Coordinate" in line:
            ret_lines = list(range(i+1, i+20))
            break
    
    if ret_lines is not None:
        # Find the earliest blank line and return the preceding lines
        for i, line_idx in enumerate(ret_lines):
            if lines[line_idx] == "":
                return ret_lines[:i]
        return []
    else:
        return []

def process_plt_code(code: str):
    code_lines = code.split('\n')
    # plt_show_idx 
    plt_show_idx = None
    for i, line in enumerate(code_lines):
        if 'plt.show()' in line:
            plt_show_idx = i
            break
    assert plt_show_idx is not None, f"Expected plt.show() in the code: {code.replace('\n', ' ')}"


    # find the line that contains 'ax.set_xlim' or 'ax.set_ylim', if not, insert them
    ax_idxs = []
    for line in code_lines:
        if 'ax.set_xlim(' in line or 'ax.set_ylim(' in line:
            ax_idxs.append(code_lines.index(line))
    print(ax_idxs)
    assert len(ax_idxs) == 2 or len(ax_idxs) == 0, f"Only ax.set_xlim or ax.set_ylim in the code: {code.replace('\n', ' ')}"
    if len(ax_idxs) == 0:
        code_lines.insert(plt_show_idx, 'ax.set_xlim(-20, 45)')
        code_lines.insert(plt_show_idx, 'ax.set_ylim(-20, 45)')

    ax_idxs = []
    # find the line that contains 'ax.set_xlim' or 'ax.set_ylim'
    for line in code_lines:
        if 'ax.set_xlim(' in line or 'ax.set_ylim(' in line:
            ax_idxs.append(code_lines.index(line))
    assert len(ax_idxs) == 2, f"Expected 2 ax.set_xlim or ax.set_ylim in the code: {code.replace('\n', ' ')}"

    # find the coordinate lines
    coordinate_lines = get_coordinate_lines(code)
    assert len(coordinate_lines) > 1, f"Expected at least 2 coordinate lines in the code: {code.replace('\n', ' ')}"
    print((ax_idxs + coordinate_lines))

    # find the radius line
    radius_lines = []
    for line in code_lines:
        if 'radius =' in line and len(line) < 30:
            radius_line = code_lines.index(line)
            radius_lines.append(radius_line)
    assert len(radius_lines) <= 1, f"Expected no or 1 radius line in the code: {code.replace('\n', ' ')}"

    # distort the coordinate lines
    distorted_lines = [(process_one_line(line) if i in (ax_idxs + coordinate_lines) else line) for i, line in enumerate(code_lines)]
    if len(radius_lines) >= 1:
        print('Radius line index Detected:', radius_lines[0])
        distorted_lines[radius_lines[0]] = process_radius_line(code_lines[radius_lines[0]])
    return "\n".join(distorted_lines)

if __name__ == "__main__":
    # --- UNIT TEST --- #
    # with open('workspace/.temp/output_matplotlib_3_step.py', 'r') as f:
    #     code = f.read()
    # distorted_code = process_plt_code(code)
    # with open('workspace/.temp/output_matplotlib_3_step_distorted.py', 'w') as f:
    #     f.write(distorted_code)

    data_path = sys.argv[1] # '.temp/test_geomverse'
    task_pths = glob.glob(f'{data_path}/*/')
    
    for i, task_pth in enumerate(task_pths):
        ex_pth = f'{task_pth}/ex.json'
        ex = json.load(open(ex_pth, 'r'))
        code = ex['code']
        try:
            distorted_code = process_plt_code(code)
            ex['code'] = distorted_code
        except Exception as e:
            print(e, f'Error ! {task_pth}')
            continue
        json.dump(ex, open(ex_pth, 'w'), indent=4, ensure_ascii=False)
        # with open(f'workspace/.temp/.temp_debug_{i}.py', 'w') as f:
        #     f.write(distorted_code)
        # with open(f'workspace/.temp/.temp_debug_{i}_before.py', 'w') as f:
        #     f.write(code)
