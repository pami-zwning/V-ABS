import argparse
import os
import json
import multiprocessing
import glob
from functools import partial
from tqdm import tqdm
import traceback
import re
import numpy as np  # for computing averages

from agent import PixelMazeAgent, GridNavAgent, FrozenLakeAgent

def extract_option(response_text, options):
    """Specific to Pixel Maze Task"""
    match = re.search(r"Answer:\s*([A-F])", response_text, re.IGNORECASE)
    if match: return match.group(1).upper()
    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        if key in response_text and len(response_text) < 5: return key
    return "F"

# --- Pixel Maze Worker ---
def process_pixel_maze_task(task_item, dataset_root, output_dir_base, model_name, search_type, beam_width, max_depth, samples_per_node, verbose, entropy_skip_threshold=-1.0, token_set='yes_no', mu_threshold=0.5, no_heuristic=False):
    task_id = task_item['id']
    try:
        # Skip if already completed
        result_path = os.path.join(output_dir_base, str(task_id), "final_result.json")
        if os.path.exists(result_path):
            return None

        img_full_path = os.path.join(dataset_root, task_item['image_1'])
        if not os.path.exists(img_full_path): return None
        
        options = {}
        for line in task_item['prompt'].split('\n'):
            line = line.strip()
            if len(line) > 2 and line[0] in ['A', 'B', 'C', 'D', 'E', 'F'] and line[1] in ['.', ')']:
                options[line[0]] = line[2:].strip()

        agent = PixelMazeAgent(
            task_id=task_id, image_path=img_full_path, prompt_text=task_item['prompt'],
            options=options, output_dir=output_dir_base, model_name=model_name,
            search_type=search_type, beam_width=beam_width, max_depth=max_depth,
            samples_per_node=samples_per_node, verbose=verbose,
            entropy_skip_threshold=entropy_skip_threshold,
            token_set=token_set, mu_threshold=mu_threshold, no_heuristic=no_heuristic
        )

        model_response, found_path = agent.run()
        pred = extract_option(model_response, options)
        
        result = {
            "id": task_id, "predicted_answer": pred, "model_raw_response": model_response,
            "ground_truth": task_item.get('answer'), "is_correct": (pred == task_item.get('answer')),
            "found_path": found_path
        }
        with open(os.path.join(agent.task_dir, "final_result.json"), 'w') as f:
            json.dump(result, f, indent=4)
        return None # Maze tasks do not require uncertainty
    except Exception as e:
        traceback.print_exc()
        return None

# --- Grid Nav Worker ---
def process_grid_nav_task(task_path, output_dir_base, model_name, search_type, beam_width, max_depth, samples_per_node, verbose, entropy_skip_threshold=-1.0, token_set='yes_no', mu_threshold=0.5, no_heuristic=False):
    try:
        if not os.path.isdir(task_path): return None
        # Skip if already completed
        task_name = os.path.basename(task_path)
        result_path = os.path.join(output_dir_base, task_name, "final_result.json")
        if os.path.exists(result_path):
            return None

        agent = GridNavAgent(
            task_input=task_path, output_dir=output_dir_base, model_name=model_name,
            search_type=search_type, beam_width=beam_width, max_depth=max_depth,
            samples_per_node=samples_per_node, verbose=verbose,
            entropy_skip_threshold=entropy_skip_threshold,
            token_set=token_set, mu_threshold=mu_threshold, no_heuristic=no_heuristic
        )
        
        # agent.run() now returns first_step_uncertain
        first_step_uncertain = agent.run()
        
        # Return uncertainty value (may be float or None)
        return first_step_uncertain

    except Exception as e:
        traceback.print_exc()
        return None

# --- Frozen Lake Worker ---
def process_frozen_lake_task(task_path, output_dir_base, model_name, search_type, beam_width, max_depth, samples_per_node, verbose, entropy_skip_threshold=-1.0, token_set='yes_no', mu_threshold=0.5, no_heuristic=False):
    try:
        if not os.path.isdir(task_path): return None

        # Skip if already completed
        task_name = os.path.basename(task_path)
        parent_dir_name = os.path.basename(os.path.dirname(task_path))
        if parent_dir_name in ['4x4', '6x6', '8x8']:
            final_output_dir = os.path.join(output_dir_base, parent_dir_name)
        else:
            final_output_dir = output_dir_base

        result_path = os.path.join(final_output_dir, task_name, "final_result.json")
        if os.path.exists(result_path):
            return None

        agent = FrozenLakeAgent(
            task_input=task_path, output_dir=final_output_dir, model_name=model_name,
            search_type=search_type, beam_width=beam_width, max_depth=max_depth,
            samples_per_node=samples_per_node, verbose=verbose,
            entropy_skip_threshold=entropy_skip_threshold,
            token_set=token_set, mu_threshold=mu_threshold, no_heuristic=no_heuristic
        )
        first_step_uncertain = agent.run()
        print(f'first_step_uncertain:{first_step_uncertain}')
        
        # Return uncertainty value (may be float or None)
        return first_step_uncertain
    except Exception as e:
        traceback.print_exc()
        return None

# --- Main Runner ---
def run_agent(args):
    print(f">> Mode: {args.task_mode} | Search: {args.type} | Model: {args.model_name}")
    
    tasks = []
    worker = None

    if args.task_mode == 'maze':
        # Setup for Pixel Maze
        json_path = os.path.join(args.dataset_root, args.json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        tasks = [t for t in data if t.get('task') == 'maze']
        
        worker = partial(process_pixel_maze_task, dataset_root=args.dataset_root, output_dir_base=args.output_dir,
                         model_name=args.model_name, search_type=args.type, beam_width=args.beam_width,
                         max_depth=args.max_depth, samples_per_node=args.samples_per_node, verbose=args.verbose,
                         entropy_skip_threshold=args.entropy_skip_threshold,
                         token_set=args.token_set, mu_threshold=args.mu_threshold, no_heuristic=args.no_heuristic)

    elif args.task_mode == 'frozen_lake':
        # Setup for Frozen Lake
        search_path = args.tasks_path
        if not os.path.exists(search_path):
            print(f"Path not found: {search_path}")
            return

        for root, dirs, files in os.walk(search_path):
            for d in dirs:
                if d.startswith("task_"):
                    tasks.append(os.path.join(root, d))
        tasks.sort()
        
        worker = partial(process_frozen_lake_task, output_dir_base=args.output_dir,
                         model_name=args.model_name, search_type=args.type, beam_width=args.beam_width,
                         max_depth=args.max_depth, samples_per_node=args.samples_per_node, verbose=args.verbose,
                         entropy_skip_threshold=args.entropy_skip_threshold,
                         token_set=args.token_set, mu_threshold=args.mu_threshold, no_heuristic=args.no_heuristic)

    else: # navigation
        # Setup for Grid Navigation
        tasks = [t for t in glob.glob(f"{args.tasks_path}/*") if os.path.isdir(t)]
        tasks.sort()
        
        worker = partial(process_grid_nav_task, output_dir_base=args.output_dir,
                         model_name=args.model_name, search_type=args.type, beam_width=args.beam_width,
                         max_depth=args.max_depth, samples_per_node=args.samples_per_node, verbose=args.verbose,
                         entropy_skip_threshold=args.entropy_skip_threshold,
                         token_set=args.token_set, mu_threshold=args.mu_threshold, no_heuristic=args.no_heuristic)

    if not tasks:
        print("No tasks found.")
        return

    # Collect results list
    results = []

    if args.num_workers == 1:
        for t in tqdm(tasks, desc="Running"): 
            res = worker(t)
            if res is not None: results.append(res)
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            # imap_unordered returns an iterator; iterate and collect
            for res in tqdm(pool.imap_unordered(worker, tasks, chunksize=1), total=len(tasks)):
                if res is not None:
                    results.append(res)

    # Only compute average Uncertainty for Navigation mode
    if args.task_mode == 'frozen_lake' and results:
        # Filter to numeric types (exclude possible error message strings)
        uncertainties = [r for r in results if isinstance(r, (int, float))]
        
        if uncertainties:
            avg_unc = sum(uncertainties) / len(uncertainties)
            print("\n" + "="*40)
            print(f"📊 Global Average First-Step Uncertainty: {avg_unc:.4f}")
            print(f"   (Calculated over {len(uncertainties)} successful tasks)")
            print("="*40 + "\n")
        else:
            print("\n⚠️ No valid uncertainty values collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Common Args
    parser.add_argument("--model_name", type=str, default="qwen_8b_instruct")
    parser.add_argument("--type", type=str, default='beam', help="Search strategy: direct, beam, v_labs")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--samples_per_node", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--entropy_skip_threshold", type=float, default=-1.0,
                       help="Skip observer when prior entropy < this; -1=disabled")
    parser.add_argument("--token_set", type=str, default="yes_no",
                       choices=["yes_no", "true_false", "correct_incorrect", "all"],
                       help="(E4) Logit token set for scoring")
    parser.add_argument("--mu_threshold", type=float, default=0.5,
                       help="(E5) Entropy threshold mu for adaptive weighting")
    parser.add_argument("--no_heuristic", action="store_true",
                       help="(E7) Disable heuristic score")
    parser.add_argument("--output_dir", type=str, default="",
                       help="Override output directory (default: auto-generated from model/task/type)")
    
    # Mode Selection
    parser.add_argument("--task_mode", type=str, choices=['maze', 'navigation', 'frozen_lake'], default="maze", help="Choose task type")

    # Pixel Maze Specific
    parser.add_argument("--dataset_root", type=str, default="datasets/maze")
    parser.add_argument("--json_file", type=str, default="maze.json")

    # Grid Nav / Frozen Lake Specific
    parser.add_argument("--tasks_path", type=str, default="level-3")
    
    args = parser.parse_args()

    # Path Normalization — include depth in output path to separate experiments
    def _make_output_dir(base_name):
        """Build output dir, encoding beam_width and max_depth to avoid overwriting."""
        if args.output_dir:
            return args.output_dir
        return f"outputs/{args.model_name}/{base_name}_bw{args.beam_width}_d{args.max_depth}"

    if args.task_mode == 'maze':
        args.output_dir = _make_output_dir(f"navigation_maze_{args.type}")
        run_agent(args)
    elif args.task_mode == 'frozen_lake':
        base_path = "datasets/frozen_lake"
        
        if not args.tasks_path.startswith("/"):
             if args.tasks_path in ['4x4', '6x6', '8x8']:
                 args.tasks_path = os.path.join(base_path, args.tasks_path)
             else:
                 # Default to root directory
                 args.tasks_path = base_path
        
        args.output_dir = _make_output_dir(f"navigation_frozen_{args.type}")
        run_agent(args)
    else: # navigation
        # Navigation runs three difficulty levels; loop and print per-level averages
        # For simplicity, keep the loop approach, printing each level's average
        base_config_path = "datasets/visual-navigation/configurations/"
        levels = ["level-3", "level-4", "level-5"]
        
        # If user specified a specific tasks_path other than the default, run only that level
        if args.tasks_path != "level-4" and args.tasks_path in levels:
            target_levels = [args.tasks_path]
        else:
            # If default value, or user wants to run all, adjust as needed
            # Keep original logic: loop over levels 3, 4, 5
            target_levels = levels

        for level in target_levels:
            print(f"\n🚀 Starting Navigation Task: {level}")
            full_path = os.path.join(base_config_path, level)
            
            # Create an independent args copy to avoid pollution
            current_args = argparse.Namespace(**vars(args))
            current_args.tasks_path = full_path
            current_args.output_dir = _make_output_dir(f"navigation_nav_{args.type}") + f"/{level}"
            
            run_agent(current_args)