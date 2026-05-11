import json
import os
import argparse
import glob
import sys

# ==============================================================================
# 1. Navigation Validation Logic (Step-level Accuracy & Exact Match)
# ==============================================================================

def evaluate_navigation(base_path):
    """
    Compute path overlap and exact match rate for navigation tasks.
    Operates on level-3, level-4, level-5 folder structure.
    """
    # Define task ranges (level name, start ID, end ID)
    tasks = [
        ("level-3", 1, 16),
        ("level-4", 1, 32),
        ("level-5", 1, 62)
    ]

    print(f"\n>>> Mode: Navigation Evaluation")
    print(f"Base Path: {base_path}")
    print("-" * 85)
    print(f"{'Level':<10} | {'Samples':<8} | {'Avg Step Acc':<15} | {'Exact Match Rate':<18}")
    print("-" * 85)

    total_samples = 0
    total_exact_matches = 0
    total_step_acc_sum = 0

    for level_name, start_idx, end_idx in tasks:
        scores = []
        exact_matches = 0
        valid_samples = 0
        
        for i in range(start_idx, end_idx + 1):
            folder_path = os.path.join(base_path, level_name, str(i))
            pred_file = os.path.join(folder_path, "output.json")
            gt_file = os.path.join(folder_path, "detail_solution.json")

            if not os.path.exists(pred_file) or not os.path.exists(gt_file):
                continue
            
            try:
                # Read data
                with open(pred_file, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)
                    pred_path = pred_data.get("path", [])

                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                    gt_solution = gt_data.get("solution", [])

                # Normalize
                pred_norm = [str(p).lower().strip() for p in pred_path]
                gt_norm = [str(g).lower().strip() for g in gt_solution]

                # Compute Step Accuracy
                max_len = max(len(pred_norm), len(gt_norm))
                if max_len == 0:
                    score = 1.0
                else:
                    match_count = 0
                    min_len = min(len(pred_norm), len(gt_norm))
                    for idx in range(min_len):
                        if pred_norm[idx] == gt_norm[idx]:
                            match_count += 1
                    score = match_count / max_len

                scores.append(score)
                valid_samples += 1

                if score == 1.0:
                    exact_matches += 1

            except Exception as e:
                pass

        # Statistics for current Level
        if valid_samples > 0:
            avg_accuracy = sum(scores) / valid_samples
            success_rate = exact_matches / valid_samples
        else:
            avg_accuracy = 0.0
            success_rate = 0.0

        print(f"{level_name:<10} | {valid_samples:<8} | {avg_accuracy:.4f}{' '*10} | {success_rate:.4f}")
        
        total_samples += valid_samples
        total_exact_matches += exact_matches
        total_step_acc_sum += sum(scores)

    # Print overall average
    print("-" * 85)
    if total_samples > 0:
        macro_avg_acc = total_step_acc_sum / total_samples
        macro_exact_rate = total_exact_matches / total_samples
        print(f"{'TOTAL':<10} | {total_samples:<8} | {macro_avg_acc:.4f}{' '*10} | {macro_exact_rate:.4f}")
    else:
        print("No valid samples found.")


# ==============================================================================
# 2. Frozen Lake Validation Logic (Adapted for Unique Path GT)
# ==============================================================================
def simulate_frozen_lake(map_desc, start_pos, goal_pos, actions):
    """
    Simulate the Frozen Lake walking process.
    Return: (Success: bool, Status: str, Steps Taken: int)
    """
    rows = len(map_desc)
    cols = len(map_desc[0])
    r, c = start_pos
    
    # Action mapping (compatible with uppercase/lowercase and initials)
    action_map = {
        'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1),
        'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1)
    }

    path_len = 0
    
    # Initial check
    if map_desc[r][c] == 'H':
        return False, "Start is Hole", 0
        
    for action in actions:
        act_key = str(action).lower().strip()
        if act_key not in action_map:
            continue
            
        dr, dc = action_map[act_key]
        nr, nc = r + dr, c + dc
        path_len += 1
        
        # 1. Out-of-bounds check (FrozenLake mechanic: stay in place when hitting wall)
        if not (0 <= nr < rows and 0 <= nc < cols):
            # Stay at (r, c)
            pass 
        else:
            # Update position
            r, c = nr, nc
        
        # 2. Check current cell
        cell_type = map_desc[r][c]
        
        # Fell into hole
        if cell_type == 'H':
            return False, "Fell in Hole", path_len
        
        # Reached goal (early termination)
        if [r, c] == goal_pos:
            return True, "Success", path_len
            
    # Check position after all actions exhausted
    if [r, c] == goal_pos:
        return True, "Success", path_len
    else:
        return False, "Not Reached", path_len

def evaluate_frozen_lake(base_path):
    sizes = ["4x4", "6x6", "8x8"]
    
    print(f"\n>>> Mode: Frozen Lake Evaluation (with GT Check)")
    print(f"Base Path: {base_path}")
    print("-" * 115)
    # Header: Size | Samples | Success Rate (sim) | Exact Match (EM) | Avg Steps | Avg Redundant Steps
    print(f"{'Size':<8} | {'N':<6} | {'Succ Rate':<12} | {'Exact Match':<12} | {'Avg Len':<10} | {'Diff Len':<10}")
    print("-" * 115)
    
    total_samples = 0
    total_success = 0
    total_exact = 0
    
    for size in sizes:
        size_dir = os.path.join(base_path, size)
        
        # Get all task directories
        task_dirs = []
        if os.path.exists(size_dir):
            task_dirs = glob.glob(os.path.join(size_dir, "task_*"))
        
        task_dirs.sort(key=lambda x: int(x.split('_')[-1]) if '_' in x else 0)
        
        valid_samples = 0
        success_count = 0     # Physical simulation success
        exact_match_count = 0 # Path exactly matches GT
        
        pred_len_sum = 0
        len_diff_sum = 0      # (Pred Len - GT Len)
        
        for task_d in task_dirs:
            info_file = os.path.join(task_d, "info.json")
            output_file = os.path.join(task_d, "output.json")
            
            # Ensure both files exist
            if not os.path.exists(info_file) or not os.path.exists(output_file):
                continue
                
            try:
                # 1. Read Ground Truth
                with open(info_file, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                    map_desc = info_data['map_desc']
                    start_pos = info_data['start_pos']
                    goal_pos = info_data['goal_pos']
                    gt_path = info_data.get('gt_path', []) # Newly generated info.json contains this field
                
                # 2. Read model predictions
                with open(output_file, 'r', encoding='utf-8') as f:
                    out_data = json.load(f)
                    pred_path = out_data.get("path", [])
                
                # 3. Validation A: Physical simulation (whether goal is reached)
                is_success, status, p_len = simulate_frozen_lake(map_desc, start_pos, goal_pos, pred_path)
                
                # 4. Validation B: Compare with GT
                # Normalize path strings (lowercase)
                pred_norm = [str(p).lower().strip() for p in pred_path]
                gt_norm = [str(g).lower().strip() for g in gt_path]
                
                is_exact = (pred_norm == gt_norm)
                
                # Statistics
                valid_samples += 1
                if is_success:
                    success_count += 1
                    pred_len_sum += p_len
                    # Only compute step difference for successful cases (Model Steps - Optimal Steps)
                    len_diff_sum += (p_len - len(gt_norm))
                
                if is_exact:
                    exact_match_count += 1
                    
            except Exception as e:
                # print(f"Error processing {task_d}: {e}")
                pass
        
        # Compute statistics for current size
        if valid_samples > 0:
            succ_rate = success_count / valid_samples
            em_rate = exact_match_count / valid_samples
            
            # Average steps (only for successful cases)
            avg_len = pred_len_sum / success_count if success_count > 0 else 0
            avg_diff = len_diff_sum / success_count if success_count > 0 else 0
        else:
            succ_rate = 0.0
            em_rate = 0.0
            avg_len = 0.0
            avg_diff = 0.0
            
        print(f"{size:<8} | {valid_samples:<6} | {succ_rate:.4f}{' '*6} | {em_rate:.4f}{' '*6} | {avg_len:.2f}{' '*4} | {avg_diff:+.2f}")
        
        total_samples += valid_samples
        total_success += success_count
        total_exact += exact_match_count

    print("-" * 115)
    if total_samples > 0:
        macro_succ = total_success / total_samples
        macro_em = total_exact / total_samples
        print(f"{'TOTAL':<8} | {total_samples:<6} | {macro_succ:.4f}{' '*6} | {macro_em:.4f}{' '*6} | {'-':<10} | {'-'}")
    else:
        print("No valid Frozen Lake samples found.")

        
# ==============================================================================
# 3. Maze Validation Logic (Option Selection Accuracy)
# ==============================================================================
def evaluate_maze(base_path, dataset_json):
    """
    Compute option selection accuracy for the maze task.
    Reads dataset_json as Ground Truth denominator, reads final_result.json under base_path as predictions.
    """
    print(f"\n>>> Mode: Maze Evaluation")
    print(f"Base Path: {base_path}")
    print(f"GT File  : {dataset_json}")
    
    # Load GT
    gt_map = {}
    total_tasks = 0
    if os.path.exists(dataset_json):
        with open(dataset_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Filter maze tasks
            data = [t for t in data if t.get('task') == 'maze']
            for item in data:
                gt_map[item['id']] = str(item['answer']).strip().upper()
            total_tasks = len(gt_map)
    else:
        print(f"[Warning] Dataset JSON not found. Calculating based on output files only.")

    if not os.path.exists(base_path):
        print(f"[Error] Directory not found: {base_path}")
        return

    print("-" * 85)
    print(f"{'Task ID':<10} | {'Prediction':<12} | {'Label':<8} | {'Status':<10}")
    print("-" * 85)

    correct_count = 0
    processed_count = 0
    
    # Determine task IDs to iterate over
    if total_tasks > 0:
        task_ids = sorted(list(gt_map.keys()))
    else:
        # Fallback: scan folders
        task_dirs = glob.glob(os.path.join(base_path, "task_*"))
        task_ids = []
        for d in task_dirs:
            try:
                tid = int(os.path.basename(d).split('_')[-1])
                task_ids.append(tid)
            except: pass
        task_ids.sort()
        total_tasks = len(task_ids)

    # Iterate and validate
    for tid in task_ids:
        folder_path = os.path.join(base_path, f"task_{tid}")
        result_file = os.path.join(folder_path, "final_result.json")
        
        label = gt_map.get(tid, "N/A")
        
        if not os.path.exists(result_file):
            if label != "N/A":
                # print(f"{tid:<10} | {'MISSING':<12} | {label:<8} | ❌ Not Found")
                pass
            continue

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
            
            prediction = str(res_data.get("predicted_answer", "F")).strip().upper()
            
            if label == "N/A":
                label = str(res_data.get("ground_truth", "N/A")).strip().upper()

            is_correct = (prediction == label)

            if is_correct:
                correct_count += 1
            processed_count += 1

        except Exception as e:
            print(f"[Error] Task {tid}: {e}")

    # Summary
    print("-" * 85)
    accuracy = (correct_count / total_tasks * 100) if total_tasks > 0 else 0.0
    coverage = (processed_count / total_tasks * 100) if total_tasks > 0 else 0.0

    print(f"Total Tasks (Denominator) : {total_tasks}")
    print(f"Processed Files           : {processed_count} (Coverage: {coverage:.2f}%)")
    print(f"Correct Predictions       : {correct_count}")
    print(f"Final Accuracy            : {accuracy:.4f}%")
    print("-" * 85)


# ==============================================================================
# Efficiency Stats (reads vlm_stats from search logs)
# ==============================================================================
def print_efficiency_stats(base_path):
    """Scan for search_log.json / output.json files and summarize vlm_stats."""
    log_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f in ('search_log.json', 'output.json'):
                log_files.append(os.path.join(root, f))

    if not log_files:
        return

    call_counts = []
    total_tokens = []
    prompt_tokens = []
    completion_tokens = []

    for lf in log_files:
        try:
            with open(lf, 'r') as fp:
                data = json.load(fp)
            stats = data.get('vlm_stats', {})
            if stats and stats.get('call_count', 0) > 0:
                call_counts.append(stats['call_count'])
                total_tokens.append(stats.get('total_tokens', 0))
                prompt_tokens.append(stats.get('prompt_tokens', 0))
                completion_tokens.append(stats.get('completion_tokens', 0))
        except Exception:
            continue

    if not call_counts:
        return

    n = len(call_counts)
    avg_calls = sum(call_counts) / n
    avg_tokens = sum(total_tokens) / n
    avg_prompt = sum(prompt_tokens) / n
    avg_completion = sum(completion_tokens) / n

    print(f"\n--- Efficiency Stats ({n} samples with vlm_stats) ---")
    print(f"  Avg API calls/sample:     {avg_calls:.1f}")
    print(f"  Avg total tokens/sample:  {avg_tokens:.0f} ({avg_tokens/1000:.1f}K)")
    print(f"  Avg prompt tokens/sample: {avg_prompt:.0f}")
    print(f"  Avg compl. tokens/sample: {avg_completion:.0f}")
    print(f"---------------------------------------------------")


# ==============================================================================
# Main Entry
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    
    # Supports 'frozen' alias to match common input conventions
    parser.add_argument("--task", type=str, required=True, 
                        choices=['nav', 'maze', 'frozen_lake', 'frozen'], 
                        help="Task type")

    parser.add_argument("--type", type=str, default='beam', help="Search Strategy")
    parser.add_argument("--model_name", type=str, default="qwen_8b_instruct", help="Model name")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Root output dir")
    parser.add_argument("--dataset_json", type=str, default="", help="Path to maze.json")
    parser.add_argument("--beam_width", type=int, default=0, help="Beam width (0=auto-detect)")
    parser.add_argument("--max_depth", type=int, default=0, help="Max depth (0=auto-detect)")

    args = parser.parse_args()

    # Path mapping
    prefix_map = {
        'nav': 'nav',
        'maze': 'maze',
        'frozen_lake': 'frozen',
        'frozen': 'frozen'
    }

    subdir_prefix = prefix_map[args.task]
    base_name = f"navigation_{subdir_prefix}_{args.type}"

    # Try exact match with bw/d suffix first, then glob, then legacy name
    if args.beam_width > 0 and args.max_depth > 0:
        result_folder_name = f"{base_name}_bw{args.beam_width}_d{args.max_depth}"
    else:
        result_folder_name = None

    if result_folder_name:
        base_path = os.path.join(args.output_dir, args.model_name, result_folder_name)
    else:
        base_path = os.path.join(args.output_dir, args.model_name, base_name)

    # If exact path doesn't exist, try glob for any bw/d variant
    if not os.path.exists(base_path):
        import glob as _glob
        pattern = os.path.join(args.output_dir, args.model_name, f"{base_name}_bw*_d*")
        candidates = sorted(_glob.glob(pattern))
        if candidates:
            base_path = candidates[-1]  # Use latest
            print(f"Auto-detected output dir: {base_path}")
        else:
            # Try legacy path without bw/d
            legacy = os.path.join(args.output_dir, args.model_name, base_name)
            if os.path.exists(legacy):
                base_path = legacy

    if args.task == 'nav':
        evaluate_navigation(base_path)
    elif args.task == 'maze':
        evaluate_maze(base_path, args.dataset_json)
    elif args.task in ['frozen_lake', 'frozen']:
        evaluate_frozen_lake(base_path)

    # Always print efficiency stats if search logs exist
    print_efficiency_stats(base_path)