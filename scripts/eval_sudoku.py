import json
import os
import argparse
import sys


def clean_board_str(s):
    """Normalize board string: remove . _ etc, keep only digits."""
    if not s: return ""
    return str(s).replace('.', '0').replace('_', '0')


def calculate_sudoku_accuracy(data_root, index_file, output_base_dir, args):
    # 1. Load ground truth index file
    gt_file_path = os.path.join(data_root, index_file)
    if not os.path.exists(gt_file_path):
        print(f"Error: Ground truth index file not found: {gt_file_path}")
        return

    print(f"Loading Ground Truth from: {gt_file_path}")
    with open(gt_file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # Build GT dict: {puzzle_id: solution_str}
    gt_map = {}
    for item in all_data:
        p_id = str(item.get('puzzle_id', item.get('id')))
        sol = item.get('text_data', {}).get('solution_str', '')
        if sol:
            gt_map[p_id] = clean_board_str(sol)

    print(f"Loaded {len(gt_map)} puzzles from GT.")

    # 2. Locate prediction output directory
    # New unified path: outputs/{model}/sudoku_{type}_bw{bw}/
    dir_name = f"sudoku_{args.type}_bw{args.beam_width}"
    target_dir = os.path.join(output_base_dir, args.model_name, dir_name)

    # Fallback: legacy naming patterns
    if not os.path.exists(target_dir):
        for suffix in ['_vision_full', '_vision', '']:
            legacy = os.path.join(output_base_dir, f"{args.model_name}_{args.type}_beam{args.beam_width}{suffix}")
            if os.path.exists(legacy):
                target_dir = legacy
                break

    if not os.path.exists(target_dir):
        print(f"Error: Output directory not found. Tried: {target_dir}")
        return

    print(f"Evaluating Results in: {target_dir}")
    print("=" * 100)
    print(f"{'Puzzle ID':<15} | {'Filled':<8} | {'Cell Acc':<10} | {'State'}")
    print("-" * 100)

    # 3. Statistics
    total_puzzles = 0
    perfect_puzzles = 0
    total_cells = 0
    correct_cells = 0
    filled_cells = 0

    # 4. Iterate over each puzzle in GT
    for p_id, gt_str in gt_map.items():
        solution_file = os.path.join(target_dir, p_id, 'solution.json')

        if not os.path.exists(solution_file):
            continue

        total_puzzles += 1

        try:
            with open(solution_file, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)

            # Read prediction: try final_full_board first, then incremental, then legacy
            pred_str = pred_data.get('final_full_board', "")
            if not pred_str:
                pred_str = pred_data.get('final_state_incremental', "")
            if not pred_str:
                pred_str = pred_data.get('final_state', "")

            pred_str = clean_board_str(pred_str)

            # Pad/truncate to 81 characters
            if len(pred_str) < 81:
                pred_str = pred_str.ljust(81, '0')
            elif len(pred_str) > 81:
                pred_str = pred_str[:81]

            # Per-sample evaluation
            matches = 0
            img_filled = 0

            for i in range(81):
                if pred_str[i] != '0':
                    img_filled += 1
                if pred_str[i] == gt_str[i]:
                    matches += 1

            total_cells += 81
            correct_cells += matches
            filled_cells += img_filled

            is_perfect = (matches == 81)
            if is_perfect:
                perfect_puzzles += 1

            status = "Perfect" if is_perfect else ("Wrong" if img_filled == 81 else "Incomplete")
            print(f"{p_id:<15} | {img_filled}/81   | {matches/81:.2%}     | {status}")

        except Exception as e:
            print(f"Error reading {p_id}: {e}")

    # 5. Print summary
    if total_puzzles == 0:
        print("\nNo solution files found matching GT.")
        return

    cell_acc = correct_cells / total_cells
    puzzle_acc = perfect_puzzles / total_puzzles
    fill_rate = filled_cells / total_cells

    print("=" * 100)
    print(f"Evaluation Summary | model: {args.model_name} | type: {args.type}")
    print(f"Input Path: {target_dir}")
    print("-" * 50)
    print(f"Total Puzzles Evaluated:    {total_puzzles}")
    print(f"\n[Metric 1] Puzzle Accuracy (Perfect Match)")
    print(f"  > Score: {puzzle_acc:.2%} ({perfect_puzzles}/{total_puzzles})")
    print(f"\n[Metric 2] Cell Accuracy (Digit-wise)")
    print(f"  > Score: {cell_acc:.2%} ({correct_cells}/{total_cells})")
    print(f"\n[Metric 3] Fill Rate")
    print(f"  > Score: {fill_rate:.2%} ({filled_cells}/{total_cells})")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sudoku solving accuracy")

    parser.add_argument("--type", type=str, default="beam", choices=["direct", "beam"])
    parser.add_argument("--model_name", type=str, default="internvl3_8b")
    parser.add_argument("--beam_width", type=int, default=1)

    # Path configuration (should match unified_sudoku/run.py)
    parser.add_argument("--data_root", type=str,
                        default="dataset/nikoli/processed_data_with_images")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--json_file", type=str, default="data_with_images.json")

    args = parser.parse_args()

    calculate_sudoku_accuracy(args.data_root, args.json_file, args.output_root, args)
