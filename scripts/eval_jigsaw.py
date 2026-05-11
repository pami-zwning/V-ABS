import json
import os
import argparse


def get_neighbor_score(pred_order, gt_order, res):
    """
    Compute neighbor accuracy (pairwise accuracy).
    Check if each pair of adjacent tiles (up/down/left/right) in pred
    also appears as neighbors in gt.
    """
    # Build GT position map: {tile_id: (row, col)}
    gt_map = {}
    for idx, tile_id in enumerate(gt_order):
        r, c = divmod(idx, res)
        gt_map[tile_id] = (r, c)

    correct_neighbors = 0
    total_neighbors = 0

    for i, tile_id in enumerate(pred_order):
        if tile_id not in gt_map:
            continue

        pred_r, pred_c = divmod(i, res)

        # Check right neighbor (horizontal)
        if pred_c < res - 1:
            total_neighbors += 1
            right_tile_id = pred_order[i + 1]
            if right_tile_id in gt_map:
                curr_gt_r, curr_gt_c = gt_map[tile_id]
                right_gt_r, right_gt_c = gt_map[right_tile_id]
                if curr_gt_r == right_gt_r and right_gt_c == curr_gt_c + 1:
                    correct_neighbors += 1

        # Check bottom neighbor (vertical)
        if pred_r < res - 1:
            total_neighbors += 1
            bottom_tile_id = pred_order[i + res]
            if bottom_tile_id in gt_map:
                curr_gt_r, curr_gt_c = gt_map[tile_id]
                bottom_gt_r, bottom_gt_c = gt_map[bottom_tile_id]
                if curr_gt_c == bottom_gt_c and bottom_gt_r == curr_gt_r + 1:
                    correct_neighbors += 1

    return correct_neighbors, total_neighbors


def calculate_accuracy(gt_file_path, base_output_dir, res):
    """Evaluate jigsaw puzzle accuracy for a single resolution/sample combination."""
    if not os.path.exists(gt_file_path):
        print(f"Error: Ground truth file not found: {gt_file_path}")
        return

    with open(gt_file_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)

    # Statistics
    total_tiles_absolute = 0
    correct_tiles_absolute = 0
    total_pairs_neighbor = 0
    correct_pairs_neighbor = 0
    missing_files = 0

    print(f"{'Image Name':<40} | {'Abs Acc':<8} | {'Neighbor Acc (Robust)'}")
    print("-" * 80)

    for filename, gt_order in ground_truth_data.items():
        folder_name = os.path.splitext(filename)[0]
        solution_path = os.path.join(base_output_dir, folder_name, 'solution.json')

        if not os.path.exists(solution_path):
            missing_files += 1
            continue

        try:
            with open(solution_path, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)

            pred_order = pred_data.get("final_permutation", [])

            if len(pred_order) != res * res:
                print(f"Warning: Length mismatch for {filename}")
                continue

            # Metric 1: Absolute position accuracy
            img_correct_abs = sum(1 for i in range(res * res) if gt_order[i] == pred_order[i])
            correct_tiles_absolute += img_correct_abs
            total_tiles_absolute += res * res

            # Metric 2: Neighbor accuracy
            img_correct_pairs, img_total_pairs = get_neighbor_score(pred_order, gt_order, res)
            correct_pairs_neighbor += img_correct_pairs
            total_pairs_neighbor += img_total_pairs

            abs_score = img_correct_abs / (res * res)
            nb_score = img_correct_pairs / img_total_pairs if img_total_pairs > 0 else 0

            print(f"{folder_name[:38]:<40} | {abs_score:.2%}   | {nb_score:.2%}")

        except Exception as e:
            print(f"Error reading {solution_path}: {e}")

    # Print summary
    print("-" * 80)
    if total_tiles_absolute == 0:
        print("No valid comparison data found.")
        return

    acc_absolute = correct_tiles_absolute / total_tiles_absolute
    acc_neighbor = correct_pairs_neighbor / total_pairs_neighbor if total_pairs_neighbor > 0 else 0

    print(f"Evaluation Summary:")
    print(f"Total Processed: {len(ground_truth_data) - missing_files}")
    print(f"Missing Files:   {missing_files}")
    print(f"\nMetric 1: Absolute Position Accuracy (Exact Match)")
    print(f"  > Score: {acc_absolute:.2%}")
    print(f"\nMetric 2: Neighbor Accuracy (Structure Similarity)")
    print(f"  > Score: {acc_neighbor:.2%}")
    print(f"  Note: This metric better reflects visual correctness.")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Jigsaw puzzle accuracy")

    parser.add_argument("--type", type=str, default="beam", choices=["direct", "beam"])
    parser.add_argument("--model_name", type=str, default="qwen_8b_instruct")
    parser.add_argument("--data_root", type=str, default="dataset/jigsaw",
                        help="Root directory for jigsaw dataset")
    parser.add_argument("--output_root", type=str, default="outputs",
                        help="Root directory for outputs")

    args = parser.parse_args()

    for res in [3, 4, 5]:
        for samples in [10, 50]:
            gt_file = f"{args.data_root}/res{res}/s{samples}/jigsaw_gt.json"
            # New unified path: outputs/{model}/jigsaw_{type}/res{n}/s{k}/
            output_dir = f"{args.output_root}/{args.model_name}/jigsaw_{args.type}/res{res}/s{samples}"

            print(f"\n{'='*50}")
            print(f"Resolution: {res}x{res} | Samples: {samples}")
            print(f"GT: {gt_file}")
            print(f"Output: {output_dir}")
            print(f"{'='*50}")

            calculate_accuracy(gt_file, output_dir, res)
