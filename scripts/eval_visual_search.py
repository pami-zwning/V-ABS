import argparse
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm

def normalize(text):
    """Normalize text for comparison."""
    if not text: return ""
    # Remove markdown code block markers
    text = str(text).replace("```json", "").replace("```", "")
    text = text.strip().upper()
    text = text.strip(" .")
    return text

def is_correct(gt, pred):
    """
    Check if prediction is correct.
    Uses word boundary matching for single-letter options (A/B/C/D).
    """
    if not gt or not pred:
        return False
    
    gt = normalize(gt)
    pred = normalize(pred)
        
    # Case 1: single-letter option
    if len(gt) == 1 and gt.isalpha():
        # Use word boundary to prevent "A" matching "Apple"
        pattern = re.compile(rf'\b{gt}\b')
        if pattern.search(pred):
            return True
        # Fallback matching: (A) green, Option A, Answer: A
        if f"({gt})" in pred or f"OPTION {gt}" in pred or f"ANSWER: {gt}" in pred or f"ANSWER {gt}" in pred:
            return True
        return False

    # Case 2: plain text containment match
    return gt in pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="beam", choices=["cot", "direct", "beam"])
    parser.add_argument("--model_name", type=str, default="qwen_8b_instruct")
    # Default to vstar-related values for easy testing
    parser.add_argument("--dataset_type", type=str, default="hr_bench_4k", help="used to find the correct folder")
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    args = parser.parse_args()

    # Build result file path
    folder_name = f"visual_search_{args.dataset_type}"
    result_dir = os.path.join(args.output_dir, args.model_name, folder_name)

    # Find prediction file: try exact match first, then glob for any matching pattern
    import glob
    result_file = os.path.join(result_dir, f"predictions_{args.type}.jsonl")
    if not os.path.exists(result_file):
        # Try pattern: predictions_{type}_bw*_d*.jsonl
        candidates = sorted(glob.glob(os.path.join(result_dir, f"predictions_{args.type}_*.jsonl")))
        if candidates:
            result_file = candidates[-1]  # Use the latest one

    if not os.path.exists(result_file):
        print(f"❌ Result file not found: {result_file}")
        return

    # Initialize statistics dictionaries
    stats = defaultdict(lambda: {'cor': 0, 'tot': 0})
    overall_stats = {'cor': 0, 'tot': 0}

    print(f"Loading: {result_file} ...")

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
                
                gt = item.get('label')
                pred = item.get('prediction', '')
                image_path = item.get('image_path', '')
                
                # --- Begin: subcategory classification based on image_path ---
                original_category = item.get('category', 'Uncategorized')
                
                # Check path keywords to override category
                if 'direct_attributes' in image_path:
                    category = 'direct_attributes'
                elif 'relative_position' in image_path:
                    category = 'relation_spatial'
                else:
                    category = original_category
                # --- End ---

                # Lowercase for display
                category = str(category).lower()

                # Update overall stats
                overall_stats['tot'] += 1
                correct = is_correct(gt, pred)
                
                if correct:
                    overall_stats['cor'] += 1
                
                # Update per-category stats
                stats[category]['tot'] += 1
                if correct:
                    stats[category]['cor'] += 1

            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    # --- Output report ---
    print(f"\n📊 Evaluation Report | Model: {args.model_name} | Dataset: {args.dataset_type}")
    print("=" * 60)
    
    col_cat = 25
    col_acc = 20
    col_count = 15
    
    header = f"{'Category':<{col_cat}} | {'Accuracy':<{col_acc}} | {'Count':<{col_count}}"
    print(header)
    print("-" * 60)

    # Show priority categories first
    # Added direct_attributes and relation_spatial to priority list
    priority_cats = ['single', 'cross', 'direct_attributes', 'relation_spatial']
    other_cats = sorted([c for c in stats.keys() if c not in priority_cats])
    
    for cat in priority_cats + other_cats:
        if cat not in stats: continue
        
        data = stats[cat]
        if data['tot'] > 0:
            acc_val = data['cor'] / data['tot'] * 100
            acc_str = f"{acc_val:.2f}%"
            count_str = f"{data['cor']}/{data['tot']}"
        else:
            acc_str = "N/A"
            count_str = "0/0"
            
        print(f"{cat:<{col_cat}} | {acc_str:<{col_acc}} | {count_str:<{col_count}}")

    print("-" * 60)
    
    # Output overall stats
    if overall_stats['tot'] > 0:
        total_acc = overall_stats['cor'] / overall_stats['tot'] * 100
        total_str = f"{total_acc:.2f}%"
        total_cnt = f"{overall_stats['cor']}/{overall_stats['tot']}"
    else:
        total_str = "N/A"
        total_cnt = "0/0"
        
    print(f"{'OVERALL':<{col_cat}} | {total_str:<{col_acc}} | {total_cnt:<{col_count}}")
    print("=" * 60)

if __name__ == "__main__":
    main()