"""
Entry Point for Visual Search V2
Enhanced with hierarchical beam search and multi-target cooperation
"""

import argparse
import os
import json
import jsonlines
import time
import sys
import multiprocessing
import signal
from functools import partial
from tqdm import tqdm
import traceback

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agent import VisualSearchAgent


# --- Timeout handling ---
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out")


def load_data(data_file, dataset_type):
    """
    Load data based on dataset type and standardize format

    Returns: [{'question_id': ..., 'image': ..., 'text': ..., 'label': ..., 'category': ...}, ...]
    """
    questions = []

    if dataset_type == 'vstar':
        # VStar Bench
        with jsonlines.open(data_file) as reader:
            for obj in reader:
                obj['category'] = 'vstar_default'
                questions.append(obj)

    elif 'hr_bench' in dataset_type:
        # HR-Bench
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            # HR-Bench contains multiple option orderings (Cyclic Eval), use first one
            cycle_idx = 0

            # 1. Construct ID
            q_id = item.get('index', str(item.get('image_id', '')))

            # 2. Construct text: Question + Options
            options_list = item.get('options', [])
            if isinstance(options_list, list) and len(options_list) > 0:
                option_str = options_list[cycle_idx]
            else:
                option_str = str(options_list)

            question_text = item.get('question', '')
            full_text = f"{question_text}\n{option_str}"

            # 3. Get Label
            answer_list = item.get('answer', [])
            if isinstance(answer_list, list) and len(answer_list) > 0:
                label = answer_list[cycle_idx]
            else:
                label = str(answer_list)

            # 4. Get image relative path
            img_path = item.get('input_image', '')

            # 5. Get Category (single / cross)
            category = item.get('category', 'unknown')

            questions.append({
                "question_id": q_id,
                "image": img_path,
                "text": full_text,
                "label": label,
                "category": category
            })
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return questions


def process_single_item(item, args, final_output_dir):
    """
    Process a single item with timeout control
    """
    # Set timeout alarm (Linux/Mac only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)

    try:
        # Initialize Agent
        agent = VisualSearchAgent(
            output_dir=final_output_dir,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            model_name=args.model_name,
            num_threads=args.num_threads,
            verbose=False,  # Reduce log clutter in multiprocessing
            token_set=getattr(args, 'token_set', 'yes_no'),
            mu_threshold=getattr(args, 'mu_threshold', 0.5),
            no_heuristic=getattr(args, 'no_heuristic', False),
            entropy_skip_threshold=getattr(args, 'entropy_skip_threshold', -1.0),
        )

        q_id = item['question_id']
        img_rel_path = item['image']

        # Construct full image path
        img_full_path = os.path.join(args.img_root, img_rel_path)

        if not os.path.exists(img_full_path):
            # Try alternative path: sometimes json has only filename
            alt_path = os.path.join(args.img_root, "image", os.path.basename(img_rel_path))
            if os.path.exists(alt_path):
                img_full_path = alt_path
            else:
                # Image doesn't exist, skip
                print(f"⚠️ Image not found: {img_full_path}")
                return None

        res_item = {
            "question_id": q_id,
            "label": item.get('label'),
            "image_path": img_rel_path,
            "category": item.get('category')
        }

        start_t = time.time()

        # --- Run Agent (core logic) ---
        if args.type == 'direct':
            pred, reason = agent.run_direct(
                img_path=img_full_path,
                question_text=item['text'],
                q_id=q_id
            )
        else:
            pred = agent.run(
                img_path=img_full_path,
                question_text=item['text'],
                q_id=q_id
            )
            reason = "See detailed logs in 'logs/' folder"

        res_item['prediction'] = pred
        res_item['reasoning'] = reason
        res_item['time_cost'] = time.time() - start_t

        return res_item

    except TimeoutException:
        print(f"⏰ Timeout: Task {item.get('question_id')} took > {args.timeout}s. Skipping.")
        return None
    except Exception as e:
        print(f"❌ Error processing {item.get('question_id', 'unknown')}: {e}")
        traceback.print_exc()
        return None
    finally:
        # Cancel timeout alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def run_visual_search(result_file, args):
    """
    Main execution function
    """
    # 1. Load data
    try:
        print(f"Loading data from {args.data_file} (Type: {args.dataset_type})...")
        questions = load_data(args.data_file, args.dataset_type)
        print(f"Loaded {len(questions)} questions. Model: {args.model_name}")
    except Exception as e:
        print(f"❌ Load Data Error: {e}")
        return

    # 2. Read already processed IDs
    processed_ids = set()
    if os.path.exists(result_file):
        try:
            with jsonlines.open(result_file) as reader:
                for obj in reader:
                    processed_ids.add(str(obj['question_id']))
        except Exception as e:
            print(f"Warning reading result file: {e}")

    tasks = [q for q in questions if str(q['question_id']) not in processed_ids]
    print(f"Remaining tasks: {len(tasks)}")

    if not tasks:
        print("All tasks completed.")
        return

    # Output directory
    final_output_dir = os.path.join(args.output_dir, args.model_name, f'visual_search_{args.dataset_type}')
    os.makedirs(final_output_dir, exist_ok=True)

    # 3. Execute with multiprocessing
    print(f"Starting execution with {args.num_workers} workers (Timeout: {args.timeout}s)...")
    print(f"Algorithm: Hierarchical Beam Search (beam_width={args.beam_width}, max_depth={args.max_depth})")

    process_func = partial(process_single_item, args=args, final_output_dir=final_output_dir)

    with jsonlines.open(result_file, mode='a', flush=True) as writer:
        if args.num_workers <= 1:
            # Serial execution
            for task in tqdm(tasks, desc="Processing"):
                res = process_func(task)
                if res:
                    writer.write(res)
        else:
            # Parallel execution
            with multiprocessing.Pool(processes=args.num_workers) as pool:
                iterator = pool.imap_unordered(process_func, tasks, chunksize=1)
                for result in tqdm(iterator, total=len(tasks), desc="Processing"):
                    if result is not None:
                        writer.write(result)

    print(f"\n✅ Processing complete. Results saved to: {result_file}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Visual Search V2 - Hierarchical Beam Search")

    # Dataset configuration
    parser.add_argument("--img_root", type=str,
                       default="datasets/visual-search",
                       help="Root directory for images")
    parser.add_argument("--dataset_type", type=str, default="hr_bench_4k",
                       choices=["vstar", "hr_bench_4k", "hr_bench_8k"],
                       help="Dataset type")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for results")

    # Algorithm configuration
    parser.add_argument("--max_depth", type=int, default=4,
                       help="Maximum search depth")
    parser.add_argument("--beam_width", type=int, default=3,
                       help="Beam width (K candidates per depth)")
    parser.add_argument("--num_threads", type=int, default=8,
                       help="Number of threads for parallel VLM queries")

    # Execution mode
    parser.add_argument("--type", type=str, default="beam",
                       choices=["beam", "direct"],
                       help="Execution type: 'beam' for hierarchical search, 'direct' for baseline VQA")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="qwen_8b_instruct",
                       help="VLM model name")

    # Parallel execution
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of parallel workers for dataset processing")
    parser.add_argument("--timeout", type=int, default=900,
                       help="Timeout per sample in seconds (default: 15 min)")

    # === Rebuttal experiment flags ===
    parser.add_argument("--token_set", type=str, default="yes_no",
                       choices=["yes_no", "true_false", "correct_incorrect", "all"],
                       help="(E4) Logit token set for scoring")
    parser.add_argument("--mu_threshold", type=float, default=0.5,
                       help="(E5) Entropy threshold mu for adaptive weighting")
    parser.add_argument("--no_heuristic", action="store_true",
                       help="(E7) Disable heuristic score (set w_heuristic=0)")
    parser.add_argument("--entropy_skip_threshold", type=float, default=-1.0,
                       help="(E3) Skip observer when prior entropy < this; -1=disabled")

    args = parser.parse_args()

    # Construct data file path
    args.img_root = os.path.join(args.img_root, args.dataset_type)
    if args.dataset_type == "vstar":
        anno_filename = f"annotation_{args.dataset_type}.jsonl"
    else:
        anno_filename = f"annotation_{args.dataset_type}.json"
    args.data_file = os.path.join(args.img_root, anno_filename)

    # Print configuration
    print("=" * 80)
    print("Visual Search V2 - Configuration")
    print("=" * 80)
    print(f"📂 Dataset Name   : {args.dataset_type}")
    print(f"🖼️  Image Root    : {args.img_root}")
    print(f"📄 Data File     : {args.data_file}")
    print(f"🤖 Model         : {args.model_name}")
    print(f"🔍 Search Type   : {args.type}")
    if args.type == 'beam':
        print(f"📊 Beam Width    : {args.beam_width}")
        print(f"📏 Max Depth     : {args.max_depth}")
        print(f"🧵 Num Threads   : {args.num_threads}")
    print(f"⚙️  Num Workers   : {args.num_workers}")
    print(f"⏱️  Timeout       : {args.timeout}s")
    print("=" * 80)

    # Determine result file path
    final_output_dir = os.path.join(args.output_dir, args.model_name, f"visual_search_{args.dataset_type}")
    os.makedirs(final_output_dir, exist_ok=True)

    res_filename = f"predictions_{args.type}_bw{args.beam_width}_d{args.max_depth}.jsonl"
    result_file = os.path.join(final_output_dir, res_filename)

    print(f"📝 Result File   : {result_file}\n")

    run_visual_search(result_file, args)
