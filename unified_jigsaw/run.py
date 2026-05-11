import argparse
import glob
import os
import json
import time
import sys
import traceback
import multiprocessing
from functools import partial

# Ensure modules in the same directory can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import JigsawEnvironment
from agent import JigsawAgent

def process_single_image(img_info):
    """
    Worker function for processing a single image.
    Returns: first_step_uncertainty (float) or None
    """
    img_path, output_dir, args = img_info
    
    img_name = os.path.basename(img_path)
    img_stem = os.path.splitext(img_name)[0]
    
    # Create a separate output directory for each image
    task_output_dir = os.path.join(output_dir, img_stem)
    
    try:
        os.makedirs(task_output_dir, exist_ok=True)
    except Exception:
        pass
    
    log_path = os.path.join(task_output_dir, 'solution.json')

    # Resume from checkpoint: skip if result already exists
    if os.path.exists(log_path):
        if args.verbose:
            print(f"[Skip] {img_name} (Already exists)")
        return None  # Skipped files are excluded from the average

    print(f"[Run] Processing {img_name}...")

    try:
        # --- Initialize environment and agent ---
        env = JigsawEnvironment(img_path, args.res, output_dir)
        agent = JigsawAgent(env, model_name=args.model_name, verbose=args.verbose,
                            entropy_skip_threshold=args.entropy_skip_threshold)
        
        start_time = time.time()
        final_perm = []
        first_step_uncertainty = None  # Initialize variable
        
        # --- Dispatch algorithm based on type ---
        if args.type == "direct":
            final_perm = agent.run_direct()
            first_step_uncertainty = 0.0  # No search uncertainty in direct mode
            
        elif args.type in ["beam", "v_labs", "v-bas"]:
            # Receive uncertainty return value from run_beam
            res = agent.run_beam(
                beam_width=args.beam_width, 
                max_depth=args.max_depth
            )
            
            if isinstance(res, tuple) and len(res) >= 2:
                final_perm = res[0]
                first_step_uncertainty = res[1]
            else:
                final_perm = res
                first_step_uncertainty = None

        duration = time.time() - start_time
        
        # --- Save results ---
        print(f"  ✅ [{img_name}] Result: {final_perm}")
        
        # 1. Save final puzzle image
        env.rearrange_tiles(final_perm, tag="FINAL_RESULT", is_simulation=False)
        
        # 2. Save JSON log (including uncertainty)
        log_data = {
            "method": args.type,
            "model_name": args.model_name,
            "dataset_res": args.res,
            "final_permutation": final_perm,
            "first_step_uncertainty": first_step_uncertainty,
            "duration": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)

        return first_step_uncertainty

    except Exception as e:
        print(f"  ❌ Error processing {img_path}: {e}")
        traceback.print_exc()
        return None

def process_images(data_dir, output_dir, args):
    """
    Batch-process images in a folder (multiprocessing version).
    Returns: list of valid uncertainties for this batch.
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist. Skipping...")
        return []

    # 1. Collect all images
    images = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not images:
        print(f"No images found in {data_dir}. Checking subfolders...")
        images = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)

    if not images:
        print(f"Warning: No images found in {data_dir} even after recursive check. Skipping...")
        return []

    images = sorted(images)
    print(f"\n{'='*60}")
    print(f"📂 Processing Batch: Res={args.res} | Path={data_dir}")
    print(f"Found {len(images)} images. Mode: {args.type}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"{'='*60}")
    
    # 2. Build task list
    tasks = []
    for img_path in images:
        tasks.append((img_path, output_dir, args))

    # 3. Execute with multiprocessing and collect results
    results = []
    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # Collect return values
            results = pool.map(process_single_image, tasks, chunksize=1)
    else:
        for task in tasks:
            results.append(process_single_image(task))
            
    # Filter out None values
    valid_uncertainties = [x for x in results if x is not None]
    
    # Print statistics for the current batch
    if valid_uncertainties:
        batch_avg = sum(valid_uncertainties) / len(valid_uncertainties)
        print(f"📊 Batch Average Uncertainty: {batch_avg:.4f} (Count: {len(valid_uncertainties)})")
    
    return valid_uncertainties

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", type=str, default="beam", choices=["direct", "beam", "v_labs", "v-bas"])
    parser.add_argument("--data_root", type=str, default="dataset/jigsaw")
    
    parser.add_argument("--beam_width", type=int, default=3, help="Beam width for search")
    parser.add_argument("--max_depth", type=int, default=4, help="Max search depth")
    parser.add_argument("--model_name", type=str, default="qwen_8b_instruct")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--entropy_skip_threshold", type=float, default=-1.0,
                       help="Skip observer when prior entropy < this; -1=disabled")

    args = parser.parse_args()
    
    target_resolutions = [3, 4, 5]
    
    # Global collection list for uncertainties
    global_uncertainties = []

    for res_num in target_resolutions:
        res_folder = f"res{res_num}"
        res_path = os.path.join(args.data_root, res_folder)
        
        if not os.path.exists(res_path):
            print(f"Skipping {res_path} (Not found)")
            continue
            
        all_subs = [d for d in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, d))]
        sample_folders = sorted([d for d in all_subs if d.startswith('s')])
        
        if not sample_folders:
            print(f"No sample folders (s*) found in {res_path}")
            continue

        for sample_folder in sample_folders:
            args.res = res_num
            
            data_dir = os.path.join(res_path, sample_folder, "images")
            
            output_dir = os.path.join(
                "outputs",
                f"{args.model_name}/jigsaw_{args.type}/{res_folder}/{sample_folder}/"
            )
            
            # Collect return values
            batch_vals = process_images(data_dir, output_dir, args)
            if batch_vals:
                global_uncertainties.extend(batch_vals)
    
    print("\n" + "="*60)
    print("✅ All tasks completed.")
    
    # Compute and print global average
    if global_uncertainties:
        global_avg = sum(global_uncertainties) / len(global_uncertainties)
        print(f"📈 Global Average First-Step Uncertainty: {global_avg:.4f}")
        print(f"   (Calculated over {len(global_uncertainties)} newly processed samples)")
    else:
        print("📈 Global Average First-Step Uncertainty: N/A (No data collected)")
    print("="*60)