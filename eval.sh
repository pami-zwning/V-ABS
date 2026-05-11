#!/bin/bash
# V-ABS Full Evaluation Suite
# Runs all tasks (visual search, navigation, sudoku, jigsaw) across models

cd "$(dirname "$0")"

echo "========================================================"
echo "Starting visual search tasks: $(date)"
echo "========================================================"

echo "Running: model=qwen_8b_instruct"

echo "  beam_hr_bench_4k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type beam --model_name qwen_8b_instruct

echo "  beam_hr_bench_8k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type beam --model_name qwen_8b_instruct

echo "  beam_vstar..."
python -u unified_visual_search/run.py --dataset_type vstar --type beam --model_name qwen_8b_instruct

echo "  direct_hr_bench_4k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type direct --model_name qwen_8b_instruct

echo "  direct_hr_bench_8k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type direct --model_name qwen_8b_instruct

echo "  direct_vstar..."
python -u unified_visual_search/run.py --dataset_type vstar --type direct --model_name qwen_8b_instruct


echo "Running: model=qwen25_vl_7b"

echo "  beam_hr_bench_4k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type beam --model_name qwen25_vl_7b

echo "  beam_hr_bench_8k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type beam --model_name qwen25_vl_7b

echo "  beam_vstar..."
python -u unified_visual_search/run.py --dataset_type vstar --type beam --model_name qwen25_vl_7b

echo "  direct_hr_bench_4k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type direct --model_name qwen25_vl_7b

echo "  direct_hr_bench_8k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type direct --model_name qwen25_vl_7b

echo "  direct_vstar..."
python -u unified_visual_search/run.py --dataset_type vstar --type direct --model_name qwen25_vl_7b


echo "========================================================"
echo "Starting navigation tasks: $(date)"
echo "========================================================"

echo "Running: model=qwen_8b_instruct"

echo "  beam_maze..."
python -u unified_navigation/run.py --task_mode maze --type beam --model_name qwen_8b_instruct

echo "  beam_navigation..."
python -u unified_navigation/run.py --task_mode navigation --type beam --model_name qwen_8b_instruct

echo "  direct_maze..."
python -u unified_navigation/run.py --task_mode maze --type direct --model_name qwen_8b_instruct

echo "  direct_navigation..."
python -u unified_navigation/run.py --task_mode navigation --type direct --model_name qwen_8b_instruct

echo "Running: model=qwen25_vl_7b"

echo "  beam_maze..."
python -u unified_navigation/run.py --task_mode maze --type beam --model_name qwen25_vl_7b

echo "  beam_navigation..."
python -u unified_navigation/run.py --task_mode navigation --type beam --model_name qwen25_vl_7b

echo "  direct_maze..."
python -u unified_navigation/run.py --task_mode maze --type direct --model_name qwen25_vl_7b

echo "  direct_navigation..."
python -u unified_navigation/run.py --task_mode navigation --type direct --model_name qwen25_vl_7b


echo "========================================================"
echo "Starting sudoku tasks: $(date)"
echo "========================================================"

echo "Running: model=qwen_8b_instruct"
python -u unified_sudoku/run.py --type beam --model_name qwen_8b_instruct
python -u unified_sudoku/run.py --type direct --model_name qwen_8b_instruct

echo "Running: model=qwen25_vl_7b"
python -u unified_sudoku/run.py --type beam --model_name qwen25_vl_7b
python -u unified_sudoku/run.py --type direct --model_name qwen25_vl_7b


echo "========================================================"
echo "Starting jigsaw tasks: $(date)"
echo "========================================================"

echo "Running: model=qwen_8b_instruct"
python -u unified_jigsaw/run.py --type beam --model_name qwen_8b_instruct
python -u unified_jigsaw/run.py --type direct --model_name qwen_8b_instruct

echo "Running: model=qwen25_vl_7b"
python -u unified_jigsaw/run.py --type beam --model_name qwen25_vl_7b
python -u unified_jigsaw/run.py --type direct --model_name qwen25_vl_7b


echo "========================================================"
echo "Running evaluation scripts: $(date)"
echo "========================================================"

for MODEL in qwen_8b_instruct qwen25_vl_7b; do
    echo "--- Evaluating ${MODEL} ---"

    # Visual search
    for DATASET in hr_bench_4k hr_bench_8k vstar; do
        python -u scripts/eval_visual_search.py --type beam --model_name ${MODEL} --dataset_type ${DATASET}
    done

    # Navigation
    for TASK in maze nav; do
        python -u scripts/eval_navigation.py --task ${TASK} --type beam --model_name ${MODEL}
    done

    # Sudoku
    python -u scripts/eval_sudoku.py --type beam --model_name ${MODEL}

    # Jigsaw
    python -u scripts/eval_jigsaw.py --type beam --model_name ${MODEL}
done

echo "========================================================"
echo "All tasks complete: $(date)"
echo "========================================================"
