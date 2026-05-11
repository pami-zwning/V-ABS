#!/bin/bash

# 获取开始时间
echo "========================================================"
echo "🚀 vis_search任务开始: $(date)"
echo "========================================================"

# echo "▶️  正在运行 ------model_name qwen3_8b_instruct-----------"

# echo "▶️  正在运行: beam_hr_bench_4k..."
# python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type beam --model_name qwen_8b_instruct

# echo "▶️ 正在运行: beam_hr_bench_8k..."
# python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type beam --model_name qwen_8b_instruct

# echo "▶️ 正在运行: beam_hr_vstar..."
# python -u unified_visual_search/run.py --dataset_type vstar --type beam --model_name qwen_8b_instruct

# echo "▶️ 正在运行: direct_hr_bench_4k..."
# python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type direct --model_name qwen_8b_instruct

# echo "▶️ 正在运行: direct_hr_bench_8k..."
# python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type direct --model_name qwen_8b_instruct

# echo "▶️ 正在运行: direct_vstar..."
# python -u unified_visual_search/run.py --dataset_type vstar --type direct --model_name qwen_8b_instruct


echo "▶️  正在运行 ------model_name internvl3_8b-----------"

echo "▶️  正在运行: beam_hr_bench_4k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type beam --model_name internvl3_8b

echo "▶️ 正在运行: beam_hr_bench_8k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type beam --model_name internvl3_8b

echo "▶️ 正在运行: beam_hr_vstar..."
python -u unified_visual_search/run.py --dataset_type vstar --type beam --model_name internvl3_8b

echo "▶️ 正在运行: direct_hr_bench_4k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_4k --type direct --model_name internvl3_8b

echo "▶️ 正在运行: direct_hr_bench_8k..."
python -u unified_visual_search/run.py --dataset_type hr_bench_8k --type direct --model_name internvl3_8b

echo "▶️ 正在运行: direct_vstar..."
python -u unified_visual_search/run.py --dataset_type vstar --type direct --model_name internvl3_8b


# 获取开始时间
echo "========================================================"
echo "🚀 navigation任务开始: $(date)"
echo "========================================================"

# echo "▶️  正在运行 ------model_name qwen3_8b_instruct-----------"

# echo "▶️  正在运行: beam_maze..."
# python -u unified_navigation/run.py --task_mode maze --type beam --model_name qwen_8b_instruct

# echo "▶️ 正在运行: beam_navigation..."
# python -u unified_navigation/run.py --task_mode navigation --type beam --model_name qwen_8b_instruct

# echo "▶️  正在运行: dir_maze..."
# python -u unified_navigation/run.py --task_mode maze --type direct --model_name qwen_8b_instruct

# echo "▶️ 正在运行: dir_navigation..."
# python -u unified_navigation/run.py --task_mode navigation --type direct --model_name qwen_8b_instruct

echo "▶️  正在运行 ------model_name internvl3_8b-----------"

echo "▶️  正在运行: beam_maze..."
python -u unified_navigation/run.py --task_mode maze --type beam --model_name internvl3_8b

echo "▶️ 正在运行: beam_navigation..."
python -u unified_navigation/run.py --task_mode navigation --type beam --model_name internvl3_8b

echo "▶️  正在运行: dir_maze..."
python -u unified_navigation/run.py --task_mode maze --type direct --model_name internvl3_8b

echo "▶️ 正在运行: dir_navigation..."
python -u unified_navigation/run.py --task_mode navigation --type direct --model_name internvl3_8b

echo "========================================================"
echo "🎉 所有任务已结束: $(date)"
echo "========================================================"

# nohup ./srun.sh > qwen_uni_script.log 2>&1 &