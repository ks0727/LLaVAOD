python vstar_bench_eval.py \
--vqa-model-path="liuhaotian/llava-v1.5-7b" \
--vqa-model-base=None \
--conv_type="v1" \
--benchmark-folder="/po4/ksakai/datasets/vstar_bench" \
--vsm-model-path="craigwu/seal_vsm_7b" \
--output-path="eval_result.json" \
--minimum_size_scale=4.0 \
--minimum_size=224 \

