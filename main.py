import argparse
from ODM.odm import ODM
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path',type=str,default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--bench_path',type=str,default="/po4/ksakai/datasets/vstar_bench")
    parser.add_argument('--od_model_path',type=str,default="/po4/ksakai/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--od_weight_path',type=str,default="/po4/ksakai/models/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    parser.add_argument('--od_bbox_threshold',type=float,default=0.35)
    parser.add_argument('--od_text_threshold',type=float,default=0.25)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    mllm_path = args.mllm_path
    od_model_path = args.od_model_path
    od_weight_path = args.od_weight_path
    od_bbox_threshold = args.od_bbox_threshold
    od_text_threshold = args.od_text_threshold

    bench_path = args.bench_path

    model = ODM(mllm_path,od_model_path,od_weight_path,od_bbox_threshold,od_text_threshold)

    jsonl_path = os.path.join(bench_path,"test_questions.jsonl")
    with open(jsonl_path,"r") as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]
    cates = ['direct_attributes','relative_position']

    for data in jsonl_data:
        image_path = os.path.join(bench_path,data['image'])
        query = data['text']
        output = model.run(image_path=image_path,question=query)
        print(f"Ground Truth : {data['label']}, Model Output: {output}")








