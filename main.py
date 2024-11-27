import argparse
from ODM.odm import ODM,ODM_Config
import os
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mllm_path',type=str,default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--bench_path',type=str,default="/po4/ksakai/datasets/vstar_bench")
    parser.add_argument('--od_model_path',type=str,default="/po4/ksakai/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--od_weight_path',type=str,default="/po4/ksakai/models/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    parser.add_argument('--od_bbox_threshold',type=float,default=0.35)
    parser.add_argument('--od_text_threshold',type=float,default=0.25)
    parser.add_argument('--config_path',type=str,default="/po4/ksakai/src/LLaVAOD/config/llava_grounding_dino.json")
    parser.add_argument('--result_path',type=str,default="/po4/ksakai/src/LLaVAOD/result")
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
    config_path = args.config_path


    config = ODM_Config(config_path)
    model = ODM(config)
    jsonl_path = os.path.join(bench_path,"test_questions.jsonl")
    with open(jsonl_path,"r") as f:
        jsonl_data = [json.loads(l) for l in f.readlines()]
    cates = ['direct_attributes','relative_position']
    result_dir_path = args.result_path
    result_json_file = "result.json"
    output_to_save = []

    if result_dir_path is not None:
        result_json_file = os.path.join(result_dir_path,result_json_file)

    dic = {}
    for i,data in tqdm(enumerate(jsonl_data),total=len(jsonl_data)):
        image_path = os.path.join(bench_path,data['image'])
        query = data['text']
        odm_prompt,object_list = model.get_refined_prompt(image_path=image_path,question=query)
        try:
            dic[data["image"]] = odm_prompt
        except:
            print("couldn't load the model output properly\n")
            dic[data["image"]] = ""
    try:
        with open("result.json","w") as f:
            json.dump(dic,f,indent=2)
    except:
        with open("result.txt","w") as f:
            f.write(str(dic))
    else:
        print(output_to_save)



