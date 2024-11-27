from ODM.llava import LLaVA
from ODM.grounding_dino import GroundingDino
from ODM.visual_model import VisualModel
from ODM.utils import Template_prompt

import torch
from PIL import Image
import numpy as np
import json

class ODM_Config():
    def __init__(self,config_path):
        with open(config_path,"r") as f:
            config = json.load(f)

        self.mllm_path = config.get("mllm_path","liuhaotian/llava-v1.5-7b")
        self.od_model_path = config.get("od_model_path","/po4/ksakai/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.od_weight_path = config.get("od_weight_path","/po4/ksakai/models/GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.od_bbox_threshold = config.get("od_bbox_threshold",0.35)
        self.od_text_threshold = config.get("od_text_threshold",0.25)
        self.max_num_objects = config.get("max_num_objects",30)
        self.add_description = config.get("add_description",False)
        self.result_path = config.get("result_path","/po4/ksakai/src/LLaVAOD/result")

class ODM:
    def __init__(self,config:ODM_Config):
        self.mllm_path = config.mllm_path
        self.od_model_path = config.od_model_path
        self.od_weight_path = config.od_weight_path
        self.od_bbox_threshold = config.od_bbox_threshold
        self.od_text_threshold = config.od_text_threshold
        self.max_num_objects = config.max_num_objects
        self.add_description = config.add_description
        self.result_path = config.result_path
        
        self.mllm = LLaVA(model_path = self.mllm_path)
        self.mllm.model.model
        self.object_detector = GroundingDino(self.od_model_path,self.od_weight_path,self.od_bbox_threshold,self.od_text_threshold,self.max_num_objects)
        self.visual_encoder = VisualModel(self.mllm.image_processor,self.mllm.model.model.get_vision_tower().vision_tower)
        
    def forward_model(self,image,text,state='answer'):
        prompt_template = Template_prompt(state).template_prompt
        prompt = prompt_template.format(text)
        response = self.mllm.generate(image,prompt)
        response = response.strip()

        return response

    def object_extraction(self,image_path,object_list,add_descrpiption:bool):
        """
        Args:
            add_description(bool) : whether or not to get bouding boxes with description
        """
        """
        Return
        object_bboxes (list): [{label: the name the object,bbox : bounding box}]
        """
        
        object_bboxes = []
        object_bboxes = self.object_detector.get_bboxes(image_path,object_list,add_descrpiption)
        return object_bboxes

    def read_object_list(self,object_list:str):
        object_list = object_list.strip('[]').split(',')
        object_list = [item.strip() for item in object_list]
        return object_list

    def get_refined_prompt(self,image_path,question):
        object_list = self.forward_model(image_path,question,state='object_extraction')
        if type(object_list) == str:
            object_list = self.read_object_list(object_list)
        object_bboxes = self.object_extraction(image_path,object_list,self.add_description)
        """
        for object_label,bbox in object_bboxes:
            left,upper,right,lower = bbox
            image = Image.open(image_path)
            sub_image = image.crop((left,upper,right,lower))
            caption = self.forward_model(self,sub_image,question,state='caption')
            print("Caption : ",caption)
            prompt += caption
        """
        if self.add_description:
            prompt_template = Template_prompt(state="answer").template_prompt
            object_bboxes = self.get_desctiption(image_path=image_path,object_bboxes=object_bboxes,is_original_size=False,is_long=False)
            prompt = prompt_template.format(object_bboxes)
        else:
            prompt_template = Template_prompt(state='empty').template_prompt
            prompt = prompt_template.format(object_bboxes)

        return prompt,object_bboxes
    
    def get_desctiption(self,image_path:str,object_bboxes:dict,is_original_size:bool=False,is_long:bool=False):
        """
        Args:
            image_path (str): the path of the image to be described by the model
            object_bboxes (dict {"label" : LABEL_OF_OBJECT, "bounding_boxes : [(center_x,center_y,width,height)]"})  
        """
        image_pil = Image.open(image_path)
        image_width,image_height = image_pil.size
        output = []
        
        if type(object_bboxes) is not list:
            object_bboxes = [object_bboxes]

        for object_bbox in object_bboxes:
            description = ""
            if object_bbox is None:
                continue
            label = object_bbox["label"]
            bboxes = object_bbox["bounding_boxes"]
            description_prompt = "Please describe this image."

            if is_long:
                for bbox_i in bboxes:
                    cropped_image = image_pil.crop(bbox_i)
                    description_i = self.mllm.generate(cropped_image,description_prompt)
                    description += description_i
            else:
                if len(bboxes) == 0:
                    continue
                bbox = bboxes[0] if type(bboxes) is list else bboxes
                cropped_image = image_pil.crop(bbox)
                description_i = self.mllm.generate(cropped_image,description_prompt)
                description += description_i
            
            if not is_original_size:
                bboxes = self.normalize_boxes(bboxes,image_width,image_height)

            output.append(dict(label=label,bounding_box=bboxes,description=description))
        
        return output

    def get_bounding_box_on_original_image(self,normalized_bounding_box,original_height:int,original_width:int):
        """
        (cx,cy,w,h) -> (top_left_x,top_left_y,bottom_right_x,bottom_right_y)
        """
        if isinstance(normalized_bounding_box,torch.Tensor):
            normalized_bounding_box = normalized_bounding_box.tolist()
            #normalized_bounding_box = normalized_bounding_box.to('cpu').detach().numpy().copy()

        if type(normalized_bounding_box) is not list:
            normalized_bounding_box = [normalized_bounding_box]
        
        original_bboxes = []
        for box in normalized_bounding_box:
            center_x,center_y,width,height = box

            box_width = width*original_width
            box_height = height*original_height

            x_min = (center_x*original_width) - (box_width/2)
            y_min = (center_y*original_height) - (box_height/2)
            
            x_max = (center_x*original_width) + (box_width/2)
            y_max = (center_y*original_height) + (box_height/2)
            
            original_bboxes.append((x_min,y_min,x_max,y_max))
        
        return original_bboxes
    
    
    def normalize_boxes(self,bounding_boxes, image_width, image_height):
        normalized_boxes = []
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box
            # Calculate center, width, and height
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            # Normalize
            center_x /= image_width
            center_y /= image_height
            width /= image_width
            height /= image_height
            # Append normalized box
            normalized_boxes.append((center_x, center_y, width, height))
        return normalized_boxes










