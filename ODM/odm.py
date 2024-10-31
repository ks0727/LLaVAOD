from ODM.llava import LLaVA
from ODM.grounding_dino import GroundingDino
from ODM.visual_model import VisualModel
from ODM.utils import Template_prompt

from PIL import Image
import numpy as np

class ODM:
    def __init__(self,mllm_path:str,od_model_path:str,od_weight_path:str=None,od_bbox_threshold:float=0.35,od_text_threshold:float=0.25,max_num_objects:int=20):
        self.mllm_path = mllm_path
        self.od_model_path = od_model_path

        self.mllm = LLaVA(model_path = self.mllm_path)
        self.mllm.model.model

        self.visual_encoder = VisualModel(self.mllm.image_processor,self.mllm.model.model.get_vision_tower().vision_tower)
        self.max_num_objects = max_num_objects
        self.object_detector =GroundingDino(
            model_path=self.od_model_path,
            weight_path=od_weight_path,
            bbox_threshold=od_bbox_threshold,
            text_threshold=od_text_threshold,
            max_num_objects = self.max_num_objects
        )


    def forward_model(self,image,text,state='answer'):
        prompt_template = Template_prompt(state).template_prompt
        prompt = prompt_template.format(text)



        response = self.mllm.generate(pil_image,prompt)
        response = response.strip()

        print("Response : ",response)
        return response


    def object_extraction(self,image,object_list):
        """
        Return
        object_bboxes (list): [{label: the name the object,bbox : bounding box}]
        """
        object_bboxes = []

        object_bboxes = self.object_detector(image,object_list)

        return object_bboxes


    def run(self,image_path,question):
        image = Image.open(image_path)
        image_array = np.array(image)
        object_list = self.forward_model(image,question,state='object_extraction')

        object_bboxes = self.object_extraction(self,image,object_list)

        prompt = ""

        for object_label,bbox in object_bboxes:
            left,upper,right,lower = bbox
            sub_image = image.crop((left,upper,right,lower))
            caption = self.forward_model(self,sub_image,question,state='caption')
            print("Caption : ",caption)
            prompt += caption

        answer = self.mllm.generate(image_path,prompt)
        return answer








