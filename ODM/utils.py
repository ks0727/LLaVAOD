class Template_prompt:
    def __init__(self,state:str):
        self.template_prompt = None
        if state == "object_extraction":
            self.template_prompt = """You will be given a question. Please list
            all of the objects that appear in the given question.
            Example)
            Question : "What is the material of the glove?
                        (A) rubber
                        (B) cotton
                        (C) kevlar
                        (D) leather
                        Answer with the option's letter from the given choices directly.
                        "
            Output:['the material of the glove','the glove','rubber','cotton','kevlar','leather']
            Input)
            Question :"{}"
            Output:
            """

        elif state == "answer":
            self.template_prompt = """You will be given visual information that consists of object label and object bounding boxes in JSON format. If the information is not helpfull, you don't have to use it. By utilizing the given information, answer the question.
            Visual information : {},
            Quesstion :  """

        elif state == "empty":
            self.template_prompt = "{}"
        
        else:
            print("Unknow State, please double check your state input.")

