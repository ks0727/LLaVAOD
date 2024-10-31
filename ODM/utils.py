class Template_prompt:
    def __init__(self,state:str):
        self.template_prompt = None
        if state == "object_extraction":
            self.template_prompt = """You will be given a question. Name every
            single object that appears in the question. If the object is
            modified with any words, make sure to include them too.
            Example:
            Question:"Is a red umbrella on the left side of the white table?"
            Output:['a red unbrealla','the white table']
            Input:
            Question : {}
            Output:
            """

        elif state == "answer":
            self.template = "{}"

        else:
            print("Unknow State, please double check your state input.")

