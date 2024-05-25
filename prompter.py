from resolver import Resolver
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import torch

class Prompter():
    def __init__(self, path):
        self.resolver = Resolver(path)
        self.resolver.read_functions()
        self.resolver.similar_functions_and_tests()
        # self.resolver.show_similar_functions()
        # print(self.resolver.similar_functions_matrix)
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = TFGPT2LMHeadModel.from_pretrained(self.model_name)
        self.model_answears = []
        with open('prompt.txt', 'r') as f:
            self.prompt_start = f.read()
    
    def prompt_all_functions(self):
        for i, func in enumerate(self.resolver.parser.functions):

            if i >= len(self.resolver.similar_functions_matrix) or i >= len(self.resolver.test_matrix):
                break

            similar_functions = self.resolver.similar_functions_matrix[i]
            similar_tests = self.resolver.test_matrix[i]
            prompt = self.get_prompt_one_function(func, similar_functions, similar_tests)
            # print(prompt)
            self.ask_model(prompt)


    def get_prompt_one_function(self, function, similar_functions, similar_tests):
        
        prompt = self.prompt_start
        # print(self.resolver.similar_functions_matrix)
        prompt += f'<test_function>{function}</test_function>\n'
        # print(similar_functions+similar_tests)
        for func, test in zip(similar_functions, similar_tests):
            prompt += f'<function>{func}</function>\n'
            prompt += f'<vulnerability>{test}</vulnerability>\n'
        
        return prompt
    
    def ask_model(self, prompt):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids])

        output = self.model.generate(input_ids, max_length=200, num_return_sequences=1,attention_mask=0) #TODO: dunno about these parameters
        output_decoded = self.tokenizer.decode(output[0],  skip_special_tokens=True)
        self.model_answears.append(output_decoded)
        print(output_decoded)



prompter = Prompter('example2.sol')
prompter.prompt_all_functions()