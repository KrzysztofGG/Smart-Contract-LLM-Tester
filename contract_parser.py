from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import json
import subprocess
import logging
import pickle
import re
logging.getLogger().setLevel(logging.INFO)

class Parser():

    def __init__(self, path):
        
        with open(path, 'r') as f:
            self.contract = f.readlines()

        self.path=path
        self.functions = []
        self.semantic_vectors = []
        self.output_dir = 'parsed_contracts'
        self.contract_name = path[:-4]
        self.slither_output=""
        self.echidna_output=""

    def parse_contract_to_functions(self):
        logging.info(f'Parsing contract {self.contract_name} to functions')
        curr_fun = ''
        reading_function = False
        reading_header = False
        bracket_balance = None
        first_line=0
        end_line=0
        for index,line in enumerate(self.contract):
            if 'function' in line:
                first_line=index
                if line.strip()[-1] == ';': #skips functions that are only declared, not defined (interface)
                    continue

                curr_fun += line

                if line.strip()[-1] == '{':
                    bracket_balance = 1
                else:
                    reading_header = True
                    bracket_balance = 0
                reading_function = True

            elif reading_function:

                if reading_header:
                    if ')' in line:
                        reading_header = False
                    else:
                        curr_fun += line
                        continue

                left_bracket = line.count('{')
                right_bracket = line.count('}')

                bracket_balance += left_bracket
                bracket_balance -= right_bracket

                curr_fun += line
                if bracket_balance == 0:
                    end_line=index
                    self.functions.append([curr_fun,first_line,end_line,''])
                    curr_fun = ''

                    bracket_balance = None
                    reading_function = False

    def get_semantic_vectors(self):
        logging.info(f'Creating semantic vectors for contract {self.contract_name}')
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")

        for input_text in self.functions:
            self.get_vector_from_input(tokenizer, model, input_text[0])

        self.semantic_vectors_whitening()
        # self.save_functions_and_vectors()

    def get_vector_from_input(self, tokenizer, model, input_text):
        input_ids = tokenizer.encode(input_text, return_tensors="pt", 
                                     max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(input_ids)
            semantic_vector = outputs.last_hidden_state.mean(dim=1) 
            self.semantic_vectors.append(semantic_vector.squeeze().numpy())

    def semantic_vectors_whitening(self):
        logging.info(f'Whitening semantic vectors for contract {self.contract_name}')
        vectors = np.asarray(self.semantic_vectors)
        covariance_matrix = np.cov(vectors, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        whitening_matrix = np.dot(np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues + 1e-5))), eigenvectors.T)

        whitened_vectors = np.dot(vectors, whitening_matrix)
        mean = np.mean(whitened_vectors, axis=0)
        std = np.std(whitened_vectors, axis=0)

        normalized_whitened_vectors = (whitened_vectors - mean) / std
        self.semantic_vectors = normalized_whitened_vectors

    def get_slither_tests(self):
        result = subprocess.run(['slither', self.path], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        output=result.stderr.decode('cp1252')
        self.slither_output=output
    
    def get_echidna_tests(self):
        result = subprocess.run(['echidna',self.path], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        output=result.stderr.decode()
        self.echidna_output=output

    def save_tests(self):  
        self.prepare_dir('test_outputs')
        
        with open(os.path.join(self.output_dir, self.contract_name, 'test_outputs', 'slither.txt'), 'w+')  as f:
            f.write(self.slither_output)
        with open(os.path.join(self.output_dir, self.contract_name, 'test_outputs', 'echidna.txt'), 'w+')  as f:
            f.write(self.echidna_output)

    def save_functions_and_vectors(self):
        logging.info(f'Saving functions and vectors for contract {self.contract_name}')
        self.prepare_dir('functions')
        self.prepare_dir('semantic_vectors')
        self.prepare_dir('tests')

        for i, (fun, vec) in enumerate(zip(self.functions, self.semantic_vectors)):
            
            with open(os.path.join(self.output_dir, self.contract_name, 'functions', f'{i}.txt'), 'w+')  as f:
                f.write(fun[0])

            with open(os.path.join(self.output_dir, self.contract_name, 'semantic_vectors', f'{i}.pkl'), 'wb+') as f:
                pickle.dump(vec, f)
            if fun[3]!='':
                with open(os.path.join(self.output_dir, self.contract_name, 'tests', f'{i}.txt'), 'w+')  as f:
                    f.write(fun[3])
    def parse_slither_to_functions(self):
        self.get_slither_tests()
        self.save_tests()
        found_lines = ''
        inside_block=False
        mean=0


        with open(os.path.join(self.output_dir, self.contract_name, 'test_outputs', 'slither.txt'), 'r+')  as file:
            for line in file:
                function_match = re.match(r'.*\((' + re.escape(self.contract_name+".sol") + r'#.*?)\)', line)

                if function_match and not inside_block:
                    inside_block=True
                    file_reference = function_match.group(1)
                    
                    
                    if '-' in file_reference.split('#')[1]:
                        numbers = re.findall(r'\d+', file_reference.split('#')[1])
                        num1 = int(numbers[0])
                        num2 = int(numbers[1])
                        mean = (num1 + num2) / 2
                        
                        
                    else:
                        mean = int(re.findall(r'\d+', file_reference.split('#')[1])[0])     
                        
                    found_lines+=line
                    
                elif inside_block and  line.startswith("   "):
                    found_lines+=line      
                elif inside_block and not  line.startswith("   "):
                    inside_block=False 
                    for function in self.functions:
                        start=function[1]
                        end=function[2]
                        if  start <= mean <=end:
                            function[3]+=found_lines
                            # print(function)
                    found_lines=''
                    mean=0     
    
    def prepare_dir(self, dir_name):
        if not os.path.exists(os.path.join(self.output_dir, self.contract_name, dir_name)):
            os.makedirs(os.path.join(self.output_dir, self.contract_name, dir_name))
        else:
            for filename in os.listdir(os.path.join(self.output_dir, self.contract_name, dir_name)):
                file_path = os.path.join(dir_name, filename)
                try: 
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")



# parser = Parser('example.sol')
# parser.parse_contract_to_functions()
# parser.get_semantic_vectors()
# parser.parse_slither_to_functions()
# parser.save_functions_and_vectors()