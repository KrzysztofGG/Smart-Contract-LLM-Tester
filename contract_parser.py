from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import json
import subprocess
import logging

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

        for line in self.contract:
            if 'function' in line:
                
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

                    self.functions.append(curr_fun)
                    curr_fun = ''

                    bracket_balance = None
                    reading_function = False

    def get_semantic_vectors(self):
        logging.info(f'Creating semantic vectors for contract {self.contract_name}')
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")

        for input_text in self.functions:
            self.get_vector_from_input(tokenizer, model, input_text)

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

        for i, (fun, vec) in enumerate(zip(self.functions, self.semantic_vectors)):
            
            with open(os.path.join(self.output_dir, self.contract_name, 'functions', f'{i}.txt'), 'w+')  as f:
                f.write(fun)

            with open(os.path.join(self.output_dir, self.contract_name, 'semantic_vectors', f'{i}.txt'), 'w+') as f:
                json.dump(vec.tolist(), f)
        
    
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



# parser = Parser('example2.sol')
# parser.parse_contract_to_functions()
# parser.get_semantic_vectors()
# parser.save_functions_and_vectors()