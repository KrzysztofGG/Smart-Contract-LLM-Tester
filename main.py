from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class Parser():

    def __init__(self, path):
        
        with open(path, 'r') as f:
            self.contract = f.readlines()

        self.functions = []
        self.semantic_vectors = []

    def parse_contract_to_functions(self):

        curr_fun = ''
        reading_function = False
        bracket_balance = None

        for line in self.contract:
            if 'function' in line:
                curr_fun += line
                bracket_balance = 1
                reading_function = True
            elif reading_function:
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
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")

        for input_text in self.functions:
            self.get_vectors_from_input(tokenizer, model, input_text)

        self.semantic_vectors_whitening()

    def get_vector_from_input(self, tokenizer, model, input_text):
        input_ids = tokenizer.encode(input_text, return_tensors="pt", 
                                     max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(input_ids)
            semantic_vector = outputs.last_hidden_state.mean(dim=1) 
            self.semantic_vectors.append(semantic_vector)

    def semantic_vectors_whitening(self):

        vectors = np.asarray(self.semantic_vectors)
        covariance_matrix = np.cov(vectors, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        whitening_matrix = np.dot(np.dot(eigenvectors, np.diag(1.0 / np.sqrt(eigenvalues + 1e-5))), eigenvectors.T)

        whitened_vectors = np.dot(vectors, whitening_matrix)
        mean = np.mean(whitened_vectors, axis=0)
        std = np.std(whitened_vectors, axis=0)

        normalized_whitened_vectors = (whitened_vectors - mean) / std
        self.semantic_vectors = normalized_whitened_vectors



parser = Parser('example.sol')
parser.parse_contract_to_functions()