from contract_parser import Parser
import os
import numpy as np
import pickle

class Resolver():
    def __init__(self, path):
        self.parser = Parser(path)
        self.parser.parse_contract_to_functions()
        self.parser.get_semantic_vectors()
        self.all_functions = []
        self.all_vectors = []
        self.n_best = 3
        self.similar_functions_matrix = []

    def read_functions(self):
        for dirpath, _, filenames,  in os.walk(self.parser.output_dir):
            if self.parser.contract_name in dirpath.split('\\'):
                continue

            if dirpath.endswith('functions') or dirpath.endswith('semantic_vectors'):
                self.read_file(dirpath, filenames)

        
    def read_file(self, dirpath, filenames):
        for filename in filenames:
            if dirpath.endswith('functions'):
                with open(os.path.join(dirpath, filename), 'r') as f:
                    self.all_functions.append(f.read())
            elif dirpath.endswith('semantic_vectors'):
                with open(os.path.join(dirpath, filename), 'rb') as f:
                    self.all_vectors.append(pickle.load(f))

    def similar_functions(self):
        for  func, vector in zip(self.parser.functions, self.parser.semantic_vectors):
            best_functions = self.one_vector_similar_functions(vector)
            func_row = [func]
            func_row.extend(best_functions)
            self.similar_functions_matrix.append(func_row)

    def one_vector_similar_functions(self, vector):
        vals = zip(self.all_functions, self.all_vectors)
        vals = sorted(vals, key = lambda x: self.vectors_distance(x[1], vector))
        functions = [x[0] for x in vals]

        best_functions = functions[:self.n_best]
        return best_functions

    def vectors_distance(self, v1, v2):
        return np.mean(np.square(v1 - v2))
    
    def show_similar_functions(self):
        for row in r.similar_functions_matrix:
            for i, val in enumerate(row):
                if i == 0:
                    print("Original function:\n")
                else:
                    print("Similar function:\n")
                print(val)

r = Resolver('example.sol')
r.read_functions()
r.similar_functions()
r.show_similar_functions()