from contract_parser import Parser
import os
import numpy as np
import pickle

class Resolver():
    def __init__(self, path):
        self.parser = Parser(path)
        self.parser.parse_contract_to_functions()
        self.parser.get_semantic_vectors()
        self.parser.parse_slither_to_functions()
        self.all_functions = []
        self.all_vectors = []
        self.all_tests = []
        self.n_best = 3
        self.similar_functions_matrix = []
        self.test_matrix = []

    def read_functions(self):
        for dirpath, _, filenames,  in os.walk(self.parser.output_dir):
            if self.parser.contract_name in dirpath.split('\\'):
                continue
            if dirpath.endswith('functions') or dirpath.endswith('semantic_vectors') or dirpath.endswith('tests'):
                self.read_file(dirpath, filenames)

        
    def read_file(self, dirpath, filenames):
        for filename in filenames:
            
            if dirpath.endswith('functions'):
                with open(os.path.join(dirpath, filename), 'r') as f:
                    self.all_functions.append(f.read())
            elif dirpath.endswith('semantic_vectors'):
                with open(os.path.join(dirpath, filename), 'rb') as f:
                    self.all_vectors.append(pickle.load(f))
            elif dirpath.endswith('tests'):
                with open(os.path.join(dirpath, filename), 'r') as f:
                    self.all_tests.append(f.read())

    def similar_functions_and_tests(self):
        for vector in self.parser.semantic_vectors:
            best_functions, best_tests = self.one_vector_similar_functions_and_tests(vector)
            self.similar_functions_matrix.append(best_functions)
            self.test_matrix.append(best_tests)
            
    def one_vector_similar_functions_and_tests(self, vector):
        vals = zip(self.all_functions, self.all_vectors, self.all_tests)
        vals = sorted(vals, key = lambda x: self.vectors_distance(x[1], vector))
        functions = [x[0] for x in vals]
        tests = [x[2] for x in vals]

        best_functions = functions[:self.n_best]
        best_tests = tests[:self.n_best]
        return best_functions, best_tests

    def vectors_distance(self, v1, v2):
        return np.mean(np.square(v1 - v2))
    
    def show_similar_functions(self):
        for i, orig_func in enumerate(self.parser.functions):
            orig_func = orig_func[0]
            print("Original function:\n")
            print(orig_func)
            for similar_func in self.similar_functions_matrix[i]:
                print("Similar function:\n")
                print(similar_func)


# r = Resolver('example.sol')
# r.read_functions()
# r.similar_functions_and_tests()
# r.show_similar_functions()
# print(r.similar_functions_matrix)