from contract_parser import Parser
import os

class Resolver():
    def __init__(self, path):
        self.parser = Parser(path)
        self.parser.parse_contract_to_functions()
        self.parser.get_semantic_vectors()
        self.all_functions = []
        self.all_vectors = []

    def find_similar_functions(self):
        for dirpath, dirnames, filenames,  in os.walk(self.parser.output_dir):
            if self.parser.contract_name in dirpath.split('\\'):
                continue

            if dirpath.endswith('functions') or dirpath.endswith('semantic_vectors'):
                self.read_file(dirpath, filenames)
        

    def read_file(self, dirpath, filenames):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), 'r') as f:
                if dirpath.endswith('functions'):
                    self.all_functions.append(f.read())
                elif dirpath.endswith('semantic_vectors'):
                    self.all_vectors.append(f.read())

        



r = Resolver('example.sol')
r.find_similar_functions()