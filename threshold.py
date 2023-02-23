

import numpy as np
from copy import deepcopy
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from trustee.utils.tree import get_dt_dict
from graphviz import Digraph
import webbrowser
import os

class thresholder:
    
    def __init__(self, trust_report):
        self.trust_report = trust_report
        self.paths = trust_report.max_dt_all_branches
        
    def find_paths_to_class(self, target_class):
        target_paths = []
        for path in self.paths:
            if path['class'] == target_class:
                target_path = path['path']
                target_paths.append(target_path)
        
        if len(target_paths) == 0:
            return "NO PATHS TO CLASS\n"
                
        def convert_output(path_list_list):
            result = ""
            for j, list_of_nodes in enumerate(path_list_list):
                branch_str = f"branch {j+1}: "
                for i, node in enumerate(list_of_nodes):
                    if i == len(list_of_nodes) - 1:
                        branch_str += f"(FEATURE {node[0]} {node[2]} {node[3]}) \n"
                    else:
                        branch_str += f"(FEATURE {node[0]} {node[2]} {node[3]}) AND "
                result += branch_str
            return result
        
        return convert_output(target_paths)
    
    def run(self):
        
        def guide():
            guide = """
            \'q\' to quit 
            \'open\' to open trustee explanation to web browser
            \'target (class index)\' to list all branches to that target class
            """
            print(guide)
        
        guide()
        
        while True:
            user_input = input()
            
            command = user_input.split()
            
            if len(command) == 1:
                if command[0] in ['quit', 'q']:
                    break
                elif command[0] == "open":
                    filename = 'file:///'+os.getcwd()+'/' + 'output.html'
                    print(f"opening {filename} in web browser \n")
                    webbrowser.open_new_tab(filename)
                elif command[0] == "help":
                    guide()
                    
            elif len(command) >= 2:
                if command[0] == 'target':
                    if command[1].isdigit():
                        print(self.find_paths_to_class(int(command[1])))
            
            
        