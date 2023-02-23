

import numpy as np
from copy import deepcopy
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
from trustee.utils.tree import get_dt_dict
from graphviz import Digraph
import webbrowser
import os
from jinja2 import Template
from trustee.report.trust import TrustReport   


class CLIController:
    
    def __init__(self, trust_report) -> None:
        
        self.guide = """
            \'q\' to quit 
            \'open\' to open trustee explanation to web browser
            \'target (class index)\' to list all branches to that target class
            """
        self.thresholder = Thresholder(trust_report)
        self.html_creator = htmlCreator(trust_report)
        self.explanation_path = ""
            
    def run(self):
        print(self.guide)
        while True:
            user_input = input()
            
            command = user_input.split()
            
            if len(command) == 1:
                if command[0] in ['quit', 'q']:
                    break
                elif command[0] == "open":
                    if self.explanation_path == "":
                        filename = 'file:///'+os.getcwd()+'/' + "output.html"
                        self.html_creator.convert_to_html()
                        print("building html file...")
                        print(f"opening {filename} in web browser... \n")
                        webbrowser.open_new_tab(filename)
                        self.explanation_path = filename
                    else:
                        print("opening trust report... \n")
                        webbrowser.open_new_tab(self.explanation_path)
                elif command[0] == "help":
                    print(self.guide)
                    
            elif len(command) >= 2:
                if command[0] == 'target':
                    if command[1].isdigit():
                        
                        print(self.thresholder.find_paths_to_class(int(command[1])))
            
            
        
        

class Thresholder:
    
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
    
    
class htmlCreator:
        def __init__(self, trust_report : TrustReport) -> None:
            self.trust_report = trust_report
            
        def trust_report_summary(self, trust_report : TrustReport):
            
            b_summary_info = {"b_model" : type(trust_report.blackbox).__name__,"b_dataset_size" : trust_report.dataset_size, "b_train_test_split" : f"{trust_report.train_size * 100:.2f}% / {(1 - trust_report.train_size) * 100:.2f}%", "b_input_features" : trust_report.bb_n_input_features, "b_output_classes" : trust_report.bb_n_output_classes, "b_performance_data" : trust_report._score_report(trust_report.y_test, trust_report.y_pred)}
            w_summary_info = {"w_method" : "Trustee","w_model" : type(trust_report.max_dt).__name__,"w_iterations" : trust_report.trustee_num_iter, "w_sample_size" : f"{trust_report.trustee_sample_size * 100:.2f}%", "w_size" : trust_report.max_dt.tree_.node_count, "w_depth" : trust_report.max_dt.get_depth(), "w_leaves" : trust_report.max_dt.get_n_leaves(), "w_input_features": f"{trust_report.trustee.get_n_features()} ({trust_report.trustee.get_n_features() / trust_report.bb_n_input_features * 100:.2f}%)", "w_output_classes" : f"{trust_report.trustee.get_n_classes()} ({trust_report.trustee.get_n_classes() / trust_report.bb_n_output_classes * 100:.2f}%)","w_fidelity_data" : trust_report._score_report(trust_report.y_pred, trust_report.max_dt_y_pred)}
            t_summary_info = {"t_method": "Trustee","t_model" : type(trust_report.min_dt).__name__,"t_iterations" : trust_report.trustee_num_iter, "t_sample_size" : f"{trust_report.trustee_sample_size * 100:.2f}%", "t_size" : trust_report.min_dt.tree_.node_count, "t_depth" : trust_report.min_dt.get_depth(), "t_leaves" : trust_report.min_dt.get_n_leaves(), "t_input_features" : "-" , "top_k" : trust_report.top_k, "t_output_classes" : f"{trust_report.min_dt.tree_.n_classes[0]} ({trust_report.min_dt.tree_.n_classes[0] / trust_report.bb_n_output_classes * 100:.2f}%)" ,"t_fidelity_data" : trust_report._score_report(trust_report.y_pred, trust_report.min_dt_y_pred)}
            
            return (b_summary_info | w_summary_info | t_summary_info)
        
        def summary_performance(self, trust_report):
            b = trust_report._score_report(trust_report.y_test, trust_report.y_pred).split()
            w = trust_report._score_report(trust_report.y_pred, trust_report.max_dt_y_pred).split()
            t = trust_report._score_report(trust_report.y_pred, trust_report.min_dt_y_pred).split()
            
            def process(model_list):
                class_values = model_list[4:model_list.index("accuracy")]
                accuracy_values = model_list[model_list.index("accuracy") + 1: model_list.index("accuracy") + 3 ]
                macro_avg_values = model_list[model_list.index("macro")+2:model_list.index("weighted")]
                weighted_avg_values = model_list[model_list.index("weighted")+2:]
                return class_values, accuracy_values, macro_avg_values, weighted_avg_values
            
            def class_values_into_html(class_values):
                rows = len(class_values) // 5
                html_row_template = """
                    <tr>
                        <th>{{var_one}}</th>
                        <td>{{var_two}}</td>
                        <td>{{var_three}}</td>
                        <td>{{var_four}}</td>
                        <td>{{var_one}}</td>
                    </tr>
                """
                html_output = ""
                for i in range(rows):
                    template = Template(html_row_template)
                    html_output += template.render(var_one=class_values[i*5], var_two=class_values[i*5+1],var_three=class_values[i*5+2],var_four=class_values[i*5+3], var_five=class_values[i*5+4])
                return html_output
            
            b_class_values, b_accuracy_values, b_macro_avg_values, b_weighted_avg_values = process(b)
            w_class_values, w_accuracy_values, w_macro_avg_values, w_weighted_avg_values = process(w)
            t_class_values, t_accuracy_values, t_macro_avg_values, t_weighted_avg_values = process(b)
            
            b_class_html = class_values_into_html(b_class_values)
            w_class_html = class_values_into_html(w_class_values)
            t_class_html = class_values_into_html(t_class_values)
            
            b_dict = {"b_performance_data" : b_class_html, "b_f1_score_accuracy" : b_accuracy_values[0], "b_support_accuracy" : b_accuracy_values[1], "b_precision_macro_avg" : b_macro_avg_values[0], "b_recall_macro_avg" : b_macro_avg_values[1], "b_f1_score_macro_avg" : b_macro_avg_values[2], "b_support_macro_avg" : b_macro_avg_values[3], "b_precision_weighted_avg" : b_weighted_avg_values[0], "b_recall_weighted_avg" : b_weighted_avg_values[1], "b_f1_score_weighted_avg" : b_weighted_avg_values[2], "b_support_weighted_avg" : b_weighted_avg_values[3]}
            w_dict = {"w_fidelity_data" : w_class_html, "w_f1_score_accuracy" : w_accuracy_values[0], "w_support_accuracy" : w_accuracy_values[1], "w_precision_macro_avg" : w_macro_avg_values[0], "w_recall_macro_avg" : w_macro_avg_values[1], "w_f1_score_macro_avg" : w_macro_avg_values[2], "w_support_macro_avg" : w_macro_avg_values[3], "w_precision_weighted_avg" : w_weighted_avg_values[0], "w_recall_weighted_avg" : w_weighted_avg_values[1], "w_f1_score_weighted_avg" : w_weighted_avg_values[2], "w_support_weighted_avg" : w_weighted_avg_values[3]}
            t_dict = {"t_fidelity_data" : t_class_html, "t_f1_score_accuracy" : t_accuracy_values[0], "t_support_accuracy" : t_accuracy_values[1], "t_precision_macro_avg" : t_macro_avg_values[0], "t_recall_macro_avg" : t_macro_avg_values[1], "t_f1_score_macro_avg" : t_macro_avg_values[2], "t_support_macro_avg" : t_macro_avg_values[3], "t_precision_weighted_avg" : t_weighted_avg_values[0], "t_recall_weighted_avg" : t_weighted_avg_values[1], "t_f1_score_weighted_avg" : t_weighted_avg_values[2], "t_support_weighted_avg" : t_weighted_avg_values[3]}
            
            return b_dict | w_dict | t_dict
        
            
        def convert_to_html(self, input_file_name = "template.html", output_file_name = "output.html") -> None:
            
            with open(input_file_name, "r") as file:
                template = Template(file.read())
            
            self.trust_report._save_dts(f"reports", False)
            
            trust_report_information_dict = self.trust_report_summary(self.trust_report) | self.summary_performance(self.trust_report)| {"decision_tree" : f"reports/trust_report_dt.pdf", "pruned_decision_tree" : f"reports/trust_report_pruned_dt.pdf"}
            
            html = template.render(trust_report_information_dict)
            
            with open(output_file_name, "w") as file:
                file.write(html)
        