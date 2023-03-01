from jinja2 import Template
from trustee.report.trust import TrustReport   
import os
import shutil
import json


class htmlCreator:
        def __init__(self, trust_report : TrustReport, output_directory) -> None:
            self.trust_report = trust_report
            self.output_directory = output_directory
            
        def trust_report_summary(self, trust_report : TrustReport, for_json=False):
            
            b_summary_info = {"b_model" : type(trust_report.blackbox).__name__,
                              "b_dataset_size" : trust_report.dataset_size, 
                              "b_train_test_split" : f"{trust_report.train_size * 100:.2f}% / {(1 - trust_report.train_size) * 100:.2f}%", 
                              "b_input_features" : trust_report.bb_n_input_features, 
                              "b_output_classes" : trust_report.bb_n_output_classes, 
                              "b_performance_data" : trust_report._score_report(trust_report.y_test, trust_report.y_pred)}
            w_summary_info = {"w_method" : "Trustee",
                              "w_model" : type(trust_report.max_dt).__name__,
                              "w_iterations" : trust_report.trustee_num_iter, 
                              "w_sample_size" : f"{trust_report.trustee_sample_size * 100:.2f}%", 
                              "w_size" : trust_report.max_dt.tree_.node_count, 
                              "w_depth" : trust_report.max_dt.get_depth(), 
                              "w_leaves" : trust_report.max_dt.get_n_leaves(), 
                              "w_input_features": f"{trust_report.trustee.get_n_features()} ({trust_report.trustee.get_n_features() / trust_report.bb_n_input_features * 100:.2f}%)", 
                              "w_output_classes" : f"{trust_report.trustee.get_n_classes()} ({trust_report.trustee.get_n_classes() / trust_report.bb_n_output_classes * 100:.2f}%)",
                              "w_fidelity_data" : trust_report._score_report(trust_report.y_pred, trust_report.max_dt_y_pred)}
            t_summary_info = {"t_method": "Trustee",
                              "t_model" : type(trust_report.min_dt).__name__,
                              "t_iterations" : trust_report.trustee_num_iter, 
                              "t_sample_size" : f"{trust_report.trustee_sample_size * 100:.2f}%", 
                              "t_size" : trust_report.min_dt.tree_.node_count, 
                              "t_depth" : trust_report.min_dt.get_depth(), 
                              "t_leaves" : trust_report.min_dt.get_n_leaves(), 
                              "t_input_features" : "-" , 
                              "top_k" : trust_report.top_k, 
                              "t_output_classes" : f"{trust_report.min_dt.tree_.n_classes[0]} ({trust_report.min_dt.tree_.n_classes[0] / trust_report.bb_n_output_classes * 100:.2f}%)" ,
                              "t_fidelity_data" : trust_report._score_report(trust_report.y_pred, trust_report.min_dt_y_pred)}
            
            if for_json:
                
                b_summary_info = {"model" : f"{type(trust_report.blackbox).__name__}",
                                "dataset_size" : trust_report.dataset_size, 
                                "train_test_split" : f"{trust_report.train_size * 100:.2f}% / {(1 - trust_report.train_size) * 100:.2f}%", 
                                "input_features" : trust_report.bb_n_input_features, 
                                "output_classes" : trust_report.bb_n_output_classes}
                w_summary_info = {"method" : "Trustee",
                                "model" : f"{type(trust_report.max_dt).__name__}",
                                "iterations" : trust_report.trustee_num_iter, 
                                "sample_size" : f"{trust_report.trustee_sample_size * 100:.2f}%", 
                                "size" : trust_report.max_dt.tree_.node_count, 
                                "depth" : trust_report.max_dt.get_depth(), 
                                "leaves" : f"{trust_report.max_dt.get_n_leaves()}", 
                                "input_features": f"{trust_report.trustee.get_n_features()} ({trust_report.trustee.get_n_features() / trust_report.bb_n_input_features * 100:.2f}%)", 
                                "output_classes" : f"{trust_report.trustee.get_n_classes()} ({trust_report.trustee.get_n_classes() / trust_report.bb_n_output_classes * 100:.2f}%)"}
                t_summary_info = {"method": "Trustee",
                                "model" : f"{type(trust_report.min_dt).__name__}",
                                "iterations" : trust_report.trustee_num_iter, 
                                "sample_size" : f"{trust_report.trustee_sample_size * 100:.2f}%", 
                                "size" : trust_report.min_dt.tree_.node_count, 
                                "depth" : trust_report.min_dt.get_depth(), 
                                "leaves" : f"{trust_report.min_dt.get_n_leaves()}", 
                                "input_features" : "-" , 
                                "top_k" : trust_report.top_k, 
                                "output_classes" : f"{trust_report.min_dt.tree_.n_classes[0]} ({trust_report.min_dt.tree_.n_classes[0] / trust_report.bb_n_output_classes * 100:.2f}%)"}
                
                return b_summary_info, w_summary_info, t_summary_info
            
            return (b_summary_info | w_summary_info | t_summary_info)
        
        def summary_performance(self, trust_report, for_json=False):
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
                        <th style="font-weight: normal;">{{var_one}}</th>
                        <td>{{var_two}}</td>
                        <td>{{var_three}}</td>
                        <td>{{var_four}}</td>
                        <td>{{var_five}}</td>
                    </tr>
                """
                html_output = ""
                for i in range(rows):
                    template = Template(html_row_template)
                    html_output += template.render(var_one=f"{class_values[i*5]}({self.trust_report.class_names[int(class_values[i*5])]})", var_two=class_values[i*5+1],var_three=class_values[i*5+2],var_four=class_values[i*5+3], var_five=class_values[i*5+4])
                return html_output
            
            def class_values_into_dict(class_values):
                rows = len(class_values) // 5
                dict_for_json = {}
                for i in range(rows):
                    dict_for_json[f"{class_values[i*5]}({self.trust_report.class_names[int(class_values[i*5])]})"] = {"precision" : class_values[i*5+1], "recall" : class_values[i*5+2], "f1_score" : class_values[i*5+3], "support" : class_values[i*5+4]}
                return dict_for_json
            
            b_class_values, b_accuracy_values, b_macro_avg_values, b_weighted_avg_values = process(b)
            w_class_values, w_accuracy_values, w_macro_avg_values, w_weighted_avg_values = process(w)
            t_class_values, t_accuracy_values, t_macro_avg_values, t_weighted_avg_values = process(t)
            
            b_class_html = class_values_into_html(b_class_values)
            w_class_html = class_values_into_html(w_class_values)
            t_class_html = class_values_into_html(t_class_values)
            
            b_dict = {"b_performance_data" : b_class_html, 
                      "b_f1_score_accuracy" : b_accuracy_values[0], 
                      "b_support_accuracy" : b_accuracy_values[1], 
                      "b_precision_macro_avg" : b_macro_avg_values[0], 
                      "b_recall_macro_avg" : b_macro_avg_values[1], 
                      "b_f1_score_macro_avg" : b_macro_avg_values[2], 
                      "b_support_macro_avg" : b_macro_avg_values[3], 
                      "b_precision_weighted_avg" : b_weighted_avg_values[0], 
                      "b_recall_weighted_avg" : b_weighted_avg_values[1], 
                      "b_f1_score_weighted_avg" : b_weighted_avg_values[2], 
                      "b_support_weighted_avg" : b_weighted_avg_values[3]}
            w_dict = {"w_fidelity_data" : w_class_html, 
                      "w_f1_score_accuracy" : w_accuracy_values[0], 
                      "w_support_accuracy" : w_accuracy_values[1], 
                      "w_precision_macro_avg" : w_macro_avg_values[0], 
                      "w_recall_macro_avg" : w_macro_avg_values[1], 
                      "w_f1_score_macro_avg" : w_macro_avg_values[2], 
                      "w_support_macro_avg" : w_macro_avg_values[3], 
                      "w_precision_weighted_avg" : w_weighted_avg_values[0], 
                      "w_recall_weighted_avg" : w_weighted_avg_values[1], 
                      "w_f1_score_weighted_avg" : w_weighted_avg_values[2], 
                      "w_support_weighted_avg" : w_weighted_avg_values[3]}
            t_dict = {"t_fidelity_data" : t_class_html, 
                      "t_f1_score_accuracy" : t_accuracy_values[0], 
                      "t_support_accuracy" : t_accuracy_values[1], 
                      "t_precision_macro_avg" : t_macro_avg_values[0], 
                      "t_recall_macro_avg" : t_macro_avg_values[1], 
                      "t_f1_score_macro_avg" : t_macro_avg_values[2],
                      "t_support_macro_avg" : t_macro_avg_values[3], 
                      "t_precision_weighted_avg" : t_weighted_avg_values[0], 
                      "t_recall_weighted_avg" : t_weighted_avg_values[1], 
                      "t_f1_score_weighted_avg" : t_weighted_avg_values[2], 
                      "t_support_weighted_avg" : t_weighted_avg_values[3]}
            
            if for_json:
                b_dict = {"class_performance_data" : class_values_into_dict(b_class_values),
                          "accuracy" : {"f1_score" : b_accuracy_values[0], "support" : b_accuracy_values[1]}, 
                          "macro_avg" : {"precision" : b_macro_avg_values[0], "recall" : b_macro_avg_values[1], "f1_score" : b_macro_avg_values[2], "support" : b_macro_avg_values[3]},
                          "weighted_avg" : {"precision" : b_weighted_avg_values[0], "recall" : b_weighted_avg_values[1], "f1_score" : b_weighted_avg_values[2], "support" : b_weighted_avg_values[3]}}
                
                w_dict = {"class_performance_data" : class_values_into_dict(w_class_values),
                          "accuracy" : {"f1_score" : w_accuracy_values[0], "support" : w_accuracy_values[1]}, 
                          "macro_avg" : {"precision" : w_macro_avg_values[0], "recall" : w_macro_avg_values[1], "f1_score" : w_macro_avg_values[2], "support" : w_macro_avg_values[3]},
                          "weighted_avg" : {"precision" : w_weighted_avg_values[0], "recall" : w_weighted_avg_values[1], "f1_score" : w_weighted_avg_values[2], "support" : w_weighted_avg_values[3]}}
                
                t_dict = {"class_performance_data" : class_values_into_dict(t_class_values),
                          "accuracy" : {"f1_score" : t_accuracy_values[0], "support" : t_accuracy_values[1]}, 
                          "macro_avg" : {"precision" : t_macro_avg_values[0], "recall" : t_macro_avg_values[1], "f1_score" : t_macro_avg_values[2], "support" : t_macro_avg_values[3]},
                          "weighted_avg" : {"precision" : t_weighted_avg_values[0], "recall" : t_weighted_avg_values[1], "f1_score" : t_weighted_avg_values[2], "support" : t_weighted_avg_values[3]}}
                
                return b_dict , w_dict , t_dict

            return b_dict | w_dict | t_dict
        
        def json_summary(self):
            b_summary_info, w_summary_info, t_summary_info = self.trust_report_summary(self.trust_report, True)
            b_dict , w_dict , t_dict = self.summary_performance(self.trust_report,True)
            
            b = {"info" : b_summary_info} | {"performance Data" : b_dict}
            w = {"tree Info" : w_summary_info} | {"fidelity Data" : w_dict}
            t = {"tree Info" : t_summary_info} | {"fidelity Data" :t_dict}
            
            return {"black_box" : b} | {"white_box" : w} | {"top_k":t}
        
            
        def convert_to_html(self, input_file_name = os.path.dirname(__file__) + "/template.html", output_file_name =  "/output.html") -> None:
            
            output_path = self.output_directory
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            with open(input_file_name, "r") as file:
                template = Template(file.read())
            
            self.trust_report._save_dts(f"{self.output_directory}/reports", False)
            
            trust_report_information_dict = self.trust_report_summary(self.trust_report) | self.summary_performance(self.trust_report)| {"decision_tree" : os.getcwd() + "/" + self.output_directory + "/reports/trust_report_dt.pdf", "pruned_decision_tree" : os.getcwd() + "/" + self.output_directory +"/reports/trust_report_pruned_dt.pdf"}
            
            html = template.render(trust_report_information_dict)

            with open(output_path + output_file_name, "w") as file:
                file.write(html)
                
            with open(output_path + "/trust_report_summary.json", "w") as file:
                json.dump(self.json_summary(), file) 
                           
            src_file = os.path.dirname(__file__) +"/" + 'style.css'
            dst_dir = output_path + "/style.css"
            shutil.copy(src_file, dst_dir)
            
        