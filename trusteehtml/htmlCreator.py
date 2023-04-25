from jinja2 import Template
from trustee.report.trust import TrustReport   
from copy import deepcopy
import os
import shutil
import json
import glob

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
        
        def single_run_report(self, forJson=False):
            def getFeatures():
                sum_nodes = 0
                sum_nodes_perc = 0
                sum_data_split = 0
                
                result = ""
                htmlTemp = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    </tr>
                    """
                for (feat,values) in self.trust_report.max_dt_top_features:
                    node, node_perc, data_split = (
                    values["count"],
                    (values["count"] / (self.trust_report.max_dt.tree_.node_count - self.trust_report.max_dt.tree_.n_leaves)) * 100,
                    (values["samples"] / self.trust_report.trustee.get_samples_sum()) * 100,
                    )
                    sum_nodes += node
                    sum_nodes_perc += node_perc
                    sum_data_split += data_split
                    
                    tempDict = {
                        "var_one" : self.trust_report.feature_names[feat],
                        "var_two" : f"{node} ({node_perc:.2f}%)",
                        "var_three" : f"{values['samples']} ({data_split:.2f}%)"
                    }
                    template = Template(htmlTemp)
                    result += template.render(tempDict)
                result = result + f"""
                <tr>
                <th>Top {len(self.trust_report.max_dt_top_features)} Summary</th>
                <td>{sum_nodes} ({sum_nodes_perc:.2f}%)</td>
                <td>{sum_data_split:.2f}%</td>
                </tr>
                """
                return result
            
            def getNodes():
                out = ""
                htmlTemp = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    <td>{{var_four}}</td>
                    </tr>
                """
                for node in self.trust_report.max_dt_top_nodes:
                    samplesByClass = [
                        (
                            self.trust_report.class_names[idx] if self.trust_report.class_names is not None and idx < len(self.trust_report.class_names) else idx,
                            (count_left / self.trust_report.max_dt.tree_.value[0][0][idx]) * 100,
                            (count_right / self.trust_report.max_dt.tree_.value[0][0][idx]) * 100,
                        )
                        for idx, (count_left, count_right) in enumerate(node["data_split_by_class"])
                    ]
                    samples_left = (node["data_split"][0] / self.trust_report.max_dt.tree_.n_node_samples[0]) * 100
                    samples_right = (node["data_split"][1] / self.trust_report.max_dt.tree_.n_node_samples[0]) * 100
                    tempDict = {
                        "var_one" : f"{self.trust_report.feature_names[node['feature']] if self.trust_report.feature_names else node['feature']} <= {node['threshold']}",
                        "var_two" : f"Left: {node['gini_split'][0]:.2f} \nRight: {node['gini_split'][1]:.2f}",
                        "var_three" : f"Left: {samples_left:.2f}% \nRight: {samples_right:.2f}%",
                        "var_four" : "\n".join(f"{row[0]}: {row[1]:.2f}% / {row[2]:.2f}%" for row in samplesByClass)
                    }
                    htmlR = Template(htmlTemp)
                    out += htmlR.render(tempDict)
                    
                return out
            
            def getBranches():
                result = ""
                htmlTemp = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    <td>{{var_four}}</td>
                    </tr>
                """
                
                sum_samples = 0
                sum_samples_perc = 0
                sum_class_samples_perc = {}
                for branch in self.trust_report.max_dt_top_branches:
                    samples, samples_perc, class_samples_perc = (
                    branch["samples"],
                    (branch["samples"] / self.trust_report.max_dt.tree_.n_node_samples[0]) * 100,
                    (branch["samples"] / self.trust_report.max_dt.tree_.value[0][0][branch["class"]]) * 100 if self.trust_report.is_classify else 0,
                    )
                    sum_samples += samples
                    sum_samples_perc += samples_perc
                    if branch["class"] not in sum_class_samples_perc:
                        sum_class_samples_perc[branch["class"]] = 0
                    sum_class_samples_perc[branch["class"]] += class_samples_perc
                    
                    branch_class = (
                        self.trust_report.class_names[branch["class"]]
                        if self.trust_report.class_names is not None and branch["class"] < len(self.trust_report.class_names)
                        else branch["class"],
                    )
                    htmlDict = {
                        "var_one" : [
                                f"{self.trust_report.feature_names[feat] if self.trust_report.feature_names else feat} {op} {threshold}"
                                for (_, feat, op, threshold) in branch["path"]
                            ],
                        "var_two" : f"{branch_class}\n({branch['prob']:.2f}%)",
                        "var_three" : f"{samples}\n({samples_perc:.2f}%)",
                        "var_four" : f"{class_samples_perc:.2f}%"
                    }
                    temp = Template(htmlTemp)
                    result += temp.render(htmlDict)
                return result
                
                
            return  {
                "s_r_features_num" : len(self.trust_report.max_dt_top_features),
                "s_r_nodes_num" : len(self.trust_report.max_dt_top_nodes),
                "s_r_branches_num" : len(self.trust_report.max_dt_top_branches),
                "s_r_feature_body" : getFeatures(),
                "s_r_nodes_body" : getNodes(),
                "s_r_branches_body" : getBranches()
                
            }
            
        def TopKIterationBody(self):
                rowTemplate = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    <td>{{var_four}}</td>
                    <td><table>{{var_five}}</table></td>
                    <td><table>{{var_six}}</table></td>
                    </tr>
                """
                tableTemplate = """
                    <tr>
                        <th style="font-weight: normal;">{{var_one}}</th>
                        <td>{{var_two}}</td>
                        <td>{{var_three}}</td>
                        <td>{{var_four}}</td>
                        <td>{{var_five}}</td>
                    </tr>
                """
                tableHeader = """
                    <tr>
                        <th></th>
                        <th>precision</th>
                        <th>recall</th>
                        <th>f1-score</th>
                        <th>support</th>
                    </tr>
                """
                tableFooter = """
                    <tr>
                        <th>accuracy</th>
                        <td></td>
                        <td></td>
                        <td>{{accuracy_one}}</td>
                        <td>{{accuracy_two}}</td>
                    </tr>
                    <tr>
                        <th>macro avg</th>
                        <td>{{macro_one}}</td>
                        <td>{{macro_two}}</td>
                        <td>{{macro_three}}</td>
                        <td>{{macro_four}}</td>
                    </tr>
                    <tr>
                        <th>weighted avg</th>
                        <td>{{weighted_one}}</td>
                        <td>{{weighted_two}}</td>
                        <td>{{weighted_three}}</td>
                        <td>{{weighted_four}}</td>
                    </tr>
                """
                body = ""
                for i in self.trust_report.top_k_prune_iter:
                    
                    p_class_info = i["score_report"].split()
                    p_class_values = p_class_info[4: p_class_info.index("accuracy")]
                    p_accuracy_values = p_class_info[p_class_info.index("accuracy")+ 1: p_class_info.index("accuracy") + 3]
                    p_macro_avg_values = p_class_info[p_class_info.index("macro") + 2: p_class_info.index("weighted")]
                    p_weighted_avg_values = p_class_info[p_class_info.index("weighted")+2:]
                    
                    f_class_info = i["fidelity_report"].split()
                    f_class_values = f_class_info[4: f_class_info.index("accuracy")]
                    f_accuracy_values = f_class_info[f_class_info.index("accuracy")+ 1: f_class_info.index("accuracy") + 3]
                    f_macro_avg_values = f_class_info[f_class_info.index("macro") + 2: f_class_info.index("weighted")]
                    f_weighted_avg_values = f_class_info[f_class_info.index("weighted")+2:]
                    
                    rows = len(p_class_values) // 5
                    
                    f_table = tableHeader
                    p_table = tableHeader
                    p_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        p_table += p_table_temp.render(var_one=f"{p_class_values[j*5]}({self.trust_report.class_names[int(p_class_values[j*5])]})", var_two=p_class_values[j*5+1],var_three=p_class_values[j*5+2],var_four=p_class_values[j*5+3], var_five=p_class_values[j*5+4])
                    
                    f_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        f_table += f_table_temp.render(var_one=f"{f_class_values[j*5]}({self.trust_report.class_names[int(f_class_values[j*5])]})", var_two=f_class_values[j*5+1],var_three=f_class_values[j*5+2],var_four=f_class_values[j*5+3], var_five=f_class_values[j*5+4])
                    
                    full_row_temp = Template(rowTemplate)
                    p_footer = Template(tableFooter)
                    f_footer = Template(tableFooter)
                    p_footer = p_footer.render(accuracy_one=p_accuracy_values[0], accuracy_two=[p_accuracy_values[1]],macro_one=p_macro_avg_values[0], macro_two=p_macro_avg_values[1], macro_three=p_macro_avg_values[2], macro_four=p_macro_avg_values[3], weighted_one=p_weighted_avg_values[0], weighted_two=p_weighted_avg_values[1], weighted_three=p_weighted_avg_values[2], weighted_four=p_weighted_avg_values[3])
                    f_footer = f_footer.render(accuracy_one=f_accuracy_values[0], accuracy_two=[f_accuracy_values[1]],macro_one=f_macro_avg_values[0], macro_two=f_macro_avg_values[1], macro_three=f_macro_avg_values[2], macro_four=f_macro_avg_values[3], weighted_one=f_weighted_avg_values[0], weighted_two=f_weighted_avg_values[1], weighted_three=f_weighted_avg_values[2], weighted_four=f_weighted_avg_values[3])
                    body += full_row_temp.render(var_one=i["top_k"], var_two=i["dt"].tree_.node_count, var_three=i["dt"].get_depth(),var_four=i["dt"].get_n_leaves(), var_five=p_table + p_footer, var_six=f_table + f_footer)               
                return {"TrusteeTopKIterationBody" : body}
                
        
        def CCPAlphaBody(self):
                rowTemplate = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    <td>{{var_four}}</td>
                    <td>{{var_five}}</td>
                    <td><table>{{var_six}}</table></td>
                    <td><table>{{var_seven}}</table></td>
                    </tr>
                """
                tableTemplate = """
                    <tr>
                        <th style="font-weight: normal;">{{var_one}}</th>
                        <td>{{var_two}}</td>
                        <td>{{var_three}}</td>
                        <td>{{var_four}}</td>
                        <td>{{var_five}}</td>
                    </tr>
                """
                tableHeader = """
                    <tr>
                        <th></th>
                        <th>precision</th>
                        <th>recall</th>
                        <th>f1-score</th>
                        <th>support</th>
                    </tr>
                """
                tableFooter = """
                    <tr>
                        <th>accuracy</th>
                        <td></td>
                        <td></td>
                        <td>{{accuracy_one}}</td>
                        <td>{{accuracy_two}}</td>
                    </tr>
                    <tr>
                        <th>macro avg</th>
                        <td>{{macro_one}}</td>
                        <td>{{macro_two}}</td>
                        <td>{{macro_three}}</td>
                        <td>{{macro_four}}</td>
                    </tr>
                    <tr>
                        <th>weighted avg</th>
                        <td>{{weighted_one}}</td>
                        <td>{{weighted_two}}</td>
                        <td>{{weighted_three}}</td>
                        <td>{{weighted_four}}</td>
                    </tr>
                """
                body = ""
                for i in self.trust_report.ccp_iter:
                    
                    p_class_info = i["score_report"].split()
                    p_class_values = p_class_info[4: p_class_info.index("accuracy")]
                    p_accuracy_values = p_class_info[p_class_info.index("accuracy")+ 1: p_class_info.index("accuracy") + 3]
                    p_macro_avg_values = p_class_info[p_class_info.index("macro") + 2: p_class_info.index("weighted")]
                    p_weighted_avg_values = p_class_info[p_class_info.index("weighted")+2:]
                    
                    f_class_info = i["fidelity_report"].split()
                    f_class_values = f_class_info[4: f_class_info.index("accuracy")]
                    f_accuracy_values = f_class_info[f_class_info.index("accuracy")+ 1: f_class_info.index("accuracy") + 3]
                    f_macro_avg_values = f_class_info[f_class_info.index("macro") + 2: f_class_info.index("weighted")]
                    f_weighted_avg_values = f_class_info[f_class_info.index("weighted")+2:]
                    
                    rows = len(p_class_values) // 5
                    
                    f_table = tableHeader
                    p_table = tableHeader
                    p_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        p_table += p_table_temp.render(var_one=f"{p_class_values[j*5]}({self.trust_report.class_names[int(p_class_values[j*5])]})", var_two=p_class_values[j*5+1],var_three=p_class_values[j*5+2],var_four=p_class_values[j*5+3], var_five=p_class_values[j*5+4])
                    
                    f_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        f_table += f_table_temp.render(var_one=f"{f_class_values[j*5]}({self.trust_report.class_names[int(f_class_values[j*5])]})", var_two=f_class_values[j*5+1],var_three=f_class_values[j*5+2],var_four=f_class_values[j*5+3], var_five=f_class_values[j*5+4])
                    
                    full_row_temp = Template(rowTemplate)
                    p_footer = Template(tableFooter)
                    f_footer = Template(tableFooter)
                    p_footer = p_footer.render(accuracy_one=p_accuracy_values[0], accuracy_two=[p_accuracy_values[1]],macro_one=p_macro_avg_values[0], macro_two=p_macro_avg_values[1], macro_three=p_macro_avg_values[2], macro_four=p_macro_avg_values[3], weighted_one=p_weighted_avg_values[0], weighted_two=p_weighted_avg_values[1], weighted_three=p_weighted_avg_values[2], weighted_four=p_weighted_avg_values[3])
                    f_footer = f_footer.render(accuracy_one=f_accuracy_values[0], accuracy_two=[f_accuracy_values[1]],macro_one=f_macro_avg_values[0], macro_two=f_macro_avg_values[1], macro_three=f_macro_avg_values[2], macro_four=f_macro_avg_values[3], weighted_one=f_weighted_avg_values[0], weighted_two=f_weighted_avg_values[1], weighted_three=f_weighted_avg_values[2], weighted_four=f_weighted_avg_values[3])
                    body += full_row_temp.render(var_one=i["ccp_alpha"], var_two=f"{i['gini']:.3f}", var_three=i["dt"].tree_.node_count,var_four=i["dt"].get_depth(),var_five=i["dt"].get_n_leaves(), var_six=p_table + p_footer, var_seven=f_table + f_footer)               
                return {"CCPAlphaBody" : body}             
                
        def MaxDepthBody(self):
                rowTemplate = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    <td>{{var_four}}</td>
                    <td><table>{{var_five}}</table></td>
                    <td><table>{{var_six}}</table></td>
                    </tr>
                """
                tableTemplate = """
                    <tr>
                        <th style="font-weight: normal;">{{var_one}}</th>
                        <td>{{var_two}}</td>
                        <td>{{var_three}}</td>
                        <td>{{var_four}}</td>
                        <td>{{var_five}}</td>
                    </tr>
                """
                tableHeader = """
                    <tr>
                        <th></th>
                        <th>precision</th>
                        <th>recall</th>
                        <th>f1-score</th>
                        <th>support</th>
                    </tr>
                """
                tableFooter = """
                    <tr>
                        <th>accuracy</th>
                        <td></td>
                        <td></td>
                        <td>{{accuracy_one}}</td>
                        <td>{{accuracy_two}}</td>
                    </tr>
                    <tr>
                        <th>macro avg</th>
                        <td>{{macro_one}}</td>
                        <td>{{macro_two}}</td>
                        <td>{{macro_three}}</td>
                        <td>{{macro_four}}</td>
                    </tr>
                    <tr>
                        <th>weighted avg</th>
                        <td>{{weighted_one}}</td>
                        <td>{{weighted_two}}</td>
                        <td>{{weighted_three}}</td>
                        <td>{{weighted_four}}</td>
                    </tr>
                """
                body = ""
                for i in self.trust_report.max_depth_iter:
                    
                    p_class_info = i["score_report"].split()
                    p_class_values = p_class_info[4: p_class_info.index("accuracy")]
                    p_accuracy_values = p_class_info[p_class_info.index("accuracy")+ 1: p_class_info.index("accuracy") + 3]
                    p_macro_avg_values = p_class_info[p_class_info.index("macro") + 2: p_class_info.index("weighted")]
                    p_weighted_avg_values = p_class_info[p_class_info.index("weighted")+2:]
                    
                    f_class_info = i["fidelity_report"].split()
                    f_class_values = f_class_info[4: f_class_info.index("accuracy")]
                    f_accuracy_values = f_class_info[f_class_info.index("accuracy")+ 1: f_class_info.index("accuracy") + 3]
                    f_macro_avg_values = f_class_info[f_class_info.index("macro") + 2: f_class_info.index("weighted")]
                    f_weighted_avg_values = f_class_info[f_class_info.index("weighted")+2:]
                    
                    rows = len(p_class_values) // 5
                    
                    f_table = tableHeader
                    p_table = tableHeader
                    p_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        p_table += p_table_temp.render(var_one=f"{p_class_values[j*5]}({self.trust_report.class_names[int(p_class_values[j*5])]})", var_two=p_class_values[j*5+1],var_three=p_class_values[j*5+2],var_four=p_class_values[j*5+3], var_five=p_class_values[j*5+4])
                    
                    f_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        f_table += f_table_temp.render(var_one=f"{f_class_values[j*5]}({self.trust_report.class_names[int(f_class_values[j*5])]})", var_two=f_class_values[j*5+1],var_three=f_class_values[j*5+2],var_four=f_class_values[j*5+3], var_five=f_class_values[j*5+4])
                    
                    full_row_temp = Template(rowTemplate)
                    p_footer = Template(tableFooter)
                    f_footer = Template(tableFooter)
                    p_footer = p_footer.render(accuracy_one=p_accuracy_values[0], accuracy_two=[p_accuracy_values[1]],macro_one=p_macro_avg_values[0], macro_two=p_macro_avg_values[1], macro_three=p_macro_avg_values[2], macro_four=p_macro_avg_values[3], weighted_one=p_weighted_avg_values[0], weighted_two=p_weighted_avg_values[1], weighted_three=p_weighted_avg_values[2], weighted_four=p_weighted_avg_values[3])
                    f_footer = f_footer.render(accuracy_one=f_accuracy_values[0], accuracy_two=[f_accuracy_values[1]],macro_one=f_macro_avg_values[0], macro_two=f_macro_avg_values[1], macro_three=f_macro_avg_values[2], macro_four=f_macro_avg_values[3], weighted_one=f_weighted_avg_values[0], weighted_two=f_weighted_avg_values[1], weighted_three=f_weighted_avg_values[2], weighted_four=f_weighted_avg_values[3])
                    body += full_row_temp.render(var_one=i["max_depth"], var_two=i["dt"].tree_.node_count, var_three=i["dt"].get_depth(),var_four=i["dt"].get_n_leaves(), var_five=p_table + p_footer, var_six=f_table + f_footer)               
                return {"MaxDepthBody" : body}
            
        def MaxLeavesBody(self):
                rowTemplate = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    <td>{{var_four}}</td>
                    <td><table>{{var_five}}</table></td>
                    <td><table>{{var_six}}</table></td>
                    </tr>
                """
                tableTemplate = """
                    <tr>
                        <th style="font-weight: normal;">{{var_one}}</th>
                        <td>{{var_two}}</td>
                        <td>{{var_three}}</td>
                        <td>{{var_four}}</td>
                        <td>{{var_five}}</td>
                    </tr>
                """
                tableHeader = """
                    <tr>
                        <th></th>
                        <th>precision</th>
                        <th>recall</th>
                        <th>f1-score</th>
                        <th>support</th>
                    </tr>
                """
                tableFooter = """
                    <tr>
                        <th>accuracy</th>
                        <td></td>
                        <td></td>
                        <td>{{accuracy_one}}</td>
                        <td>{{accuracy_two}}</td>
                    </tr>
                    <tr>
                        <th>macro avg</th>
                        <td>{{macro_one}}</td>
                        <td>{{macro_two}}</td>
                        <td>{{macro_three}}</td>
                        <td>{{macro_four}}</td>
                    </tr>
                    <tr>
                        <th>weighted avg</th>
                        <td>{{weighted_one}}</td>
                        <td>{{weighted_two}}</td>
                        <td>{{weighted_three}}</td>
                        <td>{{weighted_four}}</td>
                    </tr>
                """
                body = ""
                for i in self.trust_report.max_leaves_iter:
                    
                    p_class_info = i["score_report"].split()
                    p_class_values = p_class_info[4: p_class_info.index("accuracy")]
                    p_accuracy_values = p_class_info[p_class_info.index("accuracy")+ 1: p_class_info.index("accuracy") + 3]
                    p_macro_avg_values = p_class_info[p_class_info.index("macro") + 2: p_class_info.index("weighted")]
                    p_weighted_avg_values = p_class_info[p_class_info.index("weighted")+2:]
                    
                    f_class_info = i["fidelity_report"].split()
                    f_class_values = f_class_info[4: f_class_info.index("accuracy")]
                    f_accuracy_values = f_class_info[f_class_info.index("accuracy")+ 1: f_class_info.index("accuracy") + 3]
                    f_macro_avg_values = f_class_info[f_class_info.index("macro") + 2: f_class_info.index("weighted")]
                    f_weighted_avg_values = f_class_info[f_class_info.index("weighted")+2:]
                    
                    rows = len(p_class_values) // 5
                    
                    f_table = tableHeader
                    p_table = tableHeader
                    p_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        p_table += p_table_temp.render(var_one=f"{p_class_values[j*5]}({self.trust_report.class_names[int(p_class_values[j*5])]})", var_two=p_class_values[j*5+1],var_three=p_class_values[j*5+2],var_four=p_class_values[j*5+3], var_five=p_class_values[j*5+4])
                    
                    f_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        f_table += f_table_temp.render(var_one=f"{f_class_values[j*5]}({self.trust_report.class_names[int(f_class_values[j*5])]})", var_two=f_class_values[j*5+1],var_three=f_class_values[j*5+2],var_four=f_class_values[j*5+3], var_five=f_class_values[j*5+4])
                    
                    full_row_temp = Template(rowTemplate)
                    p_footer = Template(tableFooter)
                    f_footer = Template(tableFooter)
                    p_footer = p_footer.render(accuracy_one=p_accuracy_values[0], accuracy_two=[p_accuracy_values[1]],macro_one=p_macro_avg_values[0], macro_two=p_macro_avg_values[1], macro_three=p_macro_avg_values[2], macro_four=p_macro_avg_values[3], weighted_one=p_weighted_avg_values[0], weighted_two=p_weighted_avg_values[1], weighted_three=p_weighted_avg_values[2], weighted_four=p_weighted_avg_values[3])
                    f_footer = f_footer.render(accuracy_one=f_accuracy_values[0], accuracy_two=[f_accuracy_values[1]],macro_one=f_macro_avg_values[0], macro_two=f_macro_avg_values[1], macro_three=f_macro_avg_values[2], macro_four=f_macro_avg_values[3], weighted_one=f_weighted_avg_values[0], weighted_two=f_weighted_avg_values[1], weighted_three=f_weighted_avg_values[2], weighted_four=f_weighted_avg_values[3])
                    body += full_row_temp.render(var_one=i["max_leaves"], var_two=i["dt"].tree_.node_count, var_three=i["dt"].get_depth(),var_four=i["dt"].get_n_leaves(), var_five=p_table + p_footer, var_six=f_table + f_footer)               
                return {"MaxLeavesBody" : body}
            
        def IterativeFeatureRemovalBody(self):
                rowTemplate = """
                    <tr>
                    <td>{{var_one}}</td>
                    <td>{{var_two}}</td>
                    <td>{{var_three}}</td>
                    <td><table>{{var_four}}</table></td>
                    <td>{{var_five}}</td>
                    <td><table>{{var_six}}</table></td>
                    </tr>
                """
                tableTemplate = """
                    <tr>
                        <th style="font-weight: normal;">{{var_one}}</th>
                        <td>{{var_two}}</td>
                        <td>{{var_three}}</td>
                        <td>{{var_four}}</td>
                        <td>{{var_five}}</td>
                    </tr>
                """
                tableHeader = """
                    <tr>
                        <th></th>
                        <th>precision</th>
                        <th>recall</th>
                        <th>f1-score</th>
                        <th>support</th>
                    </tr>
                """
                tableFooter = """
                    <tr>
                        <th>accuracy</th>
                        <td></td>
                        <td></td>
                        <td>{{accuracy_one}}</td>
                        <td>{{accuracy_two}}</td>
                    </tr>
                    <tr>
                        <th>macro avg</th>
                        <td>{{macro_one}}</td>
                        <td>{{macro_two}}</td>
                        <td>{{macro_three}}</td>
                        <td>{{macro_four}}</td>
                    </tr>
                    <tr>
                        <th>weighted avg</th>
                        <td>{{weighted_one}}</td>
                        <td>{{weighted_two}}</td>
                        <td>{{weighted_three}}</td>
                        <td>{{weighted_four}}</td>
                    </tr>
                """
                body = ""
                for i in self.trust_report.whitebox_iter:
                    
                    p_class_info = i["score_report"].split()
                    p_class_values = p_class_info[4: p_class_info.index("accuracy")]
                    p_accuracy_values = p_class_info[p_class_info.index("accuracy")+ 1: p_class_info.index("accuracy") + 3]
                    p_macro_avg_values = p_class_info[p_class_info.index("macro") + 2: p_class_info.index("weighted")]
                    p_weighted_avg_values = p_class_info[p_class_info.index("weighted")+2:]
                    
                    f_class_info = i["fidelity_report"].split()
                    f_class_values = f_class_info[4: f_class_info.index("accuracy")]
                    f_accuracy_values = f_class_info[f_class_info.index("accuracy")+ 1: f_class_info.index("accuracy") + 3]
                    f_macro_avg_values = f_class_info[f_class_info.index("macro") + 2: f_class_info.index("weighted")]
                    f_weighted_avg_values = f_class_info[f_class_info.index("weighted")+2:]
                    
                    rows = len(p_class_values) // 5
                    
                    f_table = tableHeader
                    p_table = tableHeader
                    p_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        p_table += p_table_temp.render(var_one=f"{p_class_values[j*5]}({self.trust_report.class_names[int(p_class_values[j*5])]})", var_two=p_class_values[j*5+1],var_three=p_class_values[j*5+2],var_four=p_class_values[j*5+3], var_five=p_class_values[j*5+4])
                    
                    f_table_temp = Template(tableTemplate)
                    for j in range(rows):
                        f_table += f_table_temp.render(var_one=f"{f_class_values[j*5]}({self.trust_report.class_names[int(f_class_values[j*5])]})", var_two=f_class_values[j*5+1],var_three=f_class_values[j*5+2],var_four=f_class_values[j*5+3], var_five=f_class_values[j*5+4])
                    
                    full_row_temp = Template(rowTemplate)
                    p_footer = Template(tableFooter)
                    f_footer = Template(tableFooter)
                    p_footer = p_footer.render(accuracy_one=p_accuracy_values[0], accuracy_two=[p_accuracy_values[1]],macro_one=p_macro_avg_values[0], macro_two=p_macro_avg_values[1], macro_three=p_macro_avg_values[2], macro_four=p_macro_avg_values[3], weighted_one=p_weighted_avg_values[0], weighted_two=p_weighted_avg_values[1], weighted_three=p_weighted_avg_values[2], weighted_four=p_weighted_avg_values[3])
                    f_footer = f_footer.render(accuracy_one=f_accuracy_values[0], accuracy_two=[f_accuracy_values[1]],macro_one=f_macro_avg_values[0], macro_two=f_macro_avg_values[1], macro_three=f_macro_avg_values[2], macro_four=f_macro_avg_values[3], weighted_one=f_weighted_avg_values[0], weighted_two=f_weighted_avg_values[1], weighted_three=f_weighted_avg_values[2], weighted_four=f_weighted_avg_values[3])
                    body += full_row_temp.render(var_one=i["it"], var_two=self.trust_report.feature_names[i["feature_removed"]] if self.trust_report.feature_names else i["feature_removed"], var_three=i["n_features_removed"],var_four=p_table + p_footer, var_five=i["dt"].tree_.node_count, var_six=f_table + f_footer)               
                return {"IterativeFeatureRemovalBody" : body}
            
        def Thresholds(self):
            header = "<h1>Thresholds</h1>"
            htmlTemp = """
                <a href="{{img_src}}">
                    <h2>{{threshold_values}}</h2>
                </a>
            """
            out = ""
            dir = f"./{self.output_directory}/reports/*.png"
            pdfFiles = []
            for file in glob.glob(dir):
                if file[-9:-4] == "lab56":
                    pdfFiles.append(file)
                    
            if len(pdfFiles) == 0:
                print("No Threshold")
                return {"Thresholds" : out}
            
            for file in pdfFiles:
                temp = Template(htmlTemp)
                index = file.rfind("subtree")
                values = file[index:]
                values = values.replace("_lab56.png", "")
                values = values.replace("_"," ")
                values = values.split()
                if len(values) == 3:
                    if values[1] == "qi":
                        out += temp.render(threshold_values=f"Quart Impurity: {self.trust_report.class_names[int(values[2])]}", img_src=file[file.rfind("reports"):])
                    elif values[1] == "aic":
                        out += temp.render(threshold_values=f"Avg Impurity Change: {self.trust_report.class_names[int(values[2])]}", img_src=file[file.rfind("reports"):])
                    elif values[1] == "full":
                        out += temp.render(threshold_values=f"Full: {self.trust_report.class_names[int(values[2])]}", img_src=file[file.rfind("reports"):])
                elif len(values) == 4:
                    if values[1] == "cus":
                        out += temp.render(threshold_values=f"Custom: {self.trust_report.class_names[int(values[2])]}, limit:{values[2]}", img_src=file[file.rfind("reports"):])
            return {"Thresholds" : header + out}
            
            
        def convert_to_html(self, input_file_name = os.path.dirname(__file__) + "/template.html", output_file_name =  "/output.html") -> None:
            
            output_path = self.output_directory
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            with open(input_file_name, "r") as file:
                template = Template(file.read())
            
            self.trust_report._save_dts(f"{self.output_directory}/reports", False)
            
            trust_report_information_dict = self.trust_report_summary(self.trust_report) | self.summary_performance(self.trust_report)| self.TopKIterationBody()|self.CCPAlphaBody()| self.MaxDepthBody()|self.MaxLeavesBody()|self.IterativeFeatureRemovalBody() |self.Thresholds()|{"decision_tree" : os.getcwd() + "/" + self.output_directory + "/reports/trust_report_dt.pdf", "pruned_decision_tree" : os.getcwd() + "/" + self.output_directory +"/reports/trust_report_pruned_dt.pdf"}
            trust_report_information_dict = trust_report_information_dict | self.single_run_report()
            html = template.render(trust_report_information_dict)

            with open(output_path + output_file_name, "w") as file:
                file.write(html)
                
            with open(output_path + "/trust_report_summary.json", "w") as file:
                json.dump(self.json_summary(), file) 
                           
            src_file = os.path.dirname(__file__) +"/" + 'style.css'
            dst_dir = output_path + "/style.css"
            shutil.copy(src_file, dst_dir)
            
        