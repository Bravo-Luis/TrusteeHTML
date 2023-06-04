import os
import json
import glob
from copy import deepcopy
from trustee.report.trust import TrustReport
import openai
import concurrent.futures
import graphviz


class TrusteeFormatter:
    
    def __init__(self, trust_report : TrustReport, output_dir : str, all_dts=False) -> None:
        self.trust_report = trust_report
        self.output_dir = output_dir
        self.all_dts = all_dts
        self.report_dict = {}
    
    def json(self):
        self.report_dict = {
            "summary" : self.report_summary(),
            "single_run_analysis" : self.single_run_analysis(),
            "prunning_analysis" : self.prunning_analysis(),
            "repeated_run_analysis" : self.repeated_run_analysis()
        }
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        with open(self.output_dir + '/trust_report.json', 'w') as file:
            json.dump(self.report_dict, file)
            
    def report_summary(self):
        bb = {"model" : f"{type(self.trust_report.blackbox).__name__}",
            "dataset_size" : self.trust_report.dataset_size, 
            "train_test_split" : f"{self.trust_report.train_size * 100:.2f}% / {(1 - self.trust_report.train_size) * 100:.2f}%", 
            "input_features" : self.trust_report.bb_n_input_features, 
            "output_classes" : self.trust_report.bb_n_output_classes,
            "performance" : self.parse_score_report(self.trust_report._score_report(self.trust_report.y_test, self.trust_report.y_pred).split())
            }
        max_tree = {"method" : "Trustee",
            "model" : f"{type(self.trust_report.max_dt).__name__}",
            "iterations" : self.trust_report.trustee_num_iter, 
            "sample_size" : f"{self.trust_report.trustee_sample_size * 100:.2f}%", 
            "size" : self.trust_report.max_dt.tree_.node_count, 
            "depth" : self.trust_report.max_dt.get_depth(), 
            "leaves" : f"{self.trust_report.max_dt.get_n_leaves()}", 
            "input_features": f"{self.trust_report.trustee.get_n_features()} ({self.trust_report.trustee.get_n_features() / self.trust_report.bb_n_input_features * 100:.2f}%)", 
            "output_classes" : f"{self.trust_report.trustee.get_n_classes()} ({self.trust_report.trustee.get_n_classes() / self.trust_report.bb_n_output_classes * 100:.2f}%)",
            "fidelity" : self.parse_score_report(self.trust_report._score_report(self.trust_report.y_pred, self.trust_report.min_dt_y_pred).split())
            }
        min_tree = {"method": "Trustee",
            "model" : f"{type(self.trust_report.min_dt).__name__}",
            "iterations" : self.trust_report.trustee_num_iter, 
            "sample_size" : f"{self.trust_report.trustee_sample_size * 100:.2f}%", 
            "size" : self.trust_report.min_dt.tree_.node_count, 
            "depth" : self.trust_report.min_dt.get_depth(), 
            "leaves" : f"{self.trust_report.min_dt.get_n_leaves()}", 
            "input_features" : "-" , 
            "top_k" : self.trust_report.top_k, 
            "output_classes" : f"{self.trust_report.min_dt.tree_.n_classes[0]} ({self.trust_report.min_dt.tree_.n_classes[0] / self.trust_report.bb_n_output_classes * 100:.2f}%)",
            "fidelity" : self.parse_score_report(self.trust_report._score_report(self.trust_report.y_pred, self.trust_report.min_dt_y_pred).split())
            }
        
        return {
            "black_box" : bb,
            "max_tree" : max_tree,
            "min_tree" : min_tree
        }
    
    def single_run_analysis(self):
        def top_features():
            sum_nodes = 0
            sum_nodes_perc = 0
            sum_data_split = 0
            top_feature_vals = {}
            for (feat,vals) in self.trust_report.max_dt_top_features:
                node, node_perc, data_split = (
                    vals["count"],
                    (vals["count"] / (self.trust_report.max_dt.tree_.node_count - self.trust_report.max_dt.tree_.n_leaves)) * 100,
                    (vals["samples"] / self.trust_report.trustee.get_samples_sum()) * 100,
                )
                sum_nodes += node
                sum_nodes_perc += node_perc
                sum_data_split += data_split
                top_feature_vals[self.trust_report.feature_names[feat]] = {
                    "num_nodes(%)" : f"{node} ({node_perc:.2f}%)",
                    "data_split(%)" : f"{vals['samples']} ({data_split:.2f}%)"
                }
            return top_feature_vals
           
        def top_nodes():
            top_node_vals = {}
            for node in self.trust_report.max_dt_top_nodes:
                samples_by_class = [
                    (
                        self.trust_report.class_names[idx] if self.trust_report.class_names is not None and idx < len(self.trust_report.class_names) else idx,
                        (count_left / self.trust_report.max_dt.tree_.value[0][0][idx]) * 100,
                        (count_right / self.trust_report.max_dt.tree_.value[0][0][idx]) * 100,
                    )
                    for idx, (count_left, count_right) in enumerate(node["data_split_by_class"])
                ]
                samples_left = (node["data_split"][0] / self.trust_report.max_dt.tree_.n_node_samples[0]) * 100
                samples_right = (node["data_split"][1] / self.trust_report.max_dt.tree_.n_node_samples[0]) * 100
                top_node_vals[f"{self.trust_report.feature_names[node['feature']] if self.trust_report.feature_names else node['feature']} <= {node['threshold']}"] = {
                    "gini_split" : f"Left: {node['gini_split'][0]:.2f} \nRight: {node['gini_split'][1]:.2f}",
                    "data_split" : f"Left: {samples_left:.2f}% \nRight: {samples_right:.2f}%",
                    "data_split_by_class(L/R)" : "\n".join(f"{row[0]}: {row[1]:.2f}% / {row[2]:.2f}%" for row in samples_by_class)
                }
            return top_node_vals
                
        def top_branches():
            top_branch_vals = {}
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
                
                rule = ', '.join([f"{self.trust_report.feature_names[feat] if self.trust_report.feature_names else feat} {op} {threshold}" for (_, feat, op, threshold) in branch["path"]])
                top_branch_vals[rule] = {
                    "decision(P(x))" : f"{branch_class}\n({branch['prob']:.2f}%)",
                    "sample(%)" : f"{samples}\n({samples_perc:.2f}%)",
                    "class_samples" : f"{class_samples_perc:.2f}%"
                }
            return top_branch_vals
        
        return {
            f"top_{len(self.trust_report.max_dt_top_features)}_features" : top_features(),
            f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes" : top_nodes(),
            f"top_{len(self.trust_report.max_dt_top_branches)}_branches" : top_branches()
        }
        
    def prunning_analysis(self):
        def top_k_iteration():
            top_k_iter_info = {}
            for i in self.trust_report.top_k_prune_iter:
                perf = self.parse_score_report(i["score_report"].split()) 
                fide = self.parse_score_report(i["fidelity_report"].split())
                top_k_iter_info[int(i["top_k"])] = {
                    "dt_size" : i["dt"].tree_.node_count,
                    "dt_depth" : i["dt"].get_depth(),
                    "dt_n_leaves" : int(i["dt"].get_n_leaves()),
                    "performance_score" : perf,
                    "fidelity_score" : fide
                }
            return top_k_iter_info
        
        def ccp_alpha_iteration():
            ccp_alpha_iter_info = {}
            for i in self.trust_report.ccp_iter:
                perf = self.parse_score_report(i["score_report"].split()) 
                fide = self.parse_score_report(i["fidelity_report"].split())
                ccp_alpha_iter_info[i["ccp_alpha"]] = {
                    "gini" : f"{i['gini']:.3f}",
                    "dt_size" : i["dt"].tree_.node_count,
                    "dt_depth" : i["dt"].get_depth(),
                    "dt_n_leaves" : int(i["dt"].get_n_leaves()),
                    "performance_score" : perf,
                    "fidelity_score" : fide
                }
            return ccp_alpha_iter_info
        
        def max_depth_iteration():
            max_depth_iter_info = {}
            for i in self.trust_report.max_depth_iter:
                perf = self.parse_score_report(i["score_report"].split()) 
                fide = self.parse_score_report(i["fidelity_report"].split())
                max_depth_iter_info[int(i["max_depth"])] = {
                    "dt_size" : i["dt"].tree_.node_count,
                    "dt_depth" : i["dt"].get_depth(),
                    "dt_n_leaves" : int(i["dt"].get_n_leaves()),
                    "performance_score" : perf,
                    "fidelity_score" : fide
                }
            return max_depth_iter_info
        
        def max_leaves_iteration():
            max_leaves_iter_info = {}
            for i in self.trust_report.max_leaves_iter:
                perf = self.parse_score_report(i["score_report"].split()) 
                fide = self.parse_score_report(i["fidelity_report"].split())
                max_leaves_iter_info[int(i["max_leaves"])] = {
                    "dt_size" : i["dt"].tree_.node_count,
                    "dt_depth" : i["dt"].get_depth(),
                    "dt_n_leaves" : int(i["dt"].get_n_leaves()),
                    "performance_score" : perf,
                    "fidelity_score" : fide
                }
            return max_leaves_iter_info
        
        return {
            "top_k_iteration" : top_k_iteration(),
            "ccp_alpha_iteration" : ccp_alpha_iteration(),
            "max_depth_iteration" : max_depth_iteration(),
            "max_leaves_iteration" : max_leaves_iteration()
        }
    
    def repeated_run_analysis(self):
        def iterative_feature_removal():
            iterative_feat_rem_info = {}
            for i in self.trust_report.whitebox_iter:
                perf = self.parse_score_report(i["score_report"].split())
                fide = self.parse_score_report(i["fidelity_report"].split())
                iterative_feat_rem_info[i["it"]] = {
                    "feature_removed" : self.trust_report.feature_names[i["feature_removed"]] if self.trust_report.feature_names else i["feature_removed"],
                    "n_features_removed" : i["n_features_removed"],
                    "performance_score" : perf,
                    "dt_size" : i["dt"].tree_.node_count,
                    "fidelity_score" : fide
                }
            return iterative_feat_rem_info
        return {
            "iterative_feature_removal" : iterative_feature_removal()
        }
    
    def parse_score_report(self, score_report: str):
        weighted_avg_vals = score_report[score_report.index("weighted") + 2:]
        macro_avg_vals = score_report[score_report.index("macro") + 2: score_report.index("weighted")]
        accuracy_vals = score_report[score_report.index("accuracy") + 1: score_report.index("accuracy") + 3]
        class_vals = {}
        score_report_vals = score_report[4: score_report.index("accuracy")]
        n_rows = len(score_report_vals) // 5
        
        for i in range(n_rows):
            class_vals[f"{score_report_vals[i*5]}" ] = {
                "precision": score_report_vals[i * 5 + 1],
                "recall": score_report_vals[i * 5 + 2],
                "f1_score": score_report_vals[i * 5 + 3],
                "support": score_report_vals[i * 5 + 4]
            }
        return {
            "class_performance": class_vals,
            "accuracy": {
                "f1_score": accuracy_vals[0],
                "support": accuracy_vals[1]
            },
            "macro_average": {
                "precision": macro_avg_vals[0],
                "recall": macro_avg_vals[1],
                "f1_score": macro_avg_vals[2],
                "support": macro_avg_vals[3]
            },
            "weighted_average": {
                "precision": weighted_avg_vals[0],
                "recall": weighted_avg_vals[1],
                "f1_score": weighted_avg_vals[2],
                "support": weighted_avg_vals[3]
            }
        }
        
    def html(self):
        trust_report_json = {
            "summary" : self.report_summary(),
            "single_run_analysis" : self.single_run_analysis(),
            "prunning_analysis" : self.prunning_analysis(),
            "repeated_run_analysis" : self.repeated_run_analysis()
        }
        
        self.trust_report._save_dts(f"{self.output_dir}/reports", self.all_dts)
        
        max_tree = f"{self.output_dir}/reports/trust_report_dt"
        min_tree = f"{self.output_dir}/reports/trust_report_pruned_dt"

        dot_graph = graphviz.Source.from_file(max_tree)
        dot_graph.format = 'png'
        dot_graph.render(filename=f"{max_tree}")
        dot_graph = graphviz.Source.from_file(min_tree)
        dot_graph.format = 'png'
        dot_graph.render(filename=f"{min_tree}")
        
        single_run_k_values = list(trust_report_json["single_run_analysis"].keys())
        
        def score_report_html(score_report):
            temp_html = f"""
            <tr>
                <th> </th>
                <th> precision </th>
                <th> recall </th>
                <th> f1-score </th>
                <th> support </th>
            </tr>
            
            """
            
            for _class_ in score_report['class_performance']:
                temp_html += f"""
                    <tr>
                        <th style="text-align:left;"> {_class_}. {self.trust_report.class_names[int(_class_)]} </th>
                        <td> {score_report['class_performance'][_class_]['precision']} </td>
                        <td> {score_report['class_performance'][_class_]['recall']} </td>
                        <td> {score_report['class_performance'][_class_]['f1_score']} </td>
                        <td> {score_report['class_performance'][_class_]['support']} </td>
                    </tr>
                """ 
            
            output_html = f"""
                <table>
                    {temp_html}
                    <tr>
                        <th> accuracy </th>
                        <td> </td>
                        <td> </td>
                        <td> {score_report['accuracy']['f1_score']} </td>
                        <td> {score_report['accuracy']['support']} </td>
                    </tr>
                    <tr>
                        <th> macro avg </th>
                        <td> {score_report['macro_average']['precision']} </td>
                        <td> {score_report['macro_average']['recall']} </td>
                        <td> {score_report['macro_average']['f1_score']} </td>
                        <td> {score_report['macro_average']['support']} </td>
                    </tr>
                    <tr>
                        <th> weighted avg </th>
                        <td> {score_report['weighted_average']['precision']} </td>
                        <td> {score_report['weighted_average']['recall']} </td>
                        <td> {score_report['weighted_average']['f1_score']} </td>
                        <td> {score_report['weighted_average']['support']} </td>
                    </tr>
                </table>
            """
            return output_html
        
        top_k_features = ""
        for val in trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_features)}_features"]:
            top_k_features += f"""
                <tr>
                    <td> {val} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_features)}_features"][val]['num_nodes(%)']} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_features)}_features"][val]['data_split(%)']} </td>
                </tr>
            """
        
        top_k_nodes = ""
        for val in trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"]:
            top_k_nodes += f"""
                <tr>
                    <td> {val} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"][val]['gini_split']} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"][val]['data_split']} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"][val]['data_split_by_class(L/R)']} </td>
                </tr>
            """
            
        top_k_branches = ""
        for val in trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"]:
            top_k_branches += f"""
                <tr>
                    <td> {val} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"][val]['decision(P(x))']} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"][val]['sample(%)']} </td>
                    <td> {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"][val]['class_samples']} </td>
                </tr>
            """
        
        top_k_iter = ""
        for k in trust_report_json["prunning_analysis"]['top_k_iteration']:
            top_k_iter += f"""
                <tr>
                    <td> {k} </td>
                    <td> {trust_report_json["prunning_analysis"]['top_k_iteration'][k]['dt_size']} </td>
                    <td> {trust_report_json["prunning_analysis"]['top_k_iteration'][k]['dt_depth']} </td>
                    <td> {trust_report_json["prunning_analysis"]['top_k_iteration'][k]['dt_n_leaves']} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['top_k_iteration'][k]['performance_score'])} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['top_k_iteration'][k]['fidelity_score'])} </td>
                </tr>
            """
        
        ccp_alpha_iter = ""
        for alpha in trust_report_json["prunning_analysis"]['ccp_alpha_iteration']:
            ccp_alpha_iter += f"""
                <tr>
                    <td> {alpha} </td>
                    <td> {trust_report_json["prunning_analysis"]['ccp_alpha_iteration'][alpha]['gini']} </td>
                    <td> {trust_report_json["prunning_analysis"]['ccp_alpha_iteration'][alpha]['dt_size']} </td>
                    <td> {trust_report_json["prunning_analysis"]['ccp_alpha_iteration'][alpha]['dt_depth']} </td>
                    <td> {trust_report_json["prunning_analysis"]['ccp_alpha_iteration'][alpha]['dt_n_leaves']} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['ccp_alpha_iteration'][alpha]['performance_score'])} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['ccp_alpha_iteration'][alpha]['fidelity_score'])} </td>
                </tr>
            
            """
        
        max_depth_iter = ""
        for val in trust_report_json["prunning_analysis"]['max_depth_iteration']:
            max_depth_iter += f"""
                <tr>
                    <td> {val} </td>

                    <td> {trust_report_json["prunning_analysis"]['max_depth_iteration'][val]['dt_size']} </td>
                    <td> {trust_report_json["prunning_analysis"]['max_depth_iteration'][val]['dt_depth']} </td>
                    <td> {trust_report_json["prunning_analysis"]['max_depth_iteration'][val]['dt_n_leaves']} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['max_depth_iteration'][val]['performance_score'])} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['max_depth_iteration'][val]['fidelity_score'])} </td>
                </tr>
            
            """
        
        max_leaves_iter = ""
        for val in trust_report_json["prunning_analysis"]['max_leaves_iteration']:
            max_leaves_iter += f"""
                <tr>
                    <td> {val} </td>
                    <td> {trust_report_json["prunning_analysis"]['max_leaves_iteration'][val]['dt_size']} </td>
                    <td> {trust_report_json["prunning_analysis"]['max_leaves_iteration'][val]['dt_depth']} </td>
                    <td> {trust_report_json["prunning_analysis"]['max_leaves_iteration'][val]['dt_n_leaves']} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['max_leaves_iteration'][val]['performance_score'])} </td>
                    <td> {score_report_html(trust_report_json["prunning_analysis"]['max_leaves_iteration'][val]['fidelity_score'])} </td>
                </tr>
            
            """
        
        iter_feature_removal = ""
        for val in trust_report_json["repeated_run_analysis"]['iterative_feature_removal']:
            iter_feature_removal += f"""
                <tr>
                    <td> {val} </td>
                    <td> {trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['feature_removed']} </td>
                    <td> {trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['n_features_removed']} </td>
                    <td> {score_report_html(trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['performance_score'])} </td>
                    <td> {trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['dt_size']} </td>
                    <td> {score_report_html(trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['fidelity_score'])} </td>
                </tr>
            """
            
        def get_thresholds():
            threshold_res = "<h1> Thresholds </h1>"
            
            pdf_files = glob.glob(self.output_dir + "/reports/*Trustee_Threshold.pdf")
            png_files = glob.glob(self.output_dir + "/reports/*Trustee_Threshold.png")
            check = deepcopy(threshold_res)
            for (pdf, png) in zip(pdf_files, png_files):
                val_pdf = pdf[pdf.rfind("subtree"):].replace("_Trustee_Threshold.pdf", "").replace("_"," ").split()
                if len(val_pdf) == 3:
                        if val_pdf[1] == "qi":
                            threshold_res += f"""
                            <div>
                            <a href="{pdf[pdf.rfind("reports"):]}">
                            <h2> Quart Impurity: {self.trust_report.class_names[int(val_pdf[2])]} </h2>
                            <img src="{png[png.rfind("reports"):]}"></img>
                            </a>
                            </div>
                            """
                        elif val_pdf[1] == "aic":
                            threshold_res += f"""
                            <div>
                            <a href="{pdf[pdf.rfind("reports"):]}">
                                <h2> Average Impurity Change: {self.trust_report.class_names[int(val_pdf[2])]} </h2>
                                <img src="{png[png.rfind("reports"):]}"></img>
                            </a>
                            </div>
                            """
                elif len(val_pdf) == 4:
                    if val_pdf[1] == "cus":
                            
                        threshold_res += f"""
                        <div>
                        <a href="{pdf[pdf.rfind("reports"):]}">
                            <h2> Custom: {self.trust_report.class_names[int(vals[2])]}, limit:{val_pdf[3]} </h2>
                            <img src="{png[png.rfind("reports"):]}"></img>
                        </a>
                        </div>
                        """
                    elif val_pdf[1] == "full":
                        if val_pdf[2] == "qi":
                            threshold_res += f"""
                            <div>
                            <a href="{pdf[pdf.rfind("reports"):]}">
                                <h2> Full Quart Impurity: {self.trust_report.class_names[int(val_pdf[3])]} </h2>
                                <img src="{png[png.rfind("reports"):]}"></img>
                            </a>
                            </div>
                            """
                        if val_pdf[2] == "aic":
                            threshold_res += f"""
                            <div>
                            <a href="{pdf[pdf.rfind("reports"):]}">
                                <h2> Full Avg. Impurity Change: {self.trust_report.class_names[int(val_pdf[3])]} </h2>
                                <img src="{png[png.rfind("reports"):]}"></img>
                            </a>
                            </div>
                            """
            
            if threshold_res == check:
                return "No Subtrees Created"
            else:
                return threshold_res
            
    
        
        
        report = f"""
        
        <h1> Trustee Trust Report </h1>

            <a href="{"reports/trust_report_dt.pdf"}">
                <h2>Decision Tree</h2>
                <img src="reports/trust_report_dt.png"></img>
            </a>
            <a href="{"reports/trust_report_pruned_dt.pdf"}">
                <h2>Pruned Decision Tree</h2>
                <img src="reports/trust_report_pruned_dt.png"></img>
            </a>
            
                <table>
                    <tr>
                        <th colspan="3"> Trust Report Summary </th>
                    </tr>
                    <tr>
                        <th> Black Box </th>
                        <th> White Box </th>
                        <th> Top K </th>
                    </tr>
                    <tr>
                        <td> Model: {trust_report_json['summary']['black_box']['model']} </td>
                        <td> Explanation Method: Trustee </td>
                        <td> Explanation Method: Trustee </td>
                    </tr>
                    <tr>
                        <td> Dataset size: {trust_report_json['summary']['black_box']['dataset_size']} </td>
                        <td> Model: {trust_report_json['summary']['max_tree']['model']} </td>
                        <td> Model: {trust_report_json['summary']['min_tree']['model']} </td>
                    </tr>
                    <tr>
                        <td> Train/Test Split: {trust_report_json['summary']['black_box']['train_test_split']} </td>
                        <td> Iterations: {trust_report_json['summary']['max_tree']['iterations']} </td>
                        <td> Iterations: {trust_report_json['summary']['min_tree']['iterations']} </td>
                    </tr>
                    <tr>
                        <td>  </td>
                        <td> Sample size: {trust_report_json['summary']['max_tree']['sample_size']} </td>
                        <td> Sample size: {trust_report_json['summary']['min_tree']['sample_size']} </td>
                    </tr>
                    <tr> <br> </tr>
                    <tr>
                        <td> </td>
                        <th> Decision Tree Info </th>
                        <th> Decision Tree Info </th>
                    </tr>
                    <tr>
                        <td> </td>
                        <td> Size: {trust_report_json['summary']['max_tree']['size']} </td>
                        <td> Size: {trust_report_json['summary']['min_tree']['size']} </td>
                    </tr>
                    <tr>
                        <td> </td>
                        <td> Depth: {trust_report_json['summary']['max_tree']['depth']} </td>
                        <td> Depth: {trust_report_json['summary']['min_tree']['depth']} </td>
                    </tr>
                    <tr>
                        <td> </td>
                        <td> Leaves: {trust_report_json['summary']['max_tree']['leaves']} </td>
                        <td> Leaves: {trust_report_json['summary']['min_tree']['leaves']} </td>
                    </tr>
                    <tr>
                        <td> </td>
                        <td> </td>
                        <td> Top-k: {trust_report_json['summary']['min_tree']['top_k']} </td>
                    </tr>
                    <tr>
                        <td> # Input features: {trust_report_json['summary']['black_box']['input_features']} </td>
                        <td> # Input features: {trust_report_json['summary']['max_tree']['input_features']} </td>
                        <td> # Input features:  {trust_report_json['summary']['min_tree']['input_features']} </td>
                    </tr>
                    <tr>
                        <td> # Output classes: {trust_report_json['summary']['black_box']['output_classes']} </td>
                        <td> # Output classes: {trust_report_json['summary']['max_tree']['output_classes']} </td>
                        <td> # Output classes:  {trust_report_json['summary']['min_tree']['output_classes']} </td>
                    </tr>
                    <tr>
                        <th> Performance </th>
                        <th> Fidelity </th>
                        <th> Fidelity </th>
                    </tr>
                    <tr>
                        <td> {score_report_html(trust_report_json['summary']['black_box']['performance'])} </td>
                        <td> {score_report_html(trust_report_json['summary']['max_tree']['fidelity'])} </td>
                        <td> {score_report_html(trust_report_json['summary']['min_tree']['fidelity'])} </td>
                    </tr>
                </table>
                <table>
                    <tr>
                        <th colspan="3"> Single Run Analysis </th>
                    </tr>
                    <tr>
                        <td>
                            <table>
                            <tr> <th colspan="3"> {single_run_k_values[0]} </th></tr>
                            <tr>
                                <th> Feature </th>
                                <th> # of Nodes (%) </th>
                                <th> Data Split % - ↓ </th>
                            </tr>
                            <tbody>
                                {top_k_features}
                            </tbody>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <table>
                            <tr> <th colspan="4"> {single_run_k_values[1]} </th></tr>
                            <tr>
                                <th> Decision </th>
                                <th> Gini Split - ↓ </th>
                                <th> Data Split % - ↓ </th>
                                <th> Data Split % by Class (L/R) </th>
                            </tr>
                            <tbody>
                                {top_k_nodes}
                            </tbody>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <table>
                            <tr> <th colspan="4"> {single_run_k_values[2]} </th></tr>
                            <tr>
                                <th> Rule </th>
                                <th> Decision (P(x)) </th>
                                <th> Sample (%) - ↓ </th>
                                <th> Class Samples </th>
                            </tr>
                            <tbody>
                                {top_k_branches}
                            </tbody>
                            </table>
                        </td>
                    </tr>
                </table>
                <table>
                    <tr><th colspan="6"> Prunning Analysis </th></tr>
                    <tbody>
                        <tr>
                            <td>
                                <table>
                                    <tr><th colspan="6"> Trustee Top-k Iteration </th></tr>
                                    <tr>
                                        <th> k </th>
                                        <th> DT Size </th>
                                        <th> DT Depth </th>
                                        <th> DT Num Leaves </th>
                                        <th> Performance </th>
                                        <th> Fidelity </th>
                                    </tr>
                                    <tbody>
                                        {top_k_iter}
                                    </tbody>
                                </table>
                            </td>
                        </tr>
                        <tr>
                            <td>
                                <table>
                                    <tr><th colspan="7"> CCP Alpha Iteration </th></tr>
                                    <tr>
                                        <th> Alpha </th>
                                        <th> Gini </th>
                                        <th> DT Size </th>
                                        <th> DT Depth </th>
                                        <th> DT Num Leaves </th>
                                        <th> Performance </th>
                                        <th> Fidelity </th>
                                    </tr>
                                    <tbody>
                                        {ccp_alpha_iter}
                                    </tbody>
                                </table>
                            </td>
                        </tr>
                        <tr>
                            <td>
                                <table>
                                    <tr><th colspan="6"> Max Depth Iteration </th></tr>
                                    <tr>
                                        <th> Max Depth </th>
                                        <th> DT Size </th>
                                        <th> DT Depth </th>
                                        <th> DT Num Leaves </th>
                                        <th> Performance </th>
                                        <th> Fidelity </th>
                                    </tr>
                                    <tbody>
                                        {max_depth_iter}
                                    </tbody>
                                </table>
                            </td>
                        </tr>
                        <tr>
                            <td>
                                <table>
                                    <tr><th colspan="6"> Max Leaves Iteration </th></tr>
                                    <tr>
                                        <th> Max Leaves </th>
                                        <th> DT Size </th>
                                        <th> DT Depth </th>
                                        <th> DT Num Leaves </th>
                                        <th> Performance </th>
                                        <th> Fidelity </th>
                                    </tr>
                                    <tbody>
                                        {max_leaves_iter}
                                    </tbody>
                                </table>
                            </td>
                        </tr>
                    </tbody>
                </table>
                <table>
                    <tr>
                        <th colspan="6"> Repeated-run Analysis </th>
                    </tr>
                    <tr>
                        <th> Iteration </th>
                        <th> Feature Removed </th>
                        <th> # Features Removed </th>
                        <th> Performance </th>
                        <th> Decision Tree Size </th>
                        <th> Fidelity </th>
                    </tr>
                    <tbody>
                        {iter_feature_removal}
                    </tbody>
                </table>
        
        """
    
        openai.api_key = 'API KEY'
        
        def chatGPTNodes():
            data = ""
            for val in trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"]:
                data += f"""[{val}, {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"][val]['gini_split']},  {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"][val]['data_split']}, {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_nodes)}_nodes"][val]['data_split_by_class(L/R)']}]""".replace('Right', 'R').replace('Left', 'L')
            
            cnames = self.trust_report.class_names
            
            for i in cnames:
                data = data.replace(i,str(cnames.index(i)))
            
         
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system", 
                                "content": (
                                    f"""As a network security expert, you are provided with top k nodes of a decision tree model 
                                    used for classifying network traffic. Each node detail includes its decision rule, Gini split, 
                                    data split, and data split by class. Based on these details, you are tasked with identifying 
                                    instances of shortcut learning, overfitting, or spurious correlations in the model's decisions. 
                                    Moreover, you are required to assess the model's potential to generalize to out-of-distribution data.
                                    Here are the list of feature names, refer using the names to help understandability {cnames}"""
                                )
                            },
                            {
                                "role": "user", 
                                "content": f"{data}"
                            },
                        ]
                    )
            
            return response['choices'][0]['message']['content']
        
        def chatGPTFeatures():
            data = trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_features)}_features"]
            data = str(data)
          
            response = openai.ChatCompletion.create( 
                model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system", 
                                "content": (
                                    f""" In your role as a network security expert, you are given the top k 
                                    features used by a black box model that classifies network traffic. These 
                                    feature details include the number and percentage of each feature in the data, 
                                    as well as the data split by feature. Use this information to identify potential 
                                    instances of shortcut learning, overfitting, or spurious correlations in the model's 
                                    decisions. Additionally, assess the model's ability to generalize to unseen or out-of-distribution data."""
                                )
                            },
                            {
                                "role": "user", 
                                "content": f"{data}"
                            },
                        ]
                    )
            return response['choices'][0]['message']['content']
        
        def chatGPTBranches():
            data = ""
            data = str(data)
            
            for val in trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"]:
                data += f"""[{val}, {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"][val]["decision(P(x))"]}, {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"][val]["sample(%)"]}, {trust_report_json["single_run_analysis"][f"top_{len(self.trust_report.max_dt_top_branches)}_branches"][val]["class_samples"]}]"""
            
         
            response = openai.ChatCompletion.create( 
                model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system", 
                                "content": (
                                    f""" As a network security expert analyzing a black box model for classifying network 
                                    traffic, you have access to details about the top k branches used by the model. This 
                                    information includes decision rules, the number, and percentage of each feature in the 
                                    data, as well as the data split by feature. Your task is to identify potential signs of
                                    shortcut learning, overfitting, or spurious correlations in the model's decisions. 
                                    Furthermore, evaluate the model's capacity to generalize to out-of-distribution data"""
                                )
                            },
                            {
                                "role": "user", 
                                "content": f"{data}"
                            },
                        ]
                    )
            return response['choices'][0]['message']['content']
        
        def chatGPTSummary():
            data = trust_report_json["summary"]["black_box"]
            data2 = trust_report_json["summary"]["max_tree"]
            data = str(data)
            data2 = str(data2)
            
           
            response = openai.ChatCompletion.create( 
                model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system", 
                                "content": (
                                    f"""You are tasked as a network security expert to compare the performance metrics 
                                    of a black box model and a decision tree model used for network traffic classification. 
                                    Your task includes identifying potential shortcut learning, overfitting, or reliance on 
                                    spurious correlations in the models. Moreover, you need to provide recommendations on how 
                                    the models can be improved to better handle out-of-distribution data."""
                                )
                            },
                            {
                                "role": "user", 
                                "content": f"here is the black box data {data} and here is the decision tree data {data2}"
                            },
                        ]
                    )
            return response['choices'][0]['message']['content']
        
        def chatGPTRRA():
            data = ""
            for val in trust_report_json["repeated_run_analysis"]['iterative_feature_removal']:
                data += f"""{trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]["feature_removed"]}, {trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['n_features_removed']}, black box performance : {trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['performance_score']["accuracy"]}, {trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['dt_size']}, explanation fidelity : {trust_report_json["repeated_run_analysis"]['iterative_feature_removal'][val]['fidelity_score']["accuracy"]}"""
           
          
            response = openai.ChatCompletion.create( 
                model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system", 
                                "content": (
                                    f"""As part of a repeated run analysis, you iteratively remove different features from your model and 
                                    track the changes in model performance, decision tree size, and explanation fidelity. Your aim is to 
                                    uncover features that may not be necessary or may be negatively affecting your model's performance. 
                                    Analyze patterns across iterations to identify features that, when removed, maximize the black box model's 
                                    performance and subsequently increase explanation fidelity. Note that the initial accuracy of the black 
                                    box model is {trust_report_json["summary"]["black_box"]["performance"]["accuracy"]} and the fidelity 
                                    accuracy of the explanation {trust_report_json["summary"]["max_tree"]["fidelity"]["accuracy"]}. Your 
                                    insights are needed to suggest beneficial feature removal if applicable.
                                    """
                                )
                            },
                            {
                                "role": "user", 
                                "content": f" Here is the data you will analyze: {data}"
                            },
                        ]
                    )
            return response['choices'][0]['message']['content']
        
        chatGPTNodesText = ""
        chatGPTFeaturesText = ""
        chatGPTBranchesText = ""
        chatGPTSummaryText = ""
        chatGPTRRAText = ""
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future1 = executor.submit(chatGPTNodes)
                future2 = executor.submit(chatGPTFeatures)
                future3 = executor.submit(chatGPTBranches)
                future4 = executor.submit(chatGPTSummary)
                future5 = executor.submit(chatGPTRRA)

                chatGPTNodesText = future1.result()
                chatGPTFeaturesText = future2.result()
                chatGPTBranchesText = future3.result()
                chatGPTSummaryText = future4.result()
                chatGPTRRAText = future5.result()
        except:
            print("chatGPT Error")
        
        html_template = f"""
        <!DOCTYPE html>
        <html>      
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width">
                <title> Trustee2HTML </title>
                <link href="style.css" rel="stylesheet" type="text/css" />
                <style>
                
                img {"{"}
                    padding: 2.5px;
                    max-width: 90vw;
                    max-height: 40vh;
                {"}"}
                
                th {"{"}
                    border: 1.5px solid black;
                    background-color: #e2e1e1;;
                    padding: 2.5px;
                    text-align: center;
                    
                {"}"}

                td {"{"}
                    border: 1px solid black;
                    padding: 2.5px;
                    text-align: center;
                {"}"}

                table table {"{"}
                margin: 0 auto;
                {"}"}
                
                .navbar button {"{"}
                     margin: 10px;
                {"}"}
                
                .content {"{"}
                    width: 100%;
                    padding: 15px;
                    box-sizing: border-box;
                {"}"}
                
                </style>
            </head>
            <body>

                <div class="navbar">
                    <button onclick="show('trustReport')">Trust Report</button>
                    <button onclick="show('thresholds')">Thresholds</button>
                    <button onclick="show('gptReport')">Chat-GPT Report</button>
                </div>

                <div id="trustReport" class="content"> 
                {report} 
                </div>
                
                <div id="thresholds" style="display:flex; flex-wrap: wrap;" class="content">
                {get_thresholds()}
                </div>
                
                <div id="gptReport" style="display:none" class="content">
                <h1>Chat GPT Report</h1>
                <h2> Summary </h2>
                {chatGPTSummaryText} 
                <br>
                <h2> Top k Nodes </h2>
                {chatGPTNodesText}
                <br>
                <h2> Top k Features </h2>
                {chatGPTFeaturesText}
                <br>
                <h2> Top k Branches </h2>
                {chatGPTBranchesText}
                <h2> Repeated Run Analysis </h2>
                {chatGPTRRAText}
                </div>
            </body>
            
            <script>

                function show(id) {"{"}
                var contents = document.getElementsByClassName('content');
                for (var i = 0; i < contents.length; i++) {"{"}
                    contents[i].style.display = 'none';
                {"}"}
                document.getElementById(id).style.display = 'block';
                {"}"}
                
            </script>
            
        </html>
        """
        

            
        output_file_path = self.output_dir + '/trust_report.html'
        print(output_file_path)
        with open(output_file_path, 'w') as f:
            f.write(html_template)
