import webbrowser
import os
from jinja2 import Template
from trustee.report.trust import TrustReport
from trusteehtml.Threshold import Thresholder
from trusteehtml.htmlCreator import htmlCreator
from trusteehtml.subtree import get_subtree
import graphviz as giz
import requests
import json
import openai

class CLIController:
    def __init__(self, trust_report, output_directory) -> None:
        """
        Initializes the CLIController class.

        Args:
        trust_report (TrustReport): A TrustReport object that contains the trust report data.
        
        """
        self.guide = """
            'q' to quit 
            'open' to open trustee explanation to web browser
            'target (class index)' to list all branches to that target class
            'target all' to list all branches to all target classes
            'qi (target class)' quart impurity : till the first ancestor with gini value above 25%
            'aic (target class)' average impurity change : avg all the imp change in the branches then print all nodes with less than the avg
            'cus (target class) (custom threshold)' custom : till custom threshold value inputted by the custom_threshold param
            'full (threshold type) (target class)' The full tree for whatever type qi or aic
            """
        self.thresholder = Thresholder(trust_report)
        self.output_directory = output_directory
        self.html_creator = htmlCreator(trust_report, output_directory)
        self.explanation_path = "" 
        
    def run(self):
        """
        Runs the CLIController and allows the user to input commands and interact with the program.
        """
        print(self.guide)
        while True:
            user_input = input()
            command = user_input.split()
            
            if len(command) == 1:
                if command[0] in ['quit', 'q']:
                    break
                elif command[0] == "open":
                    # create the html file and open it in the web browser
                    filename = 'file:///'+os.getcwd() +"/" +self.output_directory + "/output.html"
                    self.html_creator.convert_to_html()
                    print("building html file...")
                    print(f"opening {filename} in web browser... \n")
                    webbrowser.open_new_tab(filename)
                    self.explanation_path = filename
                elif command[0] == "help":
                    print(self.guide)
                elif command[0] == "chatGPT":
                    
                    _, w, _ = self.html_creator.trust_report_summary(trust_report=self.thresholder.trust_report, for_json=True)
                    b, wb, _ = self.html_creator.summary_performance(trust_report=self.thresholder.trust_report, for_json=True)
                    print("Purpose of this model:")
                    inp = input()
                    def getType():
                        try:
                            return type(self.thresholder.trust_report.blackbox).__name__
                        except:
                            return "Un-Identified"
                            
                    prompt = f"""
                    
Request for Black Box Model Analysis Using a White Box Decision Tree Representation:
I have a black box model, and I've created a white box decision tree representation of it to analyze its performance and identify any weaknesses. I'd like your assistance in analyzing the black box model based on the information provided below: 
Black Box Model Details:
    Model Type: {getType()}
    Dataset size: {self.thresholder.trust_report.dataset_size}
    Training Split: {f"{self.thresholder.trust_report.train_size * 100:.2f}% / {(1 - self.thresholder.trust_report.train_size) * 100:.2f}%"}
    Input feature count: {self.thresholder.trust_report.bb_n_input_features}
    output class count: {self.thresholder.trust_report.bb_n_output_classes}
Black Box Model Performance: - Overall performance metrics: 
    accuracy: {b["accuracy"]}
    macro average: {b["macro_avg"]}
    weighted average: {b["weighted_avg"]} 
Class-specific Performance Metrics for the Black Box Model:
    {b["class_performance_data"]}
Top k Features Identified by the White Box Model: 
    {self.thresholder.trust_report.max_dt_top_features}
Top k Nodes Identified by White Box Model:
    {self.thresholder.trust_report.max_dt_top_nodes}
Top k Branches Identified by White Box Model:
    {self.thresholder.trust_report.max_dt_top_branches}
White Box Model Structure Information: 
    Size: {w["size"]}
    Depth: {w["depth"]}
    Leaves: {w["leaves"]}
White Box Model Overall Fidelity Scores:
    accuracy: {wb["accuracy"]}
    macro average: {wb["macro_avg"]}
    weighted average: {wb["weighted_avg"]} 
Please provide detailed solutions and suggestions for improving the performance of a machine learning model trained for {inp}. In your response, identify any possible shortcuts, discuss ways to avoid overfitting, and recommend techniques for optimizing model performance. Make sure to include best practices and strategies relevant to the specific problem and domain.
                    """
                    
                    print(prompt)
                    
                    
                    
                    
            elif len(command) == 2 and command[0] == 'target':
                if command[1].isdigit():
                    # find all paths to the target class with the given index
                    print(self.thresholder.find_paths_to_class(int(command[1])))
                elif command[1] == 'all':
                    # find all paths to all target classes
                    self.thresholder.all_target_paths()
                    
            # Thresholding
            elif len(command) == 2 and command[0] == "qi" and command[1].isdigit():
                get_subtree(self.thresholder.trust_report.max_dt, int(command[1]) ,self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_lab56", format="pdf")
            elif len(command) == 2 and command[0] == "aic" and command[1].isdigit():
                get_subtree(self.thresholder.trust_report.max_dt, int(command[1]) ,self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names, threshold="avg imp change").render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_lab56", format="pdf")
            elif len(command) == 3 and command[0] == "cus" and command[1].isdigit() and command[2].isdigit():
                get_subtree(self.thresholder.trust_report.max_dt, int(command[1]) ,self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names, threshold="custom", custom_threshold=int(command[2])).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_lab56", format="pdf")
            elif len(command) == 3 and command[0] == "full" and command[2].isdigit():
                if command[1] == "qi":
                    get_subtree(self.thresholder.trust_report.max_dt, int(command[2]),self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names,threshold="qi" ,full_tree=True).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_lab56", format="pdf")
                elif command[1] == "aic":
                    get_subtree(self.thresholder.trust_report.max_dt, int(command[2]),self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names,threshold="aic" ,full_tree=True).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}_{command[2]}_lab56", format="pdf")
                