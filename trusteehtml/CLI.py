import webbrowser
import os
from jinja2 import Template
from trustee.report.trust import TrustReport
from trusteehtml.Threshold import Thresholder
from trusteehtml.htmlCreator import htmlCreator
from trusteehtml.subtree import get_subtree
import graphviz as giz

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
            'cus (target class) (custom threshold)' custom : till custom threshold value inputed by the custom_threshold param
            'full (target class)' The full tree
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
                    if self.explanation_path == "":
                        # create the html file and open it in the web browser
                        filename = 'file:///'+os.getcwd() +"/" +self.output_directory + "/output.html"
                        self.html_creator.convert_to_html()
                        print("building html file...")
                        print(f"opening {filename} in web browser... \n")
                        webbrowser.open_new_tab(filename)
                        self.explanation_path = filename
                    else:
                        # open the previously created html file in the web browser
                        print("opening trust report... \n")
                        webbrowser.open_new_tab(self.explanation_path)
                elif command[0] == "help":
                    print(self.guide)
                elif command[0] == "chatGPT":
                    print(self.thresholder.paths)
                    
            elif len(command) == 2 and command[0] == 'target':
                if command[1].isdigit():
                    # find all paths to the target class with the given index
                    print(self.thresholder.find_paths_to_class(int(command[1])))
                elif command[1] == 'all':
                    # find all paths to all target classes
                    self.thresholder.all_target_paths()
                    
            # Thresholding
            elif len(command) == 2 and command[0] == "qi" and command[1].isdigit():
                get_subtree(self.thresholder.trust_report.max_dt, int(command[1]) ,self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}")
            elif len(command) == 2 and command[0] == "aic" and command[1].isdigit():
                get_subtree(self.thresholder.trust_report.max_dt, int(command[1]) ,self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names, threshold="avg imp change").render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}")
            elif len(command) == 3 and command[0] == "cus" and command[1].isdigit() and command[2].isdigit():
                get_subtree(self.thresholder.trust_report.max_dt, int(command[1]) ,self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names, threshold="custom", custom_threshold=int(command[2])).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}")
            elif len(command) == 2 and command[0] == "full" and command[1].isdigit():
                get_subtree(self.thresholder.trust_report.max_dt, int(command[1]) ,self.thresholder.trust_report.class_names, self.thresholder.trust_report.feature_names, full_tree=True).render(filename=os.getcwd() +"/" +self.output_directory + f"/reports/subtree_{command[0]}_{command[1]}")
                    
                