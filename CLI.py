import webbrowser
import os
from jinja2 import Template
from trustee.report.trust import TrustReport
from Threshold import Thresholder
from htmlCreator import htmlCreator

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
                    
            elif len(command) == 2 and command[0] == 'target':
                if command[1].isdigit():
                    # find all paths to the target class with the given index
                    print(self.thresholder.find_paths_to_class(int(command[1])))
                elif command[1] == 'all':
                    # find all paths to all target classes
                    self.thresholder.all_target_paths()
