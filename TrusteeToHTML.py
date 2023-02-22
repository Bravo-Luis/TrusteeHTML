from jinja2 import Template
from trustee.report.trust import TrustReport   
from bs4 import BeautifulSoup 
    
class htmlCreator:
    def __init__(self, trust_report : TrustReport) -> None:
        self.trust_report = trust_report
        
    def convert_to_html(self, input_file_name = "template.html", output_file_name = "output.html") -> None:
        
        with open(input_file_name, "r") as file:
            template = Template(file.read())
        
        self.trust_report._save_dts("reports", False)
            
        html = template.render(decision_tree="reports/trust_report_dt.pdf", pruned_decision_tree="reports/trust_report_pruned_dt.pdf")
        
        with open(output_file_name, "w") as file:
            file.write(html)
    
    
