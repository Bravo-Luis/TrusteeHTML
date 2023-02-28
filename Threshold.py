from trustee.report.trust import TrustReport

class Thresholder:
    """
    A class that provides methods to retrieve paths from the trust report to 
    target classes or all target classes.
    """
    
    def __init__(self, trust_report: TrustReport) -> None:
        """
        Constructor method for the Thresholder class.
        
        Args:
        trust_report (TrustReport): A TrustReport object that contains the 
                                     trust report data.
        """
        self.trust_report = trust_report
        self.paths = trust_report.max_dt_all_branches
        self.sorted_paths = None
        
    def find_target_classes(self, target_class: int) -> list:
        """
        Finds paths in the trust report that lead to the specified target class.
        
        Args:
        target_class (int): The index of the target class to search for.
        
        Returns:
        list: A list of the paths that lead to the target class.
        """
        target_paths = []
        for path in self.paths:
            if path['class'] == target_class:
                target_path = path['path']
                target_paths.append(target_path)
        return target_paths
        
    def find_paths_to_class(self, target_class: int) -> str:
        """
        Returns a formatted string containing all the paths in the trust report 
        that lead to the specified target class.
        
        Args:
        target_class (int): The index of the target class to search for.
        
        Returns:
        str: A formatted string containing the paths to the target class.
        """
        target_paths = self.find_target_classes(target_class)
        
        if len(target_paths) == 0:
            return f"NO CLASSES OF INDEX {target_class} \n"
                
        def convert_output(path_list_list):
            result = ""
            for j, list_of_nodes in enumerate(path_list_list):
                branch_str = f"     branch {j+1}: "
                for i, node in enumerate(list_of_nodes):
                    if i == len(list_of_nodes) - 1:
                        branch_str += f"(FEATURE: { self.trust_report.feature_names[node[0]]} {node[2]} {node[3]}) \n"
                    else:
                        branch_str += f"(FEATURE: {self.trust_report.feature_names[node[0]]} {node[2]} {node[3]}) AND "
                result += branch_str
                
            return f"\nTARGET: {self.trust_report.class_names[target_class]} \n" + result
        
        return convert_output(target_paths)
    
    def all_target_paths(self) -> None:
        """
        Prints the formatted string containing all the paths to every target 
        class in the trust report.
        """
        for i in range(len(self.trust_report.class_names)):
            print(self.find_paths_to_class(i))
