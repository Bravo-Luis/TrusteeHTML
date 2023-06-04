import numpy as np
from trustee.utils.tree import get_dt_dict
import graphviz as giz


def get_subtree(dt, target_class, class_labels, features, threshold = "quart impurity", full_tree = False, custom_threshold = 3, k = 2):
    """

    :param dt:
    :param target_class: The class you seek to find
    :param class_labels: list of class labels
    :param features: list of features
    :param threshold: Threshold types = ["quart impurity", "custom", "avg imp change", "full tree"]
            quart impurity: till the first ancestor with gini value above 25%
            custom: till custom threshold value inputed by the custom_threshold param
            avg imp change: avg all the imp change in the branches then print all nodes with less than the avg change
            full tree: output the whole subtree
            ______height:


    :param full_tree: You want the full tree or not?
    :param custom_threshold: if threshold type if custom, input the custom threshold amount
    :param k : number of max subtree
    :return: N/A
    """
    
    # Threshold types = ["quart impurity", "custom", "avg imp change", "full tree"]
    # change tree to dict
    dict_dt = get_dt_dict(dt)
    # node details
    # print(f"{dict_dt=}")
    nodes = dict_dt["nodes"]

    class_values = dict_dt["values"]
    node_class = {}
    prev = {}
    child = {}
    # For target node indices
    targ = [-1]
    # print(f"{class_values=}")
    samp = []


    for i in range(len(nodes)):
        # find the class of each node by the max value index
        node_class[i] = np.argmax(class_values[i])
        # if leaf
        if nodes[i][0] == -1 and nodes[i][1] == -1:
            # if leaf and target class
            if node_class[i] == target_class:
                ###############

                ###############
                # save the target node index
                # if target leaf nodes are found
                if  targ != [-1]:
                    targ.append(i)
                    samp.append(nodes[i][5])
                # if no target leaf nodes are found
                else:
                    targ = [i]
                    samp.append(nodes[i][5])

        else:
            # save the index of the left and right child
            child[i] = [nodes[i][0], nodes[i][1]]
    # save parent of nodes for backtracking
    for key, value in child.items():
        for kid in value:
            prev[kid] = key

    graphing = giz.Digraph(f"subtree", strict=True)
    # sub_tree.append(nodes[targ[0]])
    # k = targ[0]
    # Function to walk through all possible target leaf

    samp_and_targets = []
    for l in range(len(targ)):
        samp_and_targets.append([samp[l], targ[l]])
    samp_and_targets.sort(reverse=True)
    print(f"{samp_and_targets=}")

    top_K_targ = []
    if k > len(samp_and_targets):
        k = len(samp_and_targets)
    for p in range(k):
        print(f"{samp_and_targets[p]=}")
        top_K_targ.append(samp_and_targets[p][1])

    def walk_back (index, threshold, full_tree):
        sub_tree = []
        # start by appending target leaf node and its index
        # sub_tree.append((nodes[targ[index]], targ[index]))
        # k = targ[index]

        sub_tree.append((nodes[top_K_targ[index]], top_K_targ[index]))
        k = top_K_targ[index]

        # if threshold == -1:


        # while node has a parent
        while k in prev.keys():
            k = prev[k]
            sub_tree.append((nodes[k], k))

        # print(f"{sub_tree=}")
        height = len(sub_tree)

        #######################
        #graphing = giz.Digraph("subtree")
        # graphing.attr("node", shape="box")
        info = ""
        ######################
        if full_tree == False:
            if threshold == "quart impurity":

                i= 0
                quart = []
                while i < len(sub_tree) and sub_tree[i][0][4] <= 0.25:
                    quart.append(sub_tree[i])
                    i += 1
                if len(quart) < 2:
                    quart.append(sub_tree[1])
                sub_tree = quart
            elif threshold == "custom":
                if custom_threshold < len(sub_tree):
                    sub_tree = sub_tree[:custom_threshold+1]
                else:
                    full_tree = True
            elif threshold == "avg imp change":
                avg_imp_change = 0
                siz = len(sub_tree) - 1
                imp_change_list = []
                for i in range(siz):
                    avg_imp_change += abs((sub_tree[i][0][4] - sub_tree[i+1][0][4])/(siz))
                for i in range(siz):
                    # print(f"{i=}\n{imp_change_list=}\n{avg_imp_change=}\n{sub_tree=}")
                    if abs(sub_tree[i][0][4] - sub_tree[i+1][0][4]) < avg_imp_change:
                        if i == 0:
                            imp_change_list.append(sub_tree[i])
                            imp_change_list.append(sub_tree[i+1])
                        else:
                            imp_change_list.append(sub_tree[i+1])
                    elif i == 0:
                        imp_change_list.append(sub_tree[i])
                        imp_change_list.append(sub_tree[i + 1])
                    else:
                        break


                # graphing.node("information", )
                info = f"The average impurity change is {round(avg_imp_change*100, 2)}%.\n"
                print(f"The average impurity change is {round(avg_imp_change*100, 2)}%. ")




                sub_tree = imp_change_list
            elif threshold == "full tree":
                full_tree = True
            else:
                info += f"*+*+*+*+*+*+*+ {threshold} is not a valid threshold type!*+*+*+*+*+*+*+\n"
                print(f"*+*+*+*+*+*+*+ {threshold} is not a valid threshold type!*+*+*+*+*+*+*+")
                return
        # print(f"{sub_tree=}")
        # reverse subtree to start from root
        sub_tree = sub_tree[::-1]
        # else:
        #     # Threshold not implemented yet
        #     while k in prev.keys() and threshold > 0:
        #         k = prev[k]
        #         sub_tree.append((nodes[k], k))
        #         threshold -= 1

        # print(prev)
        # print(child)
        # print(sub_tree)
        # Details being outputted
        details = ["feature", "threshold", "impurity", "samples", "weighted_samples"]
        info += f"Target class is {class_labels[target_class]}.\n" +f"The height of the Target leaf is {height}.\n" + f"The Threshold type is {threshold}.\n"
        print(f"Target class is {class_labels[target_class]}.")
        print(f"The height of the Target leaf is {height}.")
        print(f"The Threshold type is {threshold}.")
        print("+++++++++++++++")

        graphing.node(f"information{index}" , label=info , shape= "box")

        # Keep track of family line
        ancestor = len(sub_tree) - 2

        ishead = False
        for path in range(len(sub_tree)):

            det = ""
            # If its not the target leaf node
            if path != (len(sub_tree) - 1):
                # Top of the tree
                if path == 0:
                    if full_tree == True:
                        print("***Root***")
                    elif path == len(sub_tree) - 2:
                        print(f"***Parent***")
                    else:
                        print(f"***Ancestor {ancestor}***")
                        ancestor -= 1
                    ishead = True


                # Parent of target leaf node
                elif path == len(sub_tree) - 2:
                    print(f"***Parent***")
                    ishead = False

                # Other ancestors
                else:
                    print(f"***Ancestor {ancestor}***")
                    ishead = False
                    ancestor -= 1

                # If not head then print the branching logic
                # print(f"{path=}")
                if not ishead:
                    logic = '<=' if (child[sub_tree[path - 1][1]][0] == sub_tree[path][1]) else '>'
                    print(f"Branching from previous node logic = \'{'<=' if (child[sub_tree[path - 1][1]][0] == sub_tree[path][1]) else '>'}'")

                # Print all the feature details for node
                for p in range(2,len(sub_tree[path][0]) - 1):
                    # if feature print the name of the feature
                    if details[p - 2] != "feature":
                        if details[p - 2] == "threshold":
                            thres = f" {round(sub_tree[path][0][p],2)}"
                        det += f"{details[p - 2]} = {round(sub_tree[path][0][p], 2)}\n"
                        print(f"{details[p - 2]} = {round(sub_tree[path][0][p],2)}")
                    else:
                        det += f"{details[p - 2]} = {features[sub_tree[path][0][p]]}\n"
                        print(f"{details[p - 2]} = {features[sub_tree[path][0][p]]}")
                        feat = f"{features[sub_tree[path][0][p]]} "

                graphing.node(f"node{sub_tree[path]}", label=det, shape = "ellipse",style='filled', fillcolor='#ff000042')
                if path != 0:
                    graphing.edge(f"node{sub_tree[path-1]}",f"node{sub_tree[path]}", label=feat +logic + thres , len='1.0')
                # else:
                #     graphing.edge(f"information{index}", f"node{sub_tree[path]}", len='0.8')
                det = ""


                print("++++++++++")


            # If its the target node
            else:
                det = f"***{class_labels[node_class[targ[index]]]} Leaf***\n"
                logic = '<=' if (child[sub_tree[path - 1][1]][0] == targ[index]) else '>'
                print(f"***{class_labels[node_class[targ[index]]]} Leaf***")
                print(f"Branching from previous node logic = \'{'<=' if (child[sub_tree[path - 1][1]][0] == targ[index]) else '>'}'")
                for p in range(2,len(sub_tree[path][0]) - 1):
                    # if feature print the name of the feature
                    if details[p - 2] != "feature":
                        if details[p - 2] == "threshold":
                            thres = f" {round(sub_tree[path][0][p],2)}"
                        det += f"{details[p - 2]} = {round(sub_tree[path][0][p],2)}\n"
                        print(f"{details[p - 2]} = {round(sub_tree[path][0][p],2)}")
                    else:
                        det += f"{details[p - 2]} = {features[sub_tree[path][0][p]]}\n"
                        print(f"{details[p - 2]} = {features[sub_tree[path][0][p]]}")
                        feat = f"{features[sub_tree[path][0][p]]} "
                    graphing.edge(f"node{sub_tree[path]}", f"information{index}",  len='0.8')

                # Print class name of target node
                det += f"class_Name = {class_labels[node_class[targ[index]]]}"
                print(f"class_Name = {class_labels[node_class[targ[index]]]}")
                print("++++++++++")
                graphing.node(f"node{sub_tree[path]}", label=det, shape="ellipse", style='filled', fillcolor='#40e0d0')
                if path != 0:
                    graphing.edge(f"node{sub_tree[path-1]}",f"node{sub_tree[path]}", label=feat +logic + thres , len='1.0')
                # else:
                #     graphing.edge(f"information{index}", f"node{sub_tree[path]}", len='0.8')
                det = ""

        # graphing.view()

    # print(f"{len(samp)=}")
    # print(f"{targ=}")


    # for each target leaf node
    for targets_ in range(len(top_K_targ)):
        print(f"*****************Sub-tree {targets_}*****************")
        walk_back(targets_, threshold, full_tree)

    return graphing