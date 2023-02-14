"""
ClassificationTrustee
=====================
Simple example on how to use the ClassificationTrustee class to extract
a decision tree from a RandomForestClassifier from scikit-learn.
"""
# importing required libraries
# importing Scikit-learn library and datasets package

import graphviz
from sklearn import tree
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from trustee.utils.tree import get_dt_dict

from trustee import ClassificationTrustee

# Loading the iris plants dataset (classification)
iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y=True)

# Spliting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators=100)
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# Evaluate model accuracy
model_classification_report = classification_report(y_test, y_pred)

# Initialize Trustee and fit for classification models
trustee = ClassificationTrustee(expert=clf)
trustee.fit(X_train, y_train, num_iter=50, num_stability_iter=10, samples_size=0.3, verbose=True)

# Get the best explanation from Trustee
dt, pruned_dt, agreement, reward = trustee.explain()

explanation_training = f"{agreement}, {reward})"
explanation_size = f"{dt.tree_.node_count}"
pruned_explanation_size = f"Top-k Prunned Model explanation size: {pruned_dt.tree_.node_count}"


# Use explanations to make predictions
dt_y_pred = dt.predict(X_test)
pruned_dt_y_pred = pruned_dt.predict(X_test)

# Evaluate accuracy and fidelity of explanations
global_fidelity_report = classification_report(y_pred, dt_y_pred)
topk_global_fidelity_report = classification_report(y_pred, pruned_dt_y_pred)
score_report = classification_report(y_test, dt_y_pred)
topk_score_report = classification_report(y_test, pruned_dt_y_pred)


# Output decision tree to pdf
dot_data = tree.export_graphviz(
    dt,
    class_names=iris.target_names,
    feature_names=iris.feature_names,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("media/dt_explanation")


# Output pruned decision tree to pdf
dot_data = tree.export_graphviz(
    pruned_dt,
    class_names=iris.target_names,
    feature_names=iris.feature_names,
    filled=True,
    rounded=True,
    special_characters=True,
)

TARGET_NAMES = iris.target_names
FEATURE_NAMES = iris.feature_names

graph = graphviz.Source(dot_data)
graph.render("media/pruned_dt_explanation")

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from pdf2image import convert_from_path

focus = []

def create_tr_row(tab, row,  text_left, text_right, bd=0, pady=(0,0)):
    label_color = "#383838"
    left_label = Label(tab, text=text_left)
    left_label.grid(row=row, column=0, sticky="WE", pady=pady)
    right_label = Label(tab, text=text_right, bd=bd, relief="solid", bg=label_color)
    right_label.grid(row=row, column=1, sticky="NWE", padx=(0, 5), pady=pady)
    return left_label, right_label

def on_double_click_target(event):
    widget = event.widget
    index = widget.curselection()
    if index:
        value = widget.get(index)
        widget.delete(index)
        my_listbox3.insert(END, f"Target: {value}")
def on_double_click_feature(event):
    widget = event.widget
    index = widget.curselection()
    if index:
        value = widget.get(index)
        widget.delete(index)
        my_listbox3.insert(END, f"Feature: {value}")

def on_double_click_focus(event):
    widget = event.widget
    index = widget.curselection()
    if index:
        value = widget.get(index)
        widget.delete(index)
        if value.startswith("Feature: "):
            my_listbox2.insert(END, value.replace("Feature: ", ""))
        elif value.startswith("Target: "):
            my_listbox.insert(END, value.replace("Target: ", ""))

root = Tk()
root.title("Trustee")

# Create a notebook with tabs
notebook = ttk.Notebook(root)
notebook.pack(fill=BOTH, expand=True)

# Create the first tab and display the image from the PDF file
tab1 = Frame(notebook)
right_frame = Frame(tab1)
canvas = Canvas(right_frame, width=600, height=600)
pages = convert_from_path('media/dt_explanation.pdf', dpi=200, size=(None, None), grayscale=True)
image = pages[0]
image.thumbnail((600, 600))
photo = ImageTk.PhotoImage(image)
canvas.create_image(0, 0, anchor=NW, image=photo)
canvas.pack(side=TOP, fill=BOTH)
right_frame.pack(side=RIGHT, fill=BOTH)
notebook.add(tab1, text="DT")

list_frame = Frame(tab1)
list_frame.pack(side=LEFT, fill=BOTH)

features_label = Label(list_frame, text="Features")
features_label.pack(side=TOP, fill=BOTH)

my_listbox2 = Listbox(list_frame)
my_listbox2.pack(side=TOP, fill=BOTH)
for item in FEATURE_NAMES:
    my_listbox2.insert(END, item)

target_label = Label(list_frame, text="Targets")
target_label.pack(side=TOP, fill=BOTH)

my_listbox = Listbox(list_frame)
my_listbox.pack(side=TOP, fill=BOTH)
for item in TARGET_NAMES:
    my_listbox.insert(END, item)
    
focus_label = Label(list_frame, text="Focus")
focus_label.pack(side=TOP, fill=BOTH)

my_listbox3 = Listbox(list_frame)
my_listbox3.pack(side=TOP, fill=BOTH)
for item in focus:
    my_listbox3.insert(END, item)

my_listbox.bind('<Double-Button-1>', on_double_click_target)
my_listbox2.bind('<Double-Button-1>', on_double_click_feature)
my_listbox3.bind('<Double-Button-1>', on_double_click_focus)

under_image = Frame(right_frame)
under_image.pack(side=TOP, fill=BOTH)

under_image_l = Frame(under_image)
under_image_l.pack(side=LEFT, fill=BOTH, expand=TRUE)

under_image_r = Frame(under_image)
under_image_r.pack(side=RIGHT, fill=BOTH, expand=TRUE)

n_picker_label = Label(under_image_l, text="n-count")
n_picker_label.pack(side=TOP, fill=BOTH, expand= TRUE)

n_picker = Spinbox(under_image_l, from_=1, to=explanation_size, width=5)
n_picker.pack(side=TOP, fill=BOTH, expand=TRUE)

threshold = Label(under_image_r, text="gini threshold")
threshold.pack(side=TOP, fill=BOTH, expand = TRUE)

threshold = Spinbox(under_image_r, from_=0, to=1, increment=0.001, width=5)
threshold.pack(side=TOP, fill=BOTH , expand=TRUE)

create_button = Button(right_frame, text="generate")
create_button.pack(side=TOP, fill=BOTH)
# Create the second tab and display a different image
tab2 = Frame(notebook)
pages2 = convert_from_path('media/pruned_dt_explanation.pdf', dpi=200, size=(None, None), grayscale=True)
image2 = pages[0]
image2.thumbnail((500, 500))
photo2 = ImageTk.PhotoImage(image2)
canvas2 = Canvas(tab2, width=500, height=500)
canvas2.create_image(0, 0, anchor=NW, image=photo2)
canvas2.pack()
notebook.add(tab2, text="Pruned DT")

# Create the third tab and display the data from the print statements
tab3 = Frame(notebook)
tab3.columnconfigure(0, weight=1)
tab3.columnconfigure(1, weight=1)

agreement_label_l, agreement_label_r = create_tr_row(tab3, 0,"Agreement:", f"{agreement}", pady=(5,0))
fidelity_label_l, fidelity_label_r = create_tr_row(tab3, 1,"Fidelity:", f"{reward}" )
exp_size_label_l, exp_size_label_r = create_tr_row(tab3, 2,"Model explanation size:", explanation_size)
glbl_fid_label_l, glbl_fid_label_r = create_tr_row(tab3, 3,"Global Fidelity report:", global_fidelity_report, pady=(0,5))
tk_glbl_fid_label_l, tk_glbl_fid_label_r = create_tr_row(tab3, 4,"Top-K Global Fidelity report:", topk_global_fidelity_report, pady=(0,5))
scr_rpt_l, scr_rpt_r = create_tr_row(tab3, 5, "Score Report:", score_report, pady=(0,5))
tk_scr_rpt_l, tk_scr_rpt_r = create_tr_row(tab3, 6, "Top-K Score Report:", topk_score_report, pady=(0,5))

notebook.add(tab3, text="Trust Report")

root.mainloop()


