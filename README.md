Want to create a branch of trustee where we can put these files into trustee.report.trust 

These package improvements:
-  HTML Trust Report
-  JSON Trust Report
-  Adds cli commands to create subtrees 
-  Trees and subtrees displayed as PNG images and can be open/saved as pdf images
-  Uses OpenAI chatGPT API to try to analyze and suggest action based on data from explanation and trust report data


how to use: Iris example

'''
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from trustee.report.trust import TrustReport
from trustee.report.threshold import TrusteeThreshold
from trustee.report.format import TrusteeFormatter
import os


iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y=True, as_frame=True)

clf = RandomForestClassifier(n_estimators=100)

data = TrustReport(
    clf,
    X=X,
    y=y,
    max_iter=5,
    num_pruning_iter=5,
    train_size=0.7,
    trustee_num_iter=10,
    trustee_num_stability_iter=5,
    trustee_sample_size=0.3,
    analyze_branches=True,
    analyze_stability=True,
    top_k=10,
    verbose=True,
    class_names=iris.target_names,
    feature_names=iris.feature_names,
    is_classify=True,
)
print(data)
dir = 'iris/'

try:
       os.mkdir(dir)
except: 
       pass

test = TrusteeThreshold(data, dir)
test.run()
'''

with this way you are able to use command line commands to create subtrees


Iris example 2:
'''
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from trustee.report.trust import TrustReport
from trustee.report.threshold import TrusteeThreshold
from trustee.report.format import TrusteeFormatter
import os


iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y=True, as_frame=True)

clf = RandomForestClassifier(n_estimators=100)

trust_report = TrustReport(
    clf,
    X=X,
    y=y,
    max_iter=5,
    num_pruning_iter=5,
    train_size=0.7,
    trustee_num_iter=10,
    trustee_num_stability_iter=5,
    trustee_sample_size=0.3,
    analyze_branches=True,
    analyze_stability=True,
    top_k=10,
    verbose=False,
    class_names=iris.target_names,
    feature_names=iris.feature_names,
    is_classify=True,
)
dir = 'iris/'

try:
       os.mkdir(dir)
except: 
       pass

test = TrusteeFormatter(trust_report=trust_report, output_dir=dir)
test.json()
test.html()
'''
This one just creates the json and html code
