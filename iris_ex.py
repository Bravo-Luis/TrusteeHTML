from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from trustee.report.trust import TrustReport
import os
import sys
from CLI import CLIController

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

controller = CLIController(trust_report, "iris")
controller.run()
