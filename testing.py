from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from trustee.report.trust import TrustReport
from TrusteeToHTML import htmlCreator

    # Loading the iris plants dataset (classification)
iris = datasets.load_iris()
    # dividing the datasets into two parts i.e. training datasets and test datasets
X, y = datasets.load_iris(return_X_y=True, as_frame=True)

    # creating a RF classifier
clf = RandomForestClassifier(n_estimators=100)

    # The trust report (can) fit and explain the classifier
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
    verbose=True,
    class_names=iris.target_names,
    feature_names=iris.feature_names,
    is_classify=True,
)

gen = htmlCreator(trust_report= trust_report)
gen.convert_to_html()
print(trust_report)
