from trustee.report.trust import TrustReport
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from trusteehtml.CLI import CLIController

df = pd.read_csv('Databases/balanced_dataset_NF_UQ_NIDS.csv')
X = df.drop('Attack', axis=1)
y = df['Attack']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
        class_names=['Benign','Exploits','Reconnaissance','DoS','Generic','Shellcode',
 'Backdoor','Fuzzers','Worms','Analysis','injection','DDoS','scanning',
 'password','mitm','xss','ransomware','Infilteration','Bot','Brute Force',
 'Theft'],
        feature_names=['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS'],
        is_classify=True,
    )

controller = CLIController(trust_report, "NF_UQ_NIDS")
controller.run()