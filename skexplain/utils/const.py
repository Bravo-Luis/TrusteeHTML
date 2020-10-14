import rootpath
from skexplain.enums.feature_type import FeatureType

WINE_DATASET_META = {
    "name": "wine",
    "path": "{}/res/dataset/wine.csv".format(rootpath.detect()),
    "has_header": True,
    "delimiter": ";",
    "fields": [
        ("fixed acidity", FeatureType.NUMERICAL, False),
        ("volatile acidity", FeatureType.NUMERICAL, False),
        ("citric acid", FeatureType.NUMERICAL, False),
        ("residual sugar", FeatureType.NUMERICAL, False),
        ("chlorides", FeatureType.NUMERICAL, False),
        ("free sulfur dioxide", FeatureType.NUMERICAL, False),
        ("total sulfur dioxide", FeatureType.NUMERICAL, False),
        ("density", FeatureType.NUMERICAL, False),
        ("pH", FeatureType.NUMERICAL, False),
        ("sulphates", FeatureType.NUMERICAL, False),
        ("alcohol", FeatureType.NUMERICAL, False),
        ("quality", FeatureType.NUMERICAL, True)
    ],
    "type": "regression"
}

DIABETES_DATASET_META = {
    "name": "diabetes",
    "path": "{}/res/dataset/diabetes/diabetic_data.csv".format(rootpath.detect()),
    "has_header": True,
    "delimiter": ",",
    "fields": [
        ("encounter_id", FeatureType.IDENTIFIER, False),
        ("patient_nbr", FeatureType.IDENTIFIER, False),
        ("race", FeatureType.CATEGORICAL, False),
        ("gender", FeatureType.CATEGORICAL, False),
        ("age", FeatureType.CATEGORICAL, False),
        ("weight", FeatureType.CATEGORICAL, False),
        ("admission_type_id", FeatureType.CATEGORICAL, False),
        ("discharge_disposition_id", FeatureType.CATEGORICAL, False),
        ("admission_source_id", FeatureType.CATEGORICAL, False),
        ("time_in_hospital", FeatureType.NUMERICAL, False),
        ("payer_code", FeatureType.CATEGORICAL, False),
        ("medical_specialty", FeatureType.CATEGORICAL, False),
        ("num_lab_procedures", FeatureType.NUMERICAL, False),
        ("num_procedures", FeatureType.NUMERICAL, False),
        ("num_medications", FeatureType.NUMERICAL, False),
        ("number_outpatient", FeatureType.NUMERICAL, False),
        ("number_emergency", FeatureType.NUMERICAL, False),
        ("number_inpatient", FeatureType.NUMERICAL, False),
        ("diag_1", FeatureType.CATEGORICAL, False),
        ("diag_2", FeatureType.CATEGORICAL, False),
        ("diag_3", FeatureType.CATEGORICAL, False),
        ("number_diagnoses", FeatureType.NUMERICAL, False),
        ("max_glu_serum", FeatureType.CATEGORICAL, False),
        ("A1Cresult", FeatureType.CATEGORICAL, False),
        ("metformin", FeatureType.CATEGORICAL, False),
        ("repaglinide", FeatureType.CATEGORICAL, False),
        ("nateglinide", FeatureType.CATEGORICAL, False),
        ("chlorpropamide", FeatureType.CATEGORICAL, False),
        ("glimepiride", FeatureType.CATEGORICAL, False),
        ("acetohexamide", FeatureType.CATEGORICAL, False),
        ("glipizide", FeatureType.CATEGORICAL, False),
        ("glyburide", FeatureType.CATEGORICAL, False),
        ("tolbutamide", FeatureType.CATEGORICAL, False),
        ("pioglitazone", FeatureType.CATEGORICAL, False),
        ("rosiglitazone", FeatureType.CATEGORICAL, False),
        ("acarbose", FeatureType.CATEGORICAL, False),
        ("miglitol", FeatureType.CATEGORICAL, False),
        ("troglitazone", FeatureType.CATEGORICAL, False),
        ("tolazamide", FeatureType.CATEGORICAL, False),
        ("examide", FeatureType.CATEGORICAL, False),
        ("citoglipton", FeatureType.CATEGORICAL, False),
        ("insulin", FeatureType.CATEGORICAL, False),
        ("glyburide-metformin", FeatureType.CATEGORICAL, False),
        ("glipizide-metformin", FeatureType.CATEGORICAL, False),
        ("glimepiride-pioglitazone", FeatureType.CATEGORICAL, False),
        ("metformin-rosiglitazone", FeatureType.CATEGORICAL, False),
        ("metformin-pioglitazone", FeatureType.CATEGORICAL, False),
        ("change", FeatureType.CATEGORICAL, False),
        ("diabetesMed", FeatureType.CATEGORICAL, False),
        ("readmitted", FeatureType.CATEGORICAL, True)
    ],
    "type": "classification"
}

IOT_DATASET_META = {
    "name": "iot",
    "path": "{}/res/dataset/iot/csv_files/16-09-23-labeled.csv".format(rootpath.detect()),
    # "path": "{}/res/dataset/iot/csv_files/".format(rootpath.detect()),
    # "is_dir": True,
    "has_header": False,
    "fields": [
        ("Frame Length", FeatureType.NUMERICAL, False),
        ("Ethernet Type", FeatureType.NUMERICAL, False),
        ("IP Protocol", FeatureType.CATEGORICAL, False),
        ("IPv4 Flags", FeatureType.CATEGORICAL, False),
        ("IPv6 Next Header", FeatureType.CATEGORICAL, False),
        ("IPv6 Option", FeatureType.CATEGORICAL, False),
        ("TCP Flags", FeatureType.CATEGORICAL, False),
        ("IoT Device Type", FeatureType.CATEGORICAL, True)
    ],
    "classes": ["Smart Static", "Sensor", "Audio", "Video", "Other"],
    "converters": {1: lambda x: int(x, 16) if x else None, 3: lambda x: int(x, 16) if x else None, 6: lambda x: int(x, 16) if x else None},
    "type": "classification",
    "categories": {
        "IP Protocol": [-1, 0, 1, 2, 6, 17, 145, 242],
        "IPv4 Flags": [-1, 0, 185, 925, 8192, 8377, 8562, 8747, 8932, 16384, 48299, 60692],
        "IPv6 Next Header": [-1, 0, 6, 17, 44, 58],
        "IPv6 Option": [-1, 1],
        "TCP Flags": [-1, 1, 2, 4, 16, 17, 18, 20, 24, 25, 28, 47, 49, 56, 82, 144, 152, 153, 168, 194, 210, 1041, 2050, 2051, 2513, 3345, 3610]
    }
}

BOSTON_DATASET_META = {
    "name": "boston",
    "path": "{}/res/dataset/boston.csv".format(rootpath.detect()),
    "has_header": True,
    "fields": [
        ("CRIM", FeatureType.NUMERICAL, False),
        ("ZN", FeatureType.NUMERICAL, False),
        ("INDUS", FeatureType.NUMERICAL, False),
        ("CHAS", FeatureType.CATEGORICAL, False),
        ("NOX", FeatureType.NUMERICAL, False),
        ("RM", FeatureType.NUMERICAL, False),
        ("AGE", FeatureType.NUMERICAL, False),
        ("DIS", FeatureType.NUMERICAL, False),
        ("RAD", FeatureType.NUMERICAL, False),
        ("TAX", FeatureType.NUMERICAL, False),
        ("PTRATIO", FeatureType.NUMERICAL, False),
        ("B", FeatureType.NUMERICAL, False),
        ("LSTAT", FeatureType.NUMERICAL, False),
        ("MEDV", FeatureType.CATEGORICAL, True)
    ],
    "type": "regression"
}


def cic_ids_2017_label_converter(label):
    # print(label)

    value = -1
    labels = {
        'BENIGN': 0,
        'Bot': 1,
        'DDoS': 2,
        'DoS GoldenEye': 3,
        'DoS Hulk': 4,
        'DoS Slowhttptest': 5,
        'DoS slowloris': 6,
        'FTP-Patator': 7,
        'Heartbleed': 8,
        'Infiltration': 9,
        'PortScan': 10,
        'SSH-Patator': 11,
        'Web Attack Brute Force': 12,
        'Web Attack Sql Injection': 13,
        'Web Attack XSS': 14
    }

    try:
        value = labels.get(label)
    except Exception as err:
        print('Exception', err, label)

    return value


CIC_IDS_2017_DATASET_META = {
    "name": "cic_ids_2017",
    # "path": "{}/res/dataset/CIC-IDS-2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv".format(rootpath.detect()),
    "path": "{}/res/dataset/CIC-IDS-2017/MachineLearningCVE/".format(rootpath.detect()),
    "is_dir": True,
    "has_header": True,
    "fields": [
        ("Destination Port", FeatureType.NUMERICAL, False),
        ("Flow Duration", FeatureType.NUMERICAL, False),
        ("Total Fwd Packets", FeatureType.NUMERICAL, False),
        ("Total Backward Packets", FeatureType.NUMERICAL, False),
        ("Total Length of Fwd Packets", FeatureType.NUMERICAL, False),
        ("Total Length of Bwd Packets", FeatureType.NUMERICAL, False),
        ("Fwd Packet Length Max", FeatureType.NUMERICAL, False),
        ("Fwd Packet Length Min", FeatureType.NUMERICAL, False),
        ("Fwd Packet Length Mean", FeatureType.NUMERICAL, False),
        ("Fwd Packet Length Std", FeatureType.NUMERICAL, False),
        ("Bwd Packet Length Max", FeatureType.NUMERICAL, False),
        ("Bwd Packet Length Min", FeatureType.NUMERICAL, False),
        ("Bwd Packet Length Mean", FeatureType.NUMERICAL, False),
        ("Bwd Packet Length Std", FeatureType.NUMERICAL, False),
        ("Flow Bytes/s", FeatureType.NUMERICAL, False),
        ("Flow Packets/s", FeatureType.NUMERICAL, False),
        ("Flow IAT Mean", FeatureType.NUMERICAL, False),
        ("Flow IAT Std", FeatureType.NUMERICAL, False),
        ("Flow IAT Max", FeatureType.NUMERICAL, False),
        ("Flow IAT Min", FeatureType.NUMERICAL, False),
        ("Fwd IAT Total", FeatureType.NUMERICAL, False),
        ("Fwd IAT Mean", FeatureType.NUMERICAL, False),
        ("Fwd IAT Std", FeatureType.NUMERICAL, False),
        ("Fwd IAT Max", FeatureType.NUMERICAL, False),
        ("Fwd IAT Min", FeatureType.NUMERICAL, False),
        ("Bwd IAT Total", FeatureType.NUMERICAL, False),
        ("Bwd IAT Mean", FeatureType.NUMERICAL, False),
        ("Bwd IAT Std", FeatureType.NUMERICAL, False),
        ("Bwd IAT Max", FeatureType.NUMERICAL, False),
        ("Bwd IAT Min", FeatureType.NUMERICAL, False),
        ("Fwd PSH Flags", FeatureType.CATEGORICAL, False),
        ("Bwd PSH Flags", FeatureType.CATEGORICAL, False),
        ("Fwd URG Flags", FeatureType.CATEGORICAL, False),
        ("Bwd URG Flags", FeatureType.CATEGORICAL, False),
        ("Fwd Header Length 2", FeatureType.IDENTIFIER, False),  # duplicate column, so ignore it
        ("Bwd Header Length", FeatureType.NUMERICAL, False),
        ("Fwd Packets/s", FeatureType.NUMERICAL, False),
        ("Bwd Packets/s", FeatureType.NUMERICAL, False),
        ("Min Packet Length", FeatureType.NUMERICAL, False),
        ("Max Packet Length", FeatureType.NUMERICAL, False),
        ("Packet Length Mean", FeatureType.NUMERICAL, False),
        ("Packet Length Std", FeatureType.NUMERICAL, False),
        ("Packet Length Variance", FeatureType.NUMERICAL, False),
        ("FIN Flag Count", FeatureType.NUMERICAL, False),
        ("SYN Flag Count", FeatureType.NUMERICAL, False),
        ("RST Flag Count", FeatureType.NUMERICAL, False),
        ("PSH Flag Count", FeatureType.NUMERICAL, False),
        ("ACK Flag Count", FeatureType.NUMERICAL, False),
        ("URG Flag Count", FeatureType.NUMERICAL, False),
        ("CWE Flag Count", FeatureType.NUMERICAL, False),
        ("ECE Flag Count", FeatureType.NUMERICAL, False),
        ("Down/Up Ratio", FeatureType.NUMERICAL, False),
        ("Average Packet Size", FeatureType.NUMERICAL, False),
        ("Avg Fwd Segment Size", FeatureType.NUMERICAL, False),
        ("Avg Bwd Segment Size", FeatureType.NUMERICAL, False),
        ("Fwd Header Length", FeatureType.NUMERICAL, False),
        ("Fwd Avg Bytes/Bulk", FeatureType.NUMERICAL, False),
        ("Fwd Avg Packets/Bulk", FeatureType.NUMERICAL, False),
        ("Fwd Avg Bulk Rate", FeatureType.NUMERICAL, False),
        ("Bwd Avg Bytes/Bulk", FeatureType.NUMERICAL, False),
        ("Bwd Avg Packets/Bulk", FeatureType.NUMERICAL, False),
        ("Bwd Avg Bulk Rate", FeatureType.NUMERICAL, False),
        ("Subflow Fwd Packets", FeatureType.NUMERICAL, False),
        ("Subflow Fwd Bytes", FeatureType.NUMERICAL, False),
        ("Subflow Bwd Packets", FeatureType.NUMERICAL, False),
        ("Subflow Bwd Bytes", FeatureType.NUMERICAL, False),
        ("Init_Win_bytes_forward", FeatureType.NUMERICAL, False),
        ("Init_Win_bytes_backward", FeatureType.NUMERICAL, False),
        ("act_data_pkt_fwd", FeatureType.NUMERICAL, False),
        ("min_seg_size_forward", FeatureType.NUMERICAL, False),
        ("Active Mean", FeatureType.NUMERICAL, False),
        ("Active Std", FeatureType.NUMERICAL, False),
        ("Active Max", FeatureType.NUMERICAL, False),
        ("Active Min", FeatureType.NUMERICAL, False),
        ("Idle Mean", FeatureType.NUMERICAL, False),
        ("Idle Std", FeatureType.NUMERICAL, False),
        ("Idle Max", FeatureType.NUMERICAL, False),
        ("Idle Min", FeatureType.NUMERICAL, False),
        ("Label", FeatureType.CATEGORICAL, True)
    ],
    "classes": ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack Brute Force', 'Web Attack Sql Injection', 'Web Attack XSS'],
    "converters": {"Label": lambda x: cic_ids_2017_label_converter(x)},
    "type": "classification"
}
