import pickle
import os
from trustee.report.format import TrusteeFormatter
from trustee.report.threshold import TrusteeThreshold

with open('report.pickle', 'rb') as f:
    report = pickle.load(f)
    
dir = 'test/'

try:
       os.mkdir(dir)
except: 
       pass
   
additional_info = """

Model: Random Forest Classifier
Hyperparameters: {'n_estimators': 20, 'max_depth': 4, 'max_features': 'sqrt'}
Dataset Sample:
       rolling_median_time_since_last_DNS  ratio_time_since_last_DNS  curr_state
0                                 0.0                   0.444988         1
1                                 0.0                   0.445105         1
2                                 0.0                   0.447367         1

These are values returned by the youtube api, the dataset was created from a packet trace while pausing and playing a youtube video(simulated by selenium web driver).
(curr_state is the label)
class 1: Playing 
class 2: Not Playing
class 3: Buffering

Features:
ratio_time_since_last_DNS: 
    Description: Ratio of time since last DNS request to the median time since last DNS request
    Importance: 0.5209691063435197
    
rolling_median_time_since_last_DNS: 
    Description: Rolling median of time since last DNS request
    Importance: 0.4790308936564804
    
On Training Dataset:
Classification Report:
               precision    recall  f1-score   support

     Playing       0.73      0.69      0.71    417259
      Paused       0.70      0.72      0.71    417924
   Buffering       0.83      0.85      0.84    419053

    accuracy                           0.75   1254236
   macro avg       0.75      0.75      0.75   1254236
weighted avg       0.75      0.75      0.75   1254236

On Testing Dataset:
Classification Report:
               precision    recall  f1-score   support

     Playing       0.70      0.71      0.71    421599
      Paused       0.72      0.69      0.70    421599
   Buffering       0.83      0.86      0.85    421599

    accuracy                           0.75   1264797
   macro avg       0.75      0.75      0.75   1264797
weighted avg       0.75      0.75      0.75   1264797


"""
test = TrusteeFormatter(trust_report=report, output_dir=dir, additional_info=additional_info)
test.json()
test.html()