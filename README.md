# TrusteeHTML
Creates an HTML/JSON version of the trust report ouput of 
<a href="https://github.com/TrusteeML/trustee"> Trustee <a/>

## TrusteeFormatter

```python
import pickle
import os
from trustee.report.format import TrusteeFormatter

# Pickled Trust Report
with open('report.pickle', 'rb') as f:
    report = pickle.load(f)
    
dir = 'test/'
try:
       os.mkdir(dir)
except: 
       pass

# To Enhance ChatGPT Analysis   
additional_info = """

Model: Random Forest Classifier
Hyperparameters: {'n_estimators': 20, 'max_depth': 4, 'max_features': 'sqrt'}

These are values returned by the youtube api, the dataset was created from a packet trace while pausing and playing a youtube video (simulated by selenium web driver).
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

"""

test = TrusteeFormatter(trust_report=report, output_dir=dir, additional_info=additional_info)

# Creates JSON version of Trust Report
test.json()

# Creates HTML version of Trust Report (and optionally ChatGPT Analysis)
test.html()
```

### Here is an example output of the TrusteeFormatter html method

https://github.com/Bravo-Luis/TrusteeHTML/assets/91937163/1f20fd12-896e-4300-9a81-0e2f1f444aba

### Here is an example of the ChatGPT Analysis

<img width="800" alt="Screenshot 2023-12-19 at 9 11 09 PM" src="https://github.com/Bravo-Luis/TrusteeHTML/assets/91937163/e5da3fd9-d966-481a-b1c9-2e5654ea4ccf">

