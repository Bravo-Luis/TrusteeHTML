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
   
test = TrusteeFormatter(trust_report=report, output_dir=dir)
test.json()
test.html()