import numpy as np
#import pandas as pd
import csv, pickle #string, re

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

file_in = open('gps_test.csv', 'r',encoding = "ISO-8859-1")
data = csv.reader(file_in)
txt = ''
statuses = []
labels = []
lengths = []

for row in data:
    labels.append([row[0:5]])
    statuses.append(row[6])
    lengths.append(len(row[6]))
    txt += row[6]

maxlen = 1024
chars = sorted(set(txt))
print(chars)

