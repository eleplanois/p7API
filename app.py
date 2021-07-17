from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle

with open('lgbm_total_random.p', 'rb') as f2:
    grid_lgbm = pickle.load(f2)

df = pd.read_csv('df_total_sample.csv', index_col=0)

print(df.head())

print(df[df['SK_ID_CURR']==100002])
print(grid_lgbm.predict(df[df['SK_ID_CURR']==100002]))


#app = Flask(__name__)
#app.run(port=5000)
