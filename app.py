from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

with open('lgbm_total_random.p', 'rb') as f2:
    grid_lgbm = pickle.load(f2)

df = pd.read_csv('df_total_sample.csv', index_col=0)
df.drop(columns='TARGET', inplace=True)
num_client = df.SK_ID_CURR.unique().astype(str)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/')
def predict():
    """

    Returns
    liste des clients dans le fichier

    """
    return jsonify({"list_client_id" : list(num_client)})


@app.route('/predict/<int:sk_id>')
def predict_get(sk_id):
    """

    Parameters
    ----------
    sk_id : numero de client

    Returns
    -------
    prediction  0 pour paiement OK
                1 pour defaut de paiement

    """
#http://127.0.0.1:5000/predict/415560
    if sk_id in num_client:
        predict = grid_lgbm.predict(df[df['SK_ID_CURR']==sk_id])[0]
    else:
        predict = "client inconnu"
    return jsonify({ 'retour_prediction' : str(predict) })


app.run(port=5000)
