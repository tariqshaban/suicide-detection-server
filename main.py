from flask import Flask, request, jsonify
from transformers import pipeline
import pandas as pd

MODEL_PATH = './assets/model'

df = pd.read_csv('./assets/dataset/Suicide_Detection.csv')

trained_pipeline = pipeline('sentiment-analysis', model=MODEL_PATH, tokenizer=MODEL_PATH)

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    text = request.values.get('text')

    if text is None:
        response = jsonify('You must specify a "text" value')
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.status = 400
        return response

    prediction = trained_pipeline(text, truncation=True, max_length=4096)[0]

    response = jsonify(prediction)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.status = 200
    return response


@app.route('/sample', methods=['GET'])
def sample():
    sample_instance = df.sample(n=1)['text'].item()

    response = jsonify({'text': sample_instance})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.status = 200
    return response


if __name__ == '__main__':
    app.run()
