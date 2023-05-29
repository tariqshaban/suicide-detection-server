from flask import Flask, request, jsonify
from transformers import pipeline

MODEL_PATH = './assets/model'

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


if __name__ == '__main__':
    app.run()
