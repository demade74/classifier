from flask import request, jsonify
#from spam_classifier import classify
from spam_classifier_v2 import classify, model_scoring
from application import app


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/hello_user', methods=['POST'])
def hello_user():
    data = request.json
    user = data['user']
    return f'hello {user}'


@app.route('/increment_number', methods=['POST'])
def increment_number():
    data = request.json
    number = int(data['number'])
    number += 1
    return f'{number}'


@app.route('/classify_text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text')

    if text is None:
        params = ', '.join(data.keys())
        return jsonify({'message': f'Parameter "{params}" is invalid'}), 400
    else:
        result = classify(text)
        return jsonify({'result': result})


@app.route('/classify', methods=['POST'])
def classify_text_v2():
    data = request.json
    text = data.get('text')

    if text is None:
        params = ', '.join(data.keys())
        return jsonify({'message': f'Parameter "{params}" is invalid'}), 400
    else:
        result = classify(text)
        return jsonify({'result': result})


@app.route('/model_scoring', methods=['GET'])
def scoring():
    return str(model_scoring())
