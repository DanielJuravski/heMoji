from flask import Flask
from flask_restful import Resource, Api

from hemoji_predict import load_hemoji_model, encode_input_sentence, evaluate


class HeMoji(Resource):
    def __init__(self):
        pass

    def get(self, text):
        tokens = encode_input_sentence(sentok, input_sentence=text)
        if tokens is not None:
            emojis, emojis_probs = evaluate(model, tokens)
            result_emojis = {e: emojis[e] for e in range(len(emojis))}
            result_emojis_probs = {p: str(emojis_probs[p]) for p in range(len(emojis_probs))}
        else:
            result_emojis = result_emojis_probs = 'N/A'

        result = dict()
        result['input'] = text
        result['emojis'] = result_emojis
        result['emojis_probs'] = result_emojis_probs

        return result


def main():
    global sentok, model
    sentok, model = load_hemoji_model()

    app = Flask(__name__)
    api = Api(app)

    api.add_resource(HeMoji, '/<string:text>')
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()

