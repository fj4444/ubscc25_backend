from flask import Flask, request, jsonify
from ticketing import agent
from BlanketyBlanksAlgo import BlanketyBlanksAlgoTest
from princess_diaries_v1 import princess_diaries
from spy import investigate

app = Flask(__name__)

@app.route('/blankety', methods=['POST'])
def BlanketyBlanksAlgo():
    try:
        data = request.json
        return BlanketyBlanksAlgoTest(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/ticketing-agent', methods=['POST'])
def ticketing_agent():
    try:
        data = request.json
        result = agent(data) 
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/investigate', methods=['POST'])
def spy():
    try:
        data = request.json
        result = investigate(data)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    try:
        data = request.json
        result = princess_diaries(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)