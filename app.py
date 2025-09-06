from flask import Flask, request, jsonify
from ticketing import agent
from BlanketyBlanksAlgo import BlanketyBlanksAlgoTest

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



if __name__ == '__main__':
    app.run(debug=True)