from flask import Flask, request, jsonify
from ticketing import agent
# from BlanketyBlanksAlgo import BlanketyBlanksAlgoTest
from bbnew_ import BlanketyBlanksAlgoTest
from trade import LatexFormulaEvaluator
from princess_diaries_v1 import princess_diaries
from spy import investigate
from sail import SailingClubHandler

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

@app.route('/trading-formula', methods=['POST'])
def evaluate_formulas():
    """
    处理 POST 请求，评估 JSON 输入中的所有公式。
    """
    try:
        data = request.json
        if not isinstance(data, list):
            return jsonify({"error": "Expected a JSON array"}), 400

        results = []
        for case in data:
            # 从每个测试用例中提取公式和变量
            formula = case.get("formula")
            variables = case.get("variables")
            # 创建评估器实例并计算结果
            evaluator = LatexFormulaEvaluator(formula, variables)
            result = evaluator.evaluate()
            # 将结果添加到列表中
            results.append({"result": result})

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    try:
        data = request.json
        result = princess_diaries(data)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/sailing-club', methods=['POST'])
def sailing_club():
    try:
        data = request.json
        handler = SailingClubHandler()
        return jsonify(handler.handle_request(data)), 200
        # return handler.handle_request(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

from Duolinggo import process
@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    try:
        data = request.json
        processed_data = process(data)
        return processed_data, 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

from InkArchive import final
@app.route('/The-Ink-Archive', methods=['POST'])
def The_Ink_Archive():
    try:
        data = request.json
        return jsonify(final(data)),200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
