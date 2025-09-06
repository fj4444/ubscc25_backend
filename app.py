from flask import Flask, request, jsonify
from ticketing import agent
# from BlanketyBlanksAlgo import BlanketyBlanksAlgoTest
from bbnew_ import BlanketyBlanksAlgoTest
from trade import LatexFormulaEvaluator
from princess_diaries_v1 import solve_princess_diaries
from spy import investigate
from sail import SailingClubHandler
from micromouse import process_micromouse_request
from flask_cors import CORS # 导入 CORS 模块

app = Flask(__name__)
CORS(app) # 启用 CORS，允许所有来源的请求

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
        result = solve_princess_diaries(data)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['POST'])
def sailing_club():
    try:
        data = request.json
        handler = SailingClubHandler()
        # return jsonify(handler.handle_request(data)), 200
        return handler.handle_request(data), 200

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

from TradingBot import final_trading
@app.route('/trading-bot', methods=['POST'])
def trading_bot():
    try:
        data = request.json
        return jsonify(final_trading(data)),200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

from mage import process
@app.route('/the-mages-gambit',methods=['POST'])
def mage_time():
    try:
        data = request.json
        return jsonify(process(data)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

from firewall import final_firewall
@app.route('/operation-safeguard',methods=['POST'])
def firewall():
    try:
        data = request.json
        return jsonify(final_firewall(data)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/2048', methods=['POST'])
def handle_2048():
    """
    处理来自前端的 2048 游戏请求。
    """
    # 确保请求体是 JSON 格式
    if not request.is_json:
        return jsonify({"error": "请求体必须是 JSON 格式"}), 400

    # 从请求中获取网格和移动方向
    data = request.get_json()
    grid = data.get('grid')
    direction = data.get('mergeDirection')

    # 简单的输入验证
    if not grid or not direction:
        return jsonify({"error": "缺少 'grid' 或 'mergeDirection' 参数"}), 400

    # 调用核心游戏逻辑处理函数
    next_grid, end_game = process_grid(grid, direction)

    # 构造并返回响应
    response = {
        "nextGrid": next_grid,
        "endGame": end_game
    }
    return jsonify(response)

@app.route("/micro-mouse", methods=["POST"])
def micro_mouse_api():
    """
    微鼠竞赛API端点
    处理微鼠的移动指令请求
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "请求数据不能为空"}), 400
        
        # 调用微鼠控制器处理请求
        result = process_micromouse_request(data)
        
        # 直接返回结果（已经是正确的JSON格式）
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"微鼠控制器错误: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
