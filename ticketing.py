from flask import Flask, request, jsonify
from math import sqrt
import json

app = Flask(__name__)

def calculate_distance(coords1, coords2):
    """
    计算两个坐标点之间的欧几里得距离。
    """
    return sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)

def get_latency_points(distance):
    """
    根据距离计算延迟积分，分数与距离的倒数成正比。
    """
    # K 是一个平滑常数，防止除以零，并控制曲线的陡峭程度
    K = 1
    
    # 根据公式：分数 = 30 * K / (距离 + K)
    # 当距离为0时，分数为 30
    points = (30 * K) / (distance + K)
    
    # 确保分数不超过30，并返回整数
    return min(30, points)

@app.route('/ticketing-agent', methods=['POST'])
def ticketing_agent():
    try:
        data = request.json
        customers = data.get('customers', [])
        concerts = data.get('concerts', [])
        priority_map = data.get('priority', {})

        result = {}
        for customer in customers:
            max_score = -1
            best_concert_name = ""
            customer_name = customer['name']
            customer_location = tuple(customer['location'])
            customer_vip = customer['vip_status']
            customer_credit_card = customer['credit_card']

            for concert in concerts:
                current_score = 0
                concert_name = concert['name']
                concert_location = tuple(concert['booking_center_location'])

                # Factor 1: VIP Status
                if customer_vip:
                    current_score += 100

                # Factor 2: Credit Card Priority
                if priority_map.get(customer_credit_card) == concert_name:
                    current_score += 50

                # Factor 3: Latency
                distance = calculate_distance(customer_location, concert_location)
                # print(f"Distance from {customer_name} to {concert_name}: {distance}")
                current_score += get_latency_points(distance)
                # print(f"Latency points for {customer_name} to {concert_name}: {get_latency_points(distance)}")

                if current_score > max_score:
                    max_score = current_score
                    best_concert_name = concert_name

            result[customer_name] = best_concert_name

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)