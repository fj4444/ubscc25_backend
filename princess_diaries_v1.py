import networkx as nx
import numpy as np
import heapq
from typing import List, Tuple, Dict, Set


def solve_princess_diaries(input_data: dict) -> dict:
    """
    公主日记任务调度优化 - 主要结果输出函数
    
    使用动态规划算法求解在时间约束下的最优任务调度方案，
    最大化得分并最小化交通费用。
    
    Args:
        input_data: 包含以下键的字典
            - tasks: 任务列表，每个任务包含name, start, end, station, score
            - subway: 地铁路线列表，每个路线包含connection和fee
            - starting_station: 起始车站ID
        
    Returns:
        包含以下键的字典:
            - max_score: 最大可能得分
            - min_fee: 最小交通费用
            - schedule: 按开始时间排序的任务名称列表
    """
    try:
        # 验证输入数据格式
        validation_result = _validate_input_data(input_data)
        if not validation_result['valid']:
            return {
                'max_score': 0,
                'min_fee': 0,
                'schedule': []
            }
        
        # 解析输入数据
        tasks = []
        for task_data in input_data['tasks']:
            try:
                task = {
                    'name': str(task_data['name']),
                    'start': int(task_data['start']),
                    'end': int(task_data['end']),
                    'station': int(task_data['station']),
                    'score': int(task_data['score'])
                }
                tasks.append(task)
            except (ValueError, TypeError, KeyError):
                continue
        
        # 创建地铁路线数据
        routes = []
        for route_data in input_data['subway']:
            try:
                route = {
                    'connection': [int(x) for x in route_data['connection']],
                    'fee': int(route_data['fee'])
                }
                routes.append(route)
            except (ValueError, TypeError, KeyError):
                continue

        starting_station = int(input_data['starting_station'])
        
        # 如果没有有效任务，返回空结果
        if not tasks:
            return {
                'max_score': 0,
                'min_fee': 0,
                'schedule': []
            }
        
        # 构建地铁图
        graph = _build_subway_graph(routes)
        
        # 检查起始车站是否在地铁图中
        if starting_station not in graph.nodes:
            return {
                'max_score': 0,
                'min_fee': 0,
                'schedule': []
            }
        
        # 计算距离矩阵
        distance_matrix, station_to_idx, idx_to_station = _compute_distance_matrix(graph)
        
        # 使用动态规划求解最优解
        optimal_result = _solve_optimal_schedule_advanced(tasks, graph, distance_matrix, station_to_idx, starting_station)
        
        return optimal_result
        
    except Exception as e:
        return {
            'max_score': 0,
            'min_fee': 0,
            'schedule': []
        }


def _validate_input_data(input_data: dict) -> dict:
    """验证输入数据格式是否正确"""
    try:
        if not isinstance(input_data, dict):
            return {'valid': False, 'error': '输入数据必须是字典格式'}
        
        required_fields = ['tasks', 'subway', 'starting_station']
        for field in required_fields:
            if field not in input_data:
                return {'valid': False, 'error': f'缺少必需字段: {field}'}
        
        if not isinstance(input_data['tasks'], list) or len(input_data['tasks']) == 0:
            return {'valid': False, 'error': 'tasks字段必须是非空列表'}
        
        # 检查任务数量约束：1 <= |T| <= 400
        if len(input_data['tasks']) > 400:
            return {'valid': False, 'error': '任务数量不能超过400个'}
        
        for i, task_data in enumerate(input_data['tasks']):
            if not isinstance(task_data, dict):
                return {'valid': False, 'error': f'tasks[{i}]必须是字典'}
            
            task_required_fields = ['name', 'start', 'end', 'station', 'score']
            for field in task_required_fields:
                if field not in task_data:
                    return {'valid': False, 'error': f'tasks[{i}]缺少字段: {field}'}
            
            if not isinstance(task_data['name'], str):
                return {'valid': False, 'error': f'tasks[{i}].name必须是字符串'}
            
            for field in ['start', 'end', 'station', 'score']:
                if not isinstance(task_data[field], (int, float)):
                    return {'valid': False, 'error': f'tasks[{i}].{field}必须是数字'}
                task_data[field] = int(task_data[field])
            
            if task_data['start'] >= task_data['end']:
                return {'valid': False, 'error': f'tasks[{i}]的开始时间必须小于结束时间'}
            
            if task_data['start'] < 0 or task_data['end'] < 0:
                return {'valid': False, 'error': f'tasks[{i}]的时间不能为负数'}
            
            # 检查时间约束：0 <= start_t < end_t <= 7200
            if task_data['start'] < 0 or task_data['end'] > 7200:
                return {'valid': False, 'error': f'tasks[{i}]的时间超出范围[0, 7200]'}
            
            # 检查得分约束：1 <= w_t <= 10
            if task_data['score'] < 1 or task_data['score'] > 10:
                return {'valid': False, 'error': f'tasks[{i}]的得分必须在[1, 10]范围内'}
        
        if not isinstance(input_data['subway'], list) or len(input_data['subway']) == 0:
            return {'valid': False, 'error': 'subway字段必须是非空列表'}
        
        for i, route_data in enumerate(input_data['subway']):
            if not isinstance(route_data, dict):
                return {'valid': False, 'error': f'subway[{i}]必须是字典'}
            
            route_required_fields = ['connection', 'fee']
            for field in route_required_fields:
                if field not in route_data:
                    return {'valid': False, 'error': f'subway[{i}]缺少字段: {field}'}
            
            if not isinstance(route_data['connection'], list) or len(route_data['connection']) != 2:
                return {'valid': False, 'error': f'subway[{i}].connection必须是包含两个元素的列表'}
            
            for j, station in enumerate(route_data['connection']):
                if not isinstance(station, (int, float)):
                    return {'valid': False, 'error': f'subway[{i}].connection[{j}]必须是数字'}
                route_data['connection'][j] = int(station)
            
            if not isinstance(route_data['fee'], (int, float)):
                return {'valid': False, 'error': f'subway[{i}].fee必须是数字'}
            route_data['fee'] = int(route_data['fee'])
            
            if route_data['fee'] < 0:
                return {'valid': False, 'error': f'subway[{i}].fee不能为负数'}
            
            # 检查费用约束：1 <= c({x,y}) <= 1000
            if route_data['fee'] < 1 or route_data['fee'] > 1000:
                return {'valid': False, 'error': f'subway[{i}].fee必须在[1, 1000]范围内'}
        
        if not isinstance(input_data['starting_station'], (int, float)):
            return {'valid': False, 'error': 'starting_station必须是数字'}
        input_data['starting_station'] = int(input_data['starting_station'])
        
        return {'valid': True, 'error': None}
        
    except Exception as e:
        return {'valid': False, 'error': f'验证过程中发生错误: {str(e)}'}


def _build_subway_graph(routes: List[Dict]) -> nx.Graph:
    """根据subway数据构建地铁图"""
    G = nx.Graph()
    
    # 添加所有车站节点
    all_stations = set()
    for route in routes:
        all_stations.update(route['connection'])
    
    for station in all_stations:
        G.add_node(station)
    
    # 添加连接和费用
    for route in routes:
        u, v = route['connection']
        fee = route['fee']
        G.add_edge(u, v, weight=fee)
    
    return G


def _compute_distance_matrix(graph: nx.Graph) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    """计算所有车站之间的最短距离矩阵（基于Dijkstra算法）"""
    stations = sorted(list(graph.nodes()))
    # 新增：若stations为空，返回空矩阵和空映射
    if not stations:
        return np.array([[]]), {}, []
    n = len(stations)
    station_to_idx = {station: i for i, station in enumerate(stations)}

    distance_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    # 构建邻接表
    adj = {station: [] for station in stations}
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        adj[u].append((v, weight))
        adj[v].append((u, weight))

    # 对每个station作为源点运行Dijkstra
    for src_idx, src_station in enumerate(stations):
        # Dijkstra
        dists = [np.inf] * n
        dists[src_idx] = 0
        heap = [(0, src_station)]
        visited = set()
        while heap:
            cost_u, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            u_idx = station_to_idx[u]
            for v, weight in adj[u]:
                v_idx = station_to_idx[v]
                if dists[v_idx] > dists[u_idx] + weight:
                    dists[v_idx] = dists[u_idx] + weight
                    heapq.heappush(heap, (dists[v_idx], v))
        distance_matrix[src_idx, :] = dists
    return distance_matrix, station_to_idx, stations


def _is_compatible(task1: Dict, task2: Dict) -> bool:
    """检查两个任务是否兼容（时间不重叠）"""
    try:
        if 'start' not in task1 or 'end' not in task1 or 'start' not in task2 or 'end' not in task2:
            return False
        return (task1['end'] <= task2['start'] or task2['end'] <= task1['start'])
    except Exception:
        return False


def _calculate_travel_cost(schedule: List[Dict], distance_matrix: np.ndarray, 
                          station_to_idx: Dict[int, int], starting_station: int) -> int:
    """
    计算给定调度方案的总交通费用
    
    根据问题描述：
    f_S = d(s_0, s_{t_1}) + sum_{i=1}^{n-1} d(s_{t_i}, s_{t_{i+1}}) + d(s_{t_n}, s_0)
    """
    if not schedule:
        return 0
    
    try:
        # 按开始时间排序（确保按时间顺序访问）
        sorted_schedule = sorted(schedule, key=lambda t: t['start'])
        
        total_cost = 0
        
        # 检查起始车站是否在图中
        if starting_station not in station_to_idx:
            return 0
        
        start_idx = station_to_idx[starting_station]
        
        # 1. 从起始车站到第一个任务的车站：d(s_0, s_{t_1})
        first_station = sorted_schedule[0]['station']
        if first_station not in station_to_idx:
            return 0
            
        first_idx = station_to_idx[first_station]
        cost = distance_matrix[start_idx][first_idx]
        if not np.isnan(cost) and not np.isinf(cost):
            total_cost += int(cost)
        
        # 2. 任务之间的移动费用：sum_{i=1}^{n-1} d(s_{t_i}, s_{t_{i+1}})
        for i in range(len(sorted_schedule) - 1):
            current_station = sorted_schedule[i]['station']
            next_station = sorted_schedule[i + 1]['station']
            
            if current_station not in station_to_idx or next_station not in station_to_idx:
                return 0
                
            current_idx = station_to_idx[current_station]
            next_idx = station_to_idx[next_station]
            cost = distance_matrix[current_idx][next_idx]
            
            if not np.isnan(cost) and not np.isinf(cost):
                total_cost += int(cost)
        
        # 3. 从最后一个任务的车站回到起始车站：d(s_{t_n}, s_0)
        last_station = sorted_schedule[-1]['station']
        if last_station not in station_to_idx:
            return 0
            
        last_idx = station_to_idx[last_station]
        cost = distance_matrix[last_idx][start_idx]
        if not np.isnan(cost) and not np.isinf(cost):
            total_cost += int(cost)
        
        return total_cost
    except Exception:
        return 0


def _calculate_score(schedule: List[Dict]) -> int:
    """计算给定调度方案的总得分"""
    try:
        return sum(task['score'] for task in schedule if 'score' in task)
    except Exception:
        return 0


def _solve_optimal_schedule_advanced(tasks: List[Dict], graph: nx.Graph, distance_matrix: np.ndarray,
                                    station_to_idx: Dict[int, int], starting_station: int) -> Dict:
    """
    使用高级动态规划求解最优调度方案
    
    目标：在最大得分的前提下，最小化交通费用
    处理相同得分但不同费用的情况
    """
    if not tasks:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
    
    try:
        # 按结束时间排序任务
        sorted_tasks = sorted(tasks, key=lambda t: t['end'])
        n = len(sorted_tasks)
        
        # dp[i] = (max_score, min_cost, task_sequence) 考虑前i个任务时的最优解
        dp = [(0, 0, [])] * (n + 1)
        
        for i in range(1, n + 1):
            current_task = sorted_tasks[i - 1]
            
            # 不选择当前任务
            dp[i] = dp[i - 1]
            
            # 选择当前任务
            # 找到最后一个与当前任务兼容的任务
            last_compatible = -1
            for j in range(i - 1, 0, -1):
                if _is_compatible(sorted_tasks[j - 1], current_task):
                    last_compatible = j - 1
                    break
            
            # 计算选择当前任务后的得分和费用
            if last_compatible == -1:
                # 没有兼容的前置任务，从起始站开始
                prev_score, prev_cost, prev_sequence = 0, 0, []
                if starting_station in station_to_idx and current_task['station'] in station_to_idx:
                    start_idx = station_to_idx[starting_station]
                    current_idx = station_to_idx[current_task['station']]
                    travel_cost = distance_matrix[start_idx][current_idx]
                    if not np.isnan(travel_cost) and not np.isinf(travel_cost):
                        prev_cost = int(travel_cost)
            else:
                # 有兼容的前置任务
                prev_score, prev_cost, prev_sequence = dp[last_compatible + 1]
                # 计算从上一个任务到当前任务的费用
                prev_task = sorted_tasks[last_compatible]
                if prev_task['station'] in station_to_idx and current_task['station'] in station_to_idx:
                    prev_idx = station_to_idx[prev_task['station']]
                    current_idx = station_to_idx[current_task['station']]
                    travel_cost = distance_matrix[prev_idx][current_idx]
                    if not np.isnan(travel_cost) and not np.isinf(travel_cost):
                        prev_cost += int(travel_cost)
            
            score_with_current = prev_score + current_task['score']
            # 修复：cost_with_current应该等于prev_cost（已经包含了到当前任务的费用）
            cost_with_current = prev_cost
            sequence_with_current = prev_sequence + [current_task]
            
            # 如果选择当前任务更好（得分更高，或得分相同但费用更低）
            if (score_with_current > dp[i][0] or 
                (score_with_current == dp[i][0] and cost_with_current < dp[i][1])):
                dp[i] = (score_with_current, cost_with_current, sequence_with_current)
        
        # 获取最优解
        max_score, min_cost, optimal_sequence = dp[n]
        
        # 计算完整的交通费用（包括回到起始站的费用）
        # 注意：这里需要重新计算完整费用，因为DP过程中只计算了部分费用
        travel_cost = _calculate_travel_cost(optimal_sequence, distance_matrix, station_to_idx, starting_station)
        
        # 按开始时间排序任务名称（升序）
        schedule = sorted([task['name'] for task in optimal_sequence], 
                         key=lambda name: next(t['start'] for t in optimal_sequence if t['name'] == name))
        
        return {
            'max_score': int(max_score),
            'min_fee': int(travel_cost),  # 使用重新计算的完整费用
            'schedule': schedule
        }
        
    except Exception:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}


def _solve_optimal_schedule(tasks: List[Dict], graph: nx.Graph, distance_matrix: np.ndarray,
                           station_to_idx: Dict[int, int], starting_station: int) -> Dict:
    """
    使用动态规划求解最优调度方案
    
    目标：在最大得分的前提下，最小化交通费用
    """
    if not tasks:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
    
    try:
        # 按结束时间排序任务
        sorted_tasks = sorted(tasks, key=lambda t: t['end'])
        n = len(sorted_tasks)
        
        # dp[i] = (max_score, min_cost) 考虑前i个任务时的最优解
        dp = [(0, 0)] * (n + 1)
        choice = [False] * (n + 1)  # choice[i] 表示是否选择了第i个任务
        
        for i in range(1, n + 1):
            current_task = sorted_tasks[i - 1]
            
            # 不选择当前任务
            dp[i] = dp[i - 1]
            choice[i] = False
            
            # 选择当前任务
            # 找到最后一个与当前任务兼容的任务
            last_compatible = -1
            for j in range(i - 1, 0, -1):
                if _is_compatible(sorted_tasks[j - 1], current_task):
                    last_compatible = j - 1
                    break
            
            # 计算选择当前任务的得分和费用
            if last_compatible == -1:
                # 没有兼容的前置任务
                prev_score, prev_cost = 0, 0
            else:
                prev_score, prev_cost = dp[last_compatible + 1]
            
            # 计算选择当前任务后的总费用
            # 需要计算从上一个任务（或起始站）到当前任务的费用
            if last_compatible == -1:
                # 从起始站到当前任务
                if starting_station in station_to_idx and current_task['station'] in station_to_idx:
                    start_idx = station_to_idx[starting_station]
                    current_idx = station_to_idx[current_task['station']]
                    travel_cost = distance_matrix[start_idx][current_idx]
                    if np.isnan(travel_cost) or np.isinf(travel_cost):
                        travel_cost = 0
                    else:
                        travel_cost = int(travel_cost)
                else:
                    travel_cost = 0
            else:
                # 从上一个任务到当前任务
                prev_task = sorted_tasks[last_compatible]
                if prev_task['station'] in station_to_idx and current_task['station'] in station_to_idx:
                    prev_idx = station_to_idx[prev_task['station']]
                    current_idx = station_to_idx[current_task['station']]
                    travel_cost = distance_matrix[prev_idx][current_idx]
                    if np.isnan(travel_cost) or np.isinf(travel_cost):
                        travel_cost = 0
                    else:
                        travel_cost = int(travel_cost)
                else:
                    travel_cost = 0
            
            score_with_current = prev_score + current_task['score']
            cost_with_current = prev_cost + travel_cost
            
            # 如果选择当前任务更好（得分更高，或得分相同但费用更低）
            if (score_with_current > dp[i][0] or 
                (score_with_current == dp[i][0] and cost_with_current < dp[i][1])):
                dp[i] = (score_with_current, cost_with_current)
                choice[i] = True
        
        # 重构最优解
        optimal_tasks = []
        i = n
        while i > 0:
            if choice[i]:  # 选择了第i个任务
                optimal_tasks.append(sorted_tasks[i - 1])
                # 找到最后一个兼容的任务
                last_compatible = -1
                for j in range(i - 1, 0, -1):
                    if _is_compatible(sorted_tasks[j - 1], sorted_tasks[i - 1]):
                        last_compatible = j - 1
                        break
                if last_compatible == -1:
                    break
                else:
                    i = last_compatible + 1
            else:
                i = i - 1
        
        # 计算完整的交通费用（包括回到起始站的费用）
        travel_cost = _calculate_travel_cost(optimal_tasks, distance_matrix, station_to_idx, starting_station)
        
        # 按开始时间排序任务名称（升序）
        schedule = sorted([task['name'] for task in optimal_tasks], 
                         key=lambda name: next(t['start'] for t in optimal_tasks if t['name'] == name))
        
        return {
            'max_score': int(dp[n][0]),
            'min_fee': int(travel_cost),
            'schedule': schedule
        }
        
    except Exception:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
if __name__ == "__main__":
    input_data = {
        "tasks": [
            { "name": "A", "start": 480, "end": 540, "station": 1, "score": 2 },
            { "name": "B", "start": 600, "end": 660, "station": 2, "score": 1 },
            { "name": "C", "start": 720, "end": 780, "station": 3, "score": 3 },
            { "name": "D", "start": 840, "end": 900, "station": 4, "score": 1 },
            { "name": "E", "start": 960, "end": 1020, "station": 1, "score": 4 },
            { "name": "F", "start": 530, "end": 590, "station": 2, "score": 1 }
        ],
        "subway": [
            { "connection": [0, 1], "fee": 10 },
            { "connection": [1, 2], "fee": 10 },
            { "connection": [2, 3], "fee": 20 },
            { "connection": [3, 4], "fee": 30 }
        ],
        "starting_station": 0
    }

    result = solve_princess_diaries(input_data)
    print("测试结果:", result)
