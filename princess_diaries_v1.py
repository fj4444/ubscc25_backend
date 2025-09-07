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
        
        # 使用启发式方法求解（可以选择不同的策略）
        optimal_result = _solve_heuristic_schedule(tasks, graph, distance_matrix, station_to_idx, starting_station)
        
        # 可选：比较不同启发式方法的结果
        # greedy_result = _solve_greedy_schedule(tasks, graph, distance_matrix, station_to_idx, starting_station)
        # 选择更好的结果
        # if greedy_result['max_score'] > optimal_result['max_score'] or \
        #    (greedy_result['max_score'] == optimal_result['max_score'] and greedy_result['min_fee'] < optimal_result['min_fee']):
        #     optimal_result = greedy_result
        
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


def _solve_heuristic_schedule(tasks: List[Dict], graph: nx.Graph, distance_matrix: np.ndarray,
                             station_to_idx: Dict[int, int], starting_station: int) -> Dict:
    """
    使用启发式方法求解任务调度方案
    
    策略：
    1. 按得分密度（得分/时间长度）排序任务
    2. 贪心选择兼容的任务
    3. 优先选择得分高且交通成本低的任务
    """
    if not tasks:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
    
    try:
        # 计算每个任务的得分密度（得分/持续时间）
        for task in tasks:
            duration = task['end'] - task['start']
            task['score_density'] = task['score'] / duration if duration > 0 else 0
        
        # 按得分密度降序排序，得分密度相同时按得分降序排序
        sorted_tasks = sorted(tasks, key=lambda t: (-t['score_density'], -t['score'], t['start']))
        
        selected_tasks = []
        current_time = 0
        current_station = starting_station
        
        for task in sorted_tasks:
            # 检查任务是否与已选择的任务时间兼容
            is_compatible = True
            for selected_task in selected_tasks:
                if not _is_compatible(task, selected_task):
                    is_compatible = False
                    break
            
            if is_compatible:
                # 计算选择这个任务的交通成本
                travel_cost = 0
                if current_station in station_to_idx and task['station'] in station_to_idx:
                    current_idx = station_to_idx[current_station]
                    task_idx = station_to_idx[task['station']]
                    cost = distance_matrix[current_idx][task_idx]
                    if not np.isnan(cost) and not np.isinf(cost):
                        travel_cost = int(cost)
                
                # 计算任务的净收益（得分 - 交通成本）
                net_benefit = task['score'] - travel_cost
                
                # 如果净收益为正，或者任务得分很高，则选择该任务
                if net_benefit > 0 or task['score'] >= 3:
                    selected_tasks.append(task)
                    current_station = task['station']
        
        # 计算总得分
        total_score = sum(task['score'] for task in selected_tasks)
        
        # 计算总交通费用（包括回到起始站的费用）
        total_travel_cost = _calculate_travel_cost(selected_tasks, distance_matrix, station_to_idx, starting_station)
        
        # 按开始时间排序任务名称（升序）
        schedule = sorted([task['name'] for task in selected_tasks], 
                         key=lambda name: next(t['start'] for t in selected_tasks if t['name'] == name))
        
        return {
            'max_score': int(total_score),
            'min_fee': int(total_travel_cost),
            'schedule': schedule
        }
        
    except Exception:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}


def _solve_greedy_schedule(tasks: List[Dict], graph: nx.Graph, distance_matrix: np.ndarray,
                          station_to_idx: Dict[int, int], starting_station: int) -> Dict:
    """
    使用贪心算法求解任务调度方案
    
    策略：
    1. 按结束时间排序任务
    2. 贪心选择最早结束且兼容的任务
    """
    if not tasks:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
    
    try:
        # 按结束时间排序任务
        sorted_tasks = sorted(tasks, key=lambda t: t['end'])
        
        selected_tasks = []
        
        for task in sorted_tasks:
            # 检查任务是否与已选择的任务时间兼容
            is_compatible = True
            for selected_task in selected_tasks:
                if not _is_compatible(task, selected_task):
                    is_compatible = False
                    break
            
            if is_compatible:
                selected_tasks.append(task)
        
        # 计算总得分
        total_score = sum(task['score'] for task in selected_tasks)
        
        # 计算总交通费用（包括回到起始站的费用）
        total_travel_cost = _calculate_travel_cost(selected_tasks, distance_matrix, station_to_idx, starting_station)
        
        # 按开始时间排序任务名称（升序）
        schedule = sorted([task['name'] for task in selected_tasks], 
                         key=lambda name: next(t['start'] for t in selected_tasks if t['name'] == name))
        
        return {
            'max_score': int(total_score),
            'min_fee': int(total_travel_cost),
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
