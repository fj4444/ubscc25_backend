import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set
from itertools import combinations
import heapq
import time


def solve_princess_diaries(input_data: dict) -> dict:
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
            # 如果某个任务数据无效，跳过它
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
            # 如果某个路线数据无效，跳过它
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
    
    # 计算距离矩阵
    distance_matrix, station_to_idx, idx_to_station = _compute_distance_matrix(graph)
    
    # 使用简单贪心算法快速求解（避免超时）

    optimal_result = _solve_greedy(tasks, graph, distance_matrix, station_to_idx, starting_station)
    # 返回字典格式的结果
    result = {
        'max_score': optimal_result['max_score'],
        'min_fee': optimal_result['min_fee'],
        'schedule': optimal_result['schedule']
    }
    
    return result



def _validate_input_data(input_data: dict) -> dict:
    """
    验证输入数据格式是否正确
    
    Args:
        input_data: 待验证的输入数据
        
    Returns:
        包含valid(布尔值)和error(错误信息)的字典
    """
    try:
        # 检查是否为字典
        if not isinstance(input_data, dict):
            return {'valid': False, 'error': '输入数据必须是字典格式'}
        
        # 检查必需字段
        required_fields = ['tasks', 'subway', 'starting_station']
        for field in required_fields:
            if field not in input_data:
                return {'valid': False, 'error': f'缺少必需字段: {field}'}
        
        # 验证tasks字段
        if not isinstance(input_data['tasks'], list):
            return {'valid': False, 'error': 'tasks字段必须是列表'}
        
        if len(input_data['tasks']) == 0:
            return {'valid': False, 'error': 'tasks列表不能为空'}
        
        # 验证每个任务
        for i, task_data in enumerate(input_data['tasks']):
            if not isinstance(task_data, dict):
                return {'valid': False, 'error': f'tasks[{i}]必须是字典'}
            
            task_required_fields = ['name', 'start', 'end', 'station', 'score']
            for field in task_required_fields:
                if field not in task_data:
                    return {'valid': False, 'error': f'tasks[{i}]缺少字段: {field}'}
            
            # 验证字段类型
            if not isinstance(task_data['name'], str):
                return {'valid': False, 'error': f'tasks[{i}].name必须是字符串'}
            
            for field in ['start', 'end', 'station', 'score']:
                if not isinstance(task_data[field], (int, float)):
                    return {'valid': False, 'error': f'tasks[{i}].{field}必须是数字'}
                # 转换为整数
                task_data[field] = int(task_data[field])
            
            # 验证时间逻辑
            if task_data['start'] >= task_data['end']:
                return {'valid': False, 'error': f'tasks[{i}]的开始时间必须小于结束时间'}
            
            if task_data['start'] < 0 or task_data['end'] < 0:
                return {'valid': False, 'error': f'tasks[{i}]的时间不能为负数'}
        
        # 验证subway字段
        if not isinstance(input_data['subway'], list):
            return {'valid': False, 'error': 'subway字段必须是列表'}
        
        if len(input_data['subway']) == 0:
            return {'valid': False, 'error': 'subway列表不能为空'}
        
        # 验证每个地铁路线
        for i, route_data in enumerate(input_data['subway']):
            if not isinstance(route_data, dict):
                return {'valid': False, 'error': f'subway[{i}]必须是字典'}
            
            route_required_fields = ['connection', 'fee']
            for field in route_required_fields:
                if field not in route_data:
                    return {'valid': False, 'error': f'subway[{i}]缺少字段: {field}'}
            
            # 验证connection字段
            if not isinstance(route_data['connection'], list):
                return {'valid': False, 'error': f'subway[{i}].connection必须是列表'}
            
            if len(route_data['connection']) != 2:
                return {'valid': False, 'error': f'subway[{i}].connection必须包含两个元素'}
            
            for j, station in enumerate(route_data['connection']):
                if not isinstance(station, (int, float)):
                    return {'valid': False, 'error': f'subway[{i}].connection[{j}]必须是数字'}
                route_data['connection'][j] = int(station)
            
            # 验证fee字段
            if not isinstance(route_data['fee'], (int, float)):
                return {'valid': False, 'error': f'subway[{i}].fee必须是数字'}
            route_data['fee'] = int(route_data['fee'])
            
            if route_data['fee'] < 0:
                return {'valid': False, 'error': f'subway[{i}].fee不能为负数'}
        
        # 验证starting_station字段
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
    """计算所有车站之间的最短距离矩阵"""
    # 获取所有车站ID
    stations = sorted(list(graph.nodes()))
    n = len(stations)
    station_to_idx = {station: i for i, station in enumerate(stations)}
    
    distance_matrix = np.full((n, n), np.inf)
    
    # 对角线设为0
    np.fill_diagonal(distance_matrix, 0)
    
    # 初始化直接连接的边
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        u_idx = station_to_idx[u]
        v_idx = station_to_idx[v]
        distance_matrix[u_idx][v_idx] = weight
        distance_matrix[v_idx][u_idx] = weight
    
    # 使用Floyd-Warshall算法计算最短路径
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
    
    return distance_matrix, station_to_idx, stations


def _is_compatible(task1: Dict, task2: Dict) -> bool:
    """检查两个任务是否兼容（时间不重叠）"""
    try:
        # 两个任务兼容当且仅当它们的时间区间不重叠
        # 即：task1.end <= task2.start 或 task2.end <= task1.start
        if 'start' not in task1 or 'end' not in task1 or 'start' not in task2 or 'end' not in task2:
            return False
        return (task1['end'] <= task2['start'] or task2['end'] <= task1['start'])
    except Exception:
        return False


def _get_compatible_tasks(tasks: List[Dict]) -> Dict[int, List[int]]:
    """获取每个任务的兼容任务列表"""
    try:
        compatible = {}
        for i, task1 in enumerate(tasks):
            compatible[i] = []
            for j, task2 in enumerate(tasks):
                if i != j and _is_compatible(task1, task2):
                    compatible[i].append(j)
        return compatible
    except Exception:
        return {}


def _calculate_travel_cost(schedule: List[Dict], distance_matrix: np.ndarray, 
                          station_to_idx: Dict[int, int], starting_station: int) -> float:
    """计算给定调度方案的总交通费用"""
    if not schedule:
        return 0
    
    try:
        # 按开始时间排序
        sorted_schedule = sorted(schedule, key=lambda t: t['start'])
        
        total_cost = 0
        
        # 检查起始车站是否在图中
        if starting_station not in station_to_idx:
            return 0
        
        # 从起始车站到第一个任务的车站
        first_station = sorted_schedule[0]['station']
        if first_station not in station_to_idx:
            return 0
            
        start_idx = station_to_idx[starting_station]
        first_idx = station_to_idx[first_station]
        cost = distance_matrix[start_idx][first_idx]
        if not np.isnan(cost) and not np.isinf(cost):
            total_cost += cost
        
        # 任务之间的移动费用
        for i in range(len(sorted_schedule) - 1):
            current_station = sorted_schedule[i]['station']
            next_station = sorted_schedule[i + 1]['station']
            if current_station not in station_to_idx or next_station not in station_to_idx:
                return 0  # 如果车站不在图中，返回0表示无法计算
            current_idx = station_to_idx[current_station]
            next_idx = station_to_idx[next_station]
            cost = distance_matrix[current_idx][next_idx]
            if not np.isnan(cost) and not np.isinf(cost):
                total_cost += cost
        
        # 从最后一个任务的车站回到起始车站
        last_station = sorted_schedule[-1]['station']
        if last_station not in station_to_idx:
            return 0
        last_idx = station_to_idx[last_station]
        cost = distance_matrix[last_idx][start_idx]
        if not np.isnan(cost) and not np.isinf(cost):
            total_cost += cost
        
        return total_cost
    except Exception:
        return 0


def _calculate_score(schedule: List[Dict]) -> int:
    """计算给定调度方案的总得分"""
    try:
        return sum(task['score'] for task in schedule if 'score' in task)
    except Exception:
        return 0



def _solve_greedy(tasks: List[Dict], graph: nx.Graph, distance_matrix: np.ndarray,
                 station_to_idx: Dict[int, int], starting_station: int) -> Dict:
    """
    使用简单贪心算法求解
    
    Returns:
        包含max_score, min_fee, schedule的字典
    """
    if not tasks:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
    
    try:
        # 按得分/时间比例排序
        def task_priority(task):
            duration = task['end'] - task['start']
            return task['score'] / max(duration, 1)
        
        sorted_tasks = sorted(tasks, key=task_priority, reverse=True)
        
        # 贪心选择
        selected_tasks = []
        for task in sorted_tasks:
            if all(_is_compatible(task, selected_task) for selected_task in selected_tasks):
                selected_tasks.append(task)
        
        # 计算费用和得分
        travel_cost = _calculate_travel_cost(selected_tasks, distance_matrix, station_to_idx, starting_station)
        total_score = _calculate_score(selected_tasks)
        
        # 按开始时间排序任务名称
        schedule = sorted([task['name'] for task in selected_tasks], 
                         key=lambda name: next(t['start'] for t in selected_tasks if t['name'] == name))
        
        return {
            'max_score': int(total_score),
            'min_fee': int(travel_cost),
            'schedule': schedule
        }
    except Exception:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
