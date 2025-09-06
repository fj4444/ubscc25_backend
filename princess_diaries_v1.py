import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set
from itertools import combinations
import heapq


def princess_diaries(input_data: dict) -> dict:
    """
    公主日记任务调度优化 - 主要结果输出函数
    
    这个函数是解决Princess Diaries问题的核心入口，使用动态规划算法
    求解在时间约束下的最优任务调度方案，最大化得分并最小化交通费用。
    
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
                'error': f"输入数据格式错误: {validation_result['error']}",
                'max_score': 0,
                'min_fee': 0,
                'schedule': []
            }
        
        # 解析输入数据
        tasks = []
        for task_data in input_data['tasks']:
            task = {
                'name': str(task_data['name']),
                'start': int(task_data['start']),
                'end': int(task_data['end']),
                'station': int(task_data['station']),
                'score': int(task_data['score'])
            }
            tasks.append(task)
        
        # 创建地铁路线数据
        routes = []
        for route_data in input_data['subway']:
            route = {
                'connection': [int(x) for x in route_data['connection']],
                'fee': int(route_data['fee'])
            }
            routes.append(route)
        
        starting_station = int(input_data['starting_station'])
        
        # 构建地铁图
        graph = _build_subway_graph(routes)
        
        # 计算距离矩阵
        distance_matrix, station_to_idx, idx_to_station = _compute_distance_matrix(graph)
        
        # 使用动态规划求解最优解
        optimal_result = _solve_optimal_schedule(tasks, graph, distance_matrix, station_to_idx, starting_station)
        
        # 如果动态规划解不够好，尝试启发式算法
        if len(optimal_result['schedule']) == 0 or optimal_result['max_score'] == 0:
            heuristic_result = _solve_with_heuristic(tasks, graph, distance_matrix, station_to_idx, starting_station)
            if heuristic_result['max_score'] > optimal_result['max_score']:
                optimal_result = heuristic_result
        
        # 返回字典格式的结果
        return {
            'max_score': optimal_result['max_score'],
            'min_fee': optimal_result['min_fee'],
            'schedule': optimal_result['schedule']
        }
        
    except Exception as e:
        # 错误处理
        return {
            'error': f"处理输入数据时发生错误: {str(e)}",
            'max_score': 0,
            'min_fee': 0,
            'schedule': []
        }


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
    # 两个任务兼容当且仅当它们的时间区间不重叠
    # 即：task1.end <= task2.start 或 task2.end <= task1.start
    return (task1['end'] <= task2['start'] or task2['end'] <= task1['start'])


def _get_compatible_tasks(tasks: List[Dict]) -> Dict[int, List[int]]:
    """获取每个任务的兼容任务列表"""
    compatible = {}
    for i, task1 in enumerate(tasks):
        compatible[i] = []
        for j, task2 in enumerate(tasks):
            if i != j and _is_compatible(task1, task2):
                compatible[i].append(j)
    return compatible


def _calculate_travel_cost(schedule: List[Dict], distance_matrix: np.ndarray, 
                          station_to_idx: Dict[int, int], starting_station: int) -> float:
    """计算给定调度方案的总交通费用"""
    if not schedule:
        return 0
    
    # 按开始时间排序
    sorted_schedule = sorted(schedule, key=lambda t: t['start'])
    
    total_cost = 0
    
    # 从起始车站到第一个任务的车站
    first_station = sorted_schedule[0]['station']
    start_idx = station_to_idx[starting_station]
    first_idx = station_to_idx[first_station]
    total_cost += distance_matrix[start_idx][first_idx]
    
    # 任务之间的移动费用
    for i in range(len(sorted_schedule) - 1):
        current_station = sorted_schedule[i]['station']
        next_station = sorted_schedule[i + 1]['station']
        current_idx = station_to_idx[current_station]
        next_idx = station_to_idx[next_station]
        total_cost += distance_matrix[current_idx][next_idx]
    
    # 从最后一个任务的车站回到起始车站
    last_station = sorted_schedule[-1]['station']
    last_idx = station_to_idx[last_station]
    total_cost += distance_matrix[last_idx][start_idx]
    
    return total_cost


def _calculate_score(schedule: List[Dict]) -> int:
    """计算给定调度方案的总得分"""
    return sum(task['score'] for task in schedule)


def _solve_optimal_schedule(tasks: List[Dict], graph: nx.Graph, distance_matrix: np.ndarray,
                           station_to_idx: Dict[int, int], starting_station: int) -> Dict:
    """
    求解最优调度方案 - 使用动态规划求解最大权重独立集问题
    
    Returns:
        包含max_score, min_fee, schedule的字典
    """
    if not tasks:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
    
    # 按结束时间排序任务
    sorted_tasks = sorted(tasks, key=lambda t: t['end'])
    n = len(sorted_tasks)
    
    # dp[i] = 考虑前i个任务时的最大得分
    dp = [0] * (n + 1)
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
        
        # 计算选择当前任务的得分
        if last_compatible == -1:
            # 没有兼容的前置任务，可以选择当前任务
            score_with_current = current_task['score']
        else:
            # 有兼容的前置任务
            score_with_current = dp[last_compatible + 1] + current_task['score']
        
        # 如果选择当前任务更好
        if score_with_current > dp[i]:
            dp[i] = score_with_current
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
    
    # 计算交通费用
    travel_cost = _calculate_travel_cost(optimal_tasks, distance_matrix, station_to_idx, starting_station)
    
    # 按开始时间排序任务名称
    schedule = sorted([task['name'] for task in optimal_tasks], 
                     key=lambda name: next(t['start'] for t in optimal_tasks if t['name'] == name))
    
    return {
        'max_score': int(dp[n]),
        'min_fee': int(travel_cost),
        'schedule': schedule
    }


def _solve_with_heuristic(tasks: List[Dict], graph: nx.Graph, distance_matrix: np.ndarray,
                         station_to_idx: Dict[int, int], starting_station: int, 
                         max_iterations: int = 1000) -> Dict:
    """
    使用启发式算法求解（贪心 + 局部搜索）
    
    Args:
        max_iterations: 最大迭代次数
        
    Returns:
        包含max_score, min_fee, schedule的字典
    """
    if not tasks:
        return {'max_score': 0, 'min_fee': 0, 'schedule': []}
    
    # 贪心算法：按权重/时间比例排序
    def task_priority(task):
        duration = task['end'] - task['start']
        return task['score'] / max(duration, 1)
    
    sorted_tasks = sorted(tasks, key=task_priority, reverse=True)
    
    # 贪心选择
    current_schedule = []
    for task in sorted_tasks:
        if all(_is_compatible(task, scheduled_task) for scheduled_task in current_schedule):
            current_schedule.append(task)
    
    best_schedule = current_schedule.copy()
    best_score = _calculate_score(best_schedule)
    best_cost = _calculate_travel_cost(best_schedule, distance_matrix, station_to_idx, starting_station)
    
    # 局部搜索优化
    for _ in range(max_iterations):
        # 尝试添加新任务
        for task in tasks:
            if task not in current_schedule:
                # 检查是否可以添加
                can_add = all(_is_compatible(task, scheduled_task) for scheduled_task in current_schedule)
                if can_add:
                    new_schedule = current_schedule + [task]
                    new_score = _calculate_score(new_schedule)
                    new_cost = _calculate_travel_cost(new_schedule, distance_matrix, station_to_idx, starting_station)
                    
                    # 如果得分更高，或者得分相同但费用更低
                    if (new_score > best_score or 
                        (new_score == best_score and new_cost < best_cost)):
                        best_schedule = new_schedule.copy()
                        best_score = new_score
                        best_cost = new_cost
                        current_schedule = new_schedule.copy()
                        break
        
        # 尝试移除任务并添加其他任务
        if len(current_schedule) > 0:
            # 随机移除一个任务
            remove_idx = np.random.randint(0, len(current_schedule))
            removed_task = current_schedule.pop(remove_idx)
            
            # 尝试添加其他任务
            for task in tasks:
                if task != removed_task and task not in current_schedule:
                    can_add = all(_is_compatible(task, scheduled_task) for scheduled_task in current_schedule)
                    if can_add:
                        new_schedule = current_schedule + [task]
                        new_score = _calculate_score(new_schedule)
                        new_cost = _calculate_travel_cost(new_schedule, distance_matrix, station_to_idx, starting_station)
                        
                        if (new_score > best_score or 
                            (new_score == best_score and new_cost < best_cost)):
                            best_schedule = new_schedule.copy()
                            best_score = new_score
                            best_cost = new_cost
                            current_schedule = new_schedule.copy()
                            break
            
            # 如果没找到更好的，恢复移除的任务
            if removed_task not in current_schedule:
                can_add = all(_is_compatible(removed_task, scheduled_task) for scheduled_task in current_schedule)
                if can_add:
                    current_schedule.append(removed_task)
    
    # 按开始时间排序任务名称
    schedule = sorted([task['name'] for task in best_schedule], 
                     key=lambda name: next(t['start'] for t in best_schedule if t['name'] == name))
    
    return {
        'max_score': int(best_score),
        'min_fee': int(best_cost),
        'schedule': schedule
    }


# 向后兼容函数
def solve_princess_diaries(input_data: dict) -> dict:
    """
    解决Princess Diaries问题的入口函数（向后兼容）
    
    Args:
        input_data: 包含tasks, subway, starting_station的字典
        
    Returns:
        包含max_score, min_fee, schedule的字典
    """
    return princess_diaries(input_data)


def create_sample_input() -> Dict:
    """创建示例输入数据"""
    # 创建示例任务
    tasks = [
        {'name': "A", 'start': 480, 'end': 540, 'station': 1, 'score': 2},
        {'name': "B", 'start': 600, 'end': 660, 'station': 2, 'score': 1},
        {'name': "C", 'start': 720, 'end': 780, 'station': 3, 'score': 3},
        {'name': "D", 'start': 840, 'end': 900, 'station': 4, 'score': 1},
        {'name': "E", 'start': 960, 'end': 1020, 'station': 1, 'score': 4},
        {'name': "F", 'start': 530, 'end': 590, 'station': 2, 'score': 1},
    ]
    
    # 创建示例地铁路线
    routes = [
        {'connection': [0, 1], 'fee': 10},
        {'connection': [1, 2], 'fee': 10},
        {'connection': [2, 3], 'fee': 20},
        {'connection': [3, 4], 'fee': 30},
    ]
    
    return {
        'tasks': tasks,
        'subway': routes,
        'starting_station': 0
    }


if __name__ == "__main__":
    # 测试算法
    sample_input = create_sample_input()
    
    # 构建地铁图
    graph = _build_subway_graph(sample_input['subway'])
    distance_matrix, station_to_idx, idx_to_station = _compute_distance_matrix(graph)
    
    print("=== Princess Diaries 任务调度优化 ===")
    print(f"地铁图节点数: {len(graph.nodes())}")
    print(f"地铁图边数: {len(graph.edges())}")
    print(f"任务数量: {len(sample_input['tasks'])}")
    print(f"起始车站: {sample_input['starting_station']}")
    print()
    
    # 使用动态规划求解
    optimal_output = _solve_optimal_schedule(sample_input['tasks'], graph, distance_matrix, station_to_idx, sample_input['starting_station'])
    
    print("=== 最优解（动态规划）===")
    print(f"最大得分: {optimal_output['max_score']}")
    print(f"最小交通费用: {optimal_output['min_fee']}")
    print(f"选择的任务数量: {len(optimal_output['schedule'])}")
    print("选择的任务:")
    for task_name in optimal_output['schedule']:
        task = next(t for t in sample_input['tasks'] if t['name'] == task_name)
        print(f"  {task['name']}: 车站{task['station']}, 得分{task['score']}, 时间[{task['start']}-{task['end']}]")
    
    print()
    
    # 使用启发式算法求解
    heuristic_output = _solve_with_heuristic(sample_input['tasks'], graph, distance_matrix, station_to_idx, sample_input['starting_station'])
    
    print("=== 启发式解 ===")
    print(f"最大得分: {heuristic_output['max_score']}")
    print(f"最小交通费用: {heuristic_output['min_fee']}")
    print(f"选择的任务数量: {len(heuristic_output['schedule'])}")
    print("选择的任务:")
    for task_name in heuristic_output['schedule']:
        task = next(t for t in sample_input['tasks'] if t['name'] == task_name)
        print(f"  {task['name']}: 车站{task['station']}, 得分{task['score']}, 时间[{task['start']}-{task['end']}]")
    
    print()
    
    # 测试主函数
    print("=== 测试 princess_diaries 主函数 ===")
    result = princess_diaries(sample_input)
    print(f"princess_diaries 主函数结果: {result}")
    
    # 测试向后兼容函数
    print("\n=== 测试向后兼容函数 ===")
    result2 = solve_princess_diaries(sample_input)
    print(f"solve_princess_diaries 结果: {result2}")