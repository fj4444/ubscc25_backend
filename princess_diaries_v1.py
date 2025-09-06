"""
Princess Diaries 任务调度优化算法
解决地铁系统中的任务调度和最短路径问题
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from itertools import combinations
import heapq


@dataclass
class Task:
    """任务数据结构"""
    name: str       # 任务名称
    start: int      # 开始时间（从第1天午夜开始的分钟数）
    end: int        # 结束时间（从第1天午夜开始的分钟数）
    station: int    # 车站ID
    score: int      # 得分


@dataclass
class Route:
    """地铁路线数据结构"""
    connection: List[int]  # 连接的两个车站ID
    fee: int              # 交通费用


@dataclass
class Input:
    """输入数据结构"""
    tasks: List[Task]           # 任务列表
    subway: List[Route]         # 地铁路线列表
    starting_station: int       # 起始车站ID


@dataclass
class Output:
    """输出数据结构"""
    max_score: int              # 最大得分
    min_fee: int                # 最小交通费用
    schedule: List[str]         # 任务调度（按开始时间排序的任务名称）


class PrincessScheduler:
    """公主任务调度器"""
    
    def __init__(self, input_data: Input):
        """
        初始化调度器
        
        Args:
            input_data: 包含任务、地铁和起始车站的输入数据
        """
        self.input_data = input_data
        self.tasks = input_data.tasks
        self.starting_station = input_data.starting_station
        self.graph = self._build_subway_graph()
        self.distance_matrix = self._compute_distance_matrix()
    
    def _build_subway_graph(self) -> nx.Graph:
        """根据subway数据构建地铁图"""
        G = nx.Graph()
        
        # 添加所有车站节点
        all_stations = set()
        for route in self.input_data.subway:
            all_stations.update(route.connection)
        
        for station in all_stations:
            G.add_node(station)
        
        # 添加连接和费用
        for route in self.input_data.subway:
            u, v = route.connection
            fee = route.fee
            G.add_edge(u, v, weight=fee)
        
        return G
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """计算所有车站之间的最短距离矩阵"""
        # 获取所有车站ID
        stations = sorted(list(self.graph.nodes()))
        n = len(stations)
        station_to_idx = {station: i for i, station in enumerate(stations)}
        
        distance_matrix = np.full((n, n), np.inf)
        
        # 对角线设为0
        np.fill_diagonal(distance_matrix, 0)
        
        # 初始化直接连接的边
        for u, v, data in self.graph.edges(data=True):
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
        
        # 存储车站ID到索引的映射
        self.station_to_idx = station_to_idx
        self.idx_to_station = stations
        
        return distance_matrix
    
    def _is_compatible(self, task1: Task, task2: Task) -> bool:
        """检查两个任务是否兼容（时间不重叠）"""
        # 两个任务兼容当且仅当它们的时间区间不重叠
        # 即：task1.end <= task2.start 或 task2.end <= task1.start
        return (task1.end <= task2.start or task2.end <= task1.start)
    
    def _get_compatible_tasks(self, tasks: List[Task]) -> Dict[int, List[int]]:
        """获取每个任务的兼容任务列表"""
        compatible = {}
        for i, task1 in enumerate(tasks):
            compatible[i] = []
            for j, task2 in enumerate(tasks):
                if i != j and self._is_compatible(task1, task2):
                    compatible[i].append(j)
        return compatible
    
    def _calculate_travel_cost(self, schedule: List[Task]) -> float:
        """计算给定调度方案的总交通费用"""
        if not schedule:
            return 0
        
        # 按开始时间排序
        sorted_schedule = sorted(schedule, key=lambda t: t.start)
        
        total_cost = 0
        
        # 从起始车站到第一个任务的车站
        first_station = sorted_schedule[0].station
        start_idx = self.station_to_idx[self.starting_station]
        first_idx = self.station_to_idx[first_station]
        total_cost += self.distance_matrix[start_idx][first_idx]
        
        # 任务之间的移动费用
        for i in range(len(sorted_schedule) - 1):
            current_station = sorted_schedule[i].station
            next_station = sorted_schedule[i + 1].station
            current_idx = self.station_to_idx[current_station]
            next_idx = self.station_to_idx[next_station]
            total_cost += self.distance_matrix[current_idx][next_idx]
        
        # 从最后一个任务的车站回到起始车站
        last_station = sorted_schedule[-1].station
        last_idx = self.station_to_idx[last_station]
        total_cost += self.distance_matrix[last_idx][start_idx]
        
        return total_cost
    
    def _calculate_score(self, schedule: List[Task]) -> int:
        """计算给定调度方案的总得分"""
        return sum(task.score for task in schedule)
    
    def solve_optimal_schedule(self) -> Output:
        """
        求解最优调度方案 - 使用动态规划求解最大权重独立集问题
        
        Returns:
            Output对象包含最大得分、最小交通费用和任务调度
        """
        if not self.tasks:
            return Output(max_score=0, min_fee=0, schedule=[])
        
        # 按结束时间排序任务
        sorted_tasks = sorted(self.tasks, key=lambda t: t.end)
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
                if self._is_compatible(sorted_tasks[j - 1], current_task):
                    last_compatible = j - 1
                    break
            
            # 计算选择当前任务的得分
            if last_compatible == -1:
                # 没有兼容的前置任务，可以选择当前任务
                score_with_current = current_task.score
            else:
                # 有兼容的前置任务
                score_with_current = dp[last_compatible + 1] + current_task.score
            
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
                    if self._is_compatible(sorted_tasks[j - 1], sorted_tasks[i - 1]):
                        last_compatible = j - 1
                        break
                if last_compatible == -1:
                    break
                else:
                    i = last_compatible + 1
            else:
                i = i - 1
        
        # 计算交通费用
        travel_cost = self._calculate_travel_cost(optimal_tasks)
        
        # 按开始时间排序任务名称
        schedule = sorted([task.name for task in optimal_tasks], 
                         key=lambda name: next(t.start for t in optimal_tasks if t.name == name))
        
        return Output(
            max_score=int(dp[n]),
            min_fee=int(travel_cost),
            schedule=schedule
        )
    
    def solve_with_heuristic(self, max_iterations: int = 1000) -> Output:
        """
        使用启发式算法求解（贪心 + 局部搜索）
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            Output对象包含最大得分、最小交通费用和任务调度
        """
        if not self.tasks:
            return Output(max_score=0, min_fee=0, schedule=[])
        
        # 贪心算法：按权重/时间比例排序
        def task_priority(task):
            duration = task.end - task.start
            return task.score / max(duration, 1)
        
        sorted_tasks = sorted(self.tasks, key=task_priority, reverse=True)
        
        # 贪心选择
        current_schedule = []
        for task in sorted_tasks:
            if all(self._is_compatible(task, scheduled_task) for scheduled_task in current_schedule):
                current_schedule.append(task)
        
        best_schedule = current_schedule.copy()
        best_score = self._calculate_score(best_schedule)
        best_cost = self._calculate_travel_cost(best_schedule)
        
        # 局部搜索优化
        for _ in range(max_iterations):
            # 尝试添加新任务
            for task in self.tasks:
                if task not in current_schedule:
                    # 检查是否可以添加
                    can_add = all(self._is_compatible(task, scheduled_task) for scheduled_task in current_schedule)
                    if can_add:
                        new_schedule = current_schedule + [task]
                        new_score = self._calculate_score(new_schedule)
                        new_cost = self._calculate_travel_cost(new_schedule)
                        
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
                for task in self.tasks:
                    if task != removed_task and task not in current_schedule:
                        can_add = all(self._is_compatible(task, scheduled_task) for scheduled_task in current_schedule)
                        if can_add:
                            new_schedule = current_schedule + [task]
                            new_score = self._calculate_score(new_schedule)
                            new_cost = self._calculate_travel_cost(new_schedule)
                            
                            if (new_score > best_score or 
                                (new_score == best_score and new_cost < best_cost)):
                                best_schedule = new_schedule.copy()
                                best_score = new_score
                                best_cost = new_cost
                                current_schedule = new_schedule.copy()
                                break
                
                # 如果没找到更好的，恢复移除的任务
                if removed_task not in current_schedule:
                    can_add = all(self._is_compatible(removed_task, scheduled_task) for scheduled_task in current_schedule)
                    if can_add:
                        current_schedule.append(removed_task)
        
        # 按开始时间排序任务名称
        schedule = sorted([task.name for task in best_schedule], 
                         key=lambda name: next(t.start for t in best_schedule if t.name == name))
        
        return Output(
            max_score=int(best_score),
            min_fee=int(best_cost),
            schedule=schedule
        )


def solve_princess_diaries(input_data: dict) -> dict:
    """
    解决Princess Diaries问题的入口函数
    
    Args:
        input_data: 包含tasks, subway, starting_station的字典
        
    Returns:
        包含max_score, min_fee, schedule的字典
    """
    # 解析输入数据并创建Task对象
    tasks = []
    for task_data in input_data['tasks']:
        task = Task(
            name=task_data['name'],
            start=task_data['start'],
            end=task_data['end'],
            station=task_data['station'],
            score=task_data['score']
        )
        tasks.append(task)
    
    # 创建Route对象
    routes = []
    for route_data in input_data['subway']:
        route = Route(
            connection=route_data['connection'],
            fee=route_data['fee']
        )
        routes.append(route)
    
    # 创建Input对象
    input_obj = Input(
        tasks=tasks,
        subway=routes,
        starting_station=input_data['starting_station']
    )
    
    # 创建调度器并求解
    scheduler = PrincessScheduler(input_obj)
    output = scheduler.solve_optimal_schedule()
    
    # 返回字典格式的结果
    return {
        'max_score': output.max_score,
        'min_fee': output.min_fee,
        'schedule': output.schedule
    }


def create_sample_input() -> Input:
    """创建示例输入数据"""
    # 创建示例任务
    tasks = [
        Task(name="A", start=480, end=540, station=1, score=2),
        Task(name="B", start=600, end=660, station=2, score=1),
        Task(name="C", start=720, end=780, station=3, score=3),
        Task(name="D", start=840, end=900, station=4, score=1),
        Task(name="E", start=960, end=1020, station=1, score=4),
        Task(name="F", start=530, end=590, station=2, score=1),
    ]
    
    
    # 创建示例地铁路线
    routes = [
        Route(connection=[0, 1], fee=10),
        Route(connection=[1, 2], fee=10),
        Route(connection=[2, 3], fee=20),
        Route(connection=[3, 4], fee=30),
    ]
    
    return Input(
        tasks=tasks,
        subway=routes,
        starting_station=0
    )


if __name__ == "__main__":
    # 测试算法
    sample_input = create_sample_input()
    
    scheduler = PrincessScheduler(sample_input)
    
    print("=== Princess Diaries 任务调度优化 ===")
    print(f"地铁图节点数: {len(scheduler.graph.nodes())}")
    print(f"地铁图边数: {len(scheduler.graph.edges())}")
    print(f"任务数量: {len(sample_input.tasks)}")
    print(f"起始车站: {sample_input.starting_station}")
    print()
    
    # 使用动态规划求解
    optimal_output = scheduler.solve_optimal_schedule()
    
    print("=== 最优解（动态规划）===")
    print(f"最大得分: {optimal_output.max_score}")
    print(f"最小交通费用: {optimal_output.min_fee}")
    print(f"选择的任务数量: {len(optimal_output.schedule)}")
    print("选择的任务:")
    for task_name in optimal_output.schedule:
        task = next(t for t in sample_input.tasks if t.name == task_name)
        print(f"  {task.name}: 车站{task.station}, 得分{task.score}, 时间[{task.start}-{task.end}]")
    
    print()
    
    # 使用启发式算法求解
    heuristic_output = scheduler.solve_with_heuristic()
    
    print("=== 启发式解 ===")
    print(f"最大得分: {heuristic_output.max_score}")
    print(f"最小交通费用: {heuristic_output.min_fee}")
    print(f"选择的任务数量: {len(heuristic_output.schedule)}")
    print("选择的任务:")
    for task_name in heuristic_output.schedule:
        task = next(t for t in sample_input.tasks if t.name == task_name)
        print(f"  {task.name}: 车站{task.station}, 得分{task.score}, 时间[{task.start}-{task.end}]")
    
    print()
    
    # 测试主函数
    print("=== 测试主函数 ===")
    input_dict = {
        'tasks': [
            {'name': task.name, 'start': task.start, 'end': task.end, 'station': task.station, 'score': task.score}
            for task in sample_input.tasks
        ],
        'subway': [
            {'connection': route.connection, 'fee': route.fee}
            for route in sample_input.subway
        ],
        'starting_station': sample_input.starting_station
    }
    
    result = solve_princess_diaries(input_dict)
    print(f"主函数结果: {result}")
