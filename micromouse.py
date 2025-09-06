import math
from typing import List, Tuple, Dict, Optional
from enum import Enum


class Direction(Enum):
    """方向枚举：0=北, 1=东, 2=南, 3=西"""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class MouseState:
    """微鼠状态类"""
    def __init__(self, position=(0, 0), direction=Direction.NORTH, momentum=0, 
                 run_time_ms=0, reached_goal=False, run=0, best_time_ms=None):
        self.position = position  # (x, y) 位置坐标
        self.direction = direction  # 当前朝向
        self.momentum = momentum  # 动量值 [-4, +4]
        self.run_time_ms = run_time_ms  # 当前运行时间
        self.reached_goal = reached_goal  # 是否到达目标
        self.run = run  # 运行次数
        self.best_time_ms = best_time_ms  # 最佳时间

    @classmethod
    def from_dict(cls, d):
        """从字典创建状态"""
        return cls(
            position=tuple(d.get('position', (0, 0))),
            direction=Direction(d.get('direction', 0)),
            momentum=d.get('momentum', 0),
            run_time_ms=d.get('run_time_ms', 0),
            reached_goal=d.get('goal_reached', False),
            run=d.get('run', 0),
            best_time_ms=d.get('best_time_ms', None)
        )

    def to_dict(self):
        """转换为字典"""
        return {
            'position': self.position,
            'direction': self.direction.value,
            'momentum': self.momentum,
            'run_time_ms': self.run_time_ms,
            'goal_reached': self.reached_goal,
            'run': self.run,
            'best_time_ms': self.best_time_ms
        }


class MicroMouseController:
    """微鼠控制器主类"""
    
    def __init__(self):
        self.maze_size = 16
        self.goal_center = (7, 7)  # 目标区域中心
        self.goal_size = 2  # 目标区域大小 2x2
        self.start_position = (0, 0)  # 起始位置
        self.max_momentum = 4
        self.min_momentum = -4
        
        # 运动时间表（毫秒）
        self.base_times = {
            'in_place_turn': 200,
            'default_rest': 200,
            'half_step_cardinal': 500,
            'half_step_intercardinal': 600,
            'corner_tight': 700,
            'corner_wide': 1400
        }
        
        # 动量减少表
        self.momentum_reduction_table = {
            0.0: 0.00,
            0.5: 0.10,
            1.0: 0.20,
            1.5: 0.275,
            2.0: 0.35,
            2.5: 0.40,
            3.0: 0.45,
            3.5: 0.475,
            4.0: 0.50
        }
        
        # 迷宫探索状态
        self.explored_cells = set()
        self.wall_map = {}  # 记录墙壁信息
        self.visited_positions = set()
        self.current_position = (0, 0)
        self.current_direction = Direction.NORTH

    def get_momentum_reduction(self, m_eff: float) -> float:
        """获取动量减少百分比"""
        m_eff = max(0.0, min(4.0, m_eff))
        
        # 线性插值
        keys = sorted(self.momentum_reduction_table.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= m_eff <= keys[i + 1]:
                t = (m_eff - keys[i]) / (keys[i + 1] - keys[i])
                return self.momentum_reduction_table[keys[i]] + t * (
                    self.momentum_reduction_table[keys[i + 1]] - self.momentum_reduction_table[keys[i]]
                )
        
        return self.momentum_reduction_table[keys[-1]]

    def is_in_goal(self, position: Tuple[int, int]) -> bool:
        """检查是否在目标区域内"""
        x, y = position
        goal_x, goal_y = self.goal_center
        return (goal_x <= x <= goal_x + self.goal_size - 1 and 
                goal_y <= y <= goal_y + self.goal_size - 1)

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """检查位置是否在迷宫范围内"""
        x, y = position
        return 0 <= x < self.maze_size and 0 <= y < self.maze_size

    def get_wall_directions(self, sensor_data: List[int], direction: Direction) -> List[bool]:
        """解析传感器数据，返回各方向是否有墙壁"""
        # 传感器角度：-90°, -45°, 0°, +45°, +90°
        # 对应方向：西, 西南, 北, 东北, 东
        walls = [False] * 4  # [北, 东, 南, 西]
        
        if len(sensor_data) >= 5:
            # 前方传感器 (0°)
            if sensor_data[2] == 1:
                walls[direction.value] = True
            
            # 左前方传感器 (-45°)
            left_dir = (direction.value - 1) % 4
            if sensor_data[1] == 1:
                walls[left_dir] = True
            
            # 右前方传感器 (+45°)
            right_dir = (direction.value + 1) % 4
            if sensor_data[3] == 1:
                walls[right_dir] = True
            
            # 左侧传感器 (-90°)
            if sensor_data[0] == 1:
                walls[left_dir] = True
            
            # 右侧传感器 (+90°)
            if sensor_data[4] == 1:
                walls[right_dir] = True
        
        return walls

    def get_available_directions(self, position: Tuple[int, int], 
                               sensor_data: List[int], direction: Direction) -> List[Direction]:
        """获取可移动的方向"""
        walls = self.get_wall_directions(sensor_data, direction)
        available = []
        
        for dir_enum in Direction:
            if not walls[dir_enum.value]:
                # 检查目标位置是否有效
                next_pos = self.get_next_position(position, dir_enum)
                if self.is_valid_position(next_pos):
                    available.append(dir_enum)
        
        return available

    def get_next_position(self, position: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
        """获取指定方向的下一个位置"""
        x, y = position
        if direction == Direction.NORTH:
            return (x, y + 1)
        elif direction == Direction.EAST:
            return (x + 1, y)
        elif direction == Direction.SOUTH:
            return (x, y - 1)
        elif direction == Direction.WEST:
            return (x - 1, y)
        return position

    def calculate_distance_to_goal(self, position: Tuple[int, int]) -> int:
        """计算到目标的曼哈顿距离"""
        x, y = position
        goal_x, goal_y = self.goal_center
        return abs(x - goal_x) + abs(y - goal_y)

    def update_wall_map(self, position: Tuple[int, int], sensor_data: List[int], direction: Direction):
        """更新墙壁地图"""
        walls = self.get_wall_directions(sensor_data, direction)
        
        # 记录当前位置的墙壁信息
        self.wall_map[position] = walls
        self.explored_cells.add(position)
        self.visited_positions.add(position)

    def get_optimal_direction(self, position: Tuple[int, int], 
                            sensor_data: List[int], direction: Direction) -> Optional[Direction]:
        """获取最优移动方向（改进的探索策略）"""
        # 更新墙壁地图
        self.update_wall_map(position, sensor_data, direction)
        
        available_directions = self.get_available_directions(position, sensor_data, direction)
        
        if not available_directions:
            return None
        
        # 优先选择未探索的方向
        unexplored_directions = []
        for dir_enum in available_directions:
            next_pos = self.get_next_position(position, dir_enum)
            if next_pos not in self.explored_cells:
                unexplored_directions.append(dir_enum)
        
        # 如果有未探索的方向，优先选择
        if unexplored_directions:
            # 选择朝向目标最近的未探索方向
            best_direction = None
            min_distance = float('inf')
            
            for dir_enum in unexplored_directions:
                next_pos = self.get_next_position(position, dir_enum)
                distance = self.calculate_distance_to_goal(next_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    best_direction = dir_enum
            
            return best_direction
        
        # 如果没有未探索的方向，选择朝向目标最近的方向
        best_direction = None
        min_distance = float('inf')
        
        for dir_enum in available_directions:
            next_pos = self.get_next_position(position, dir_enum)
            distance = self.calculate_distance_to_goal(next_pos)
            
            if distance < min_distance:
                min_distance = distance
                best_direction = dir_enum
        
        return best_direction

    def generate_movement_instructions(self, state: MouseState, 
                                     sensor_data: List[int]) -> List[str]:
        """生成移动指令"""
        instructions = []
        
        # 更新当前位置和方向
        self.current_position = state.position
        self.current_direction = state.direction
        
        # 如果已经到达目标且动量为0，停止
        if state.reached_goal and state.momentum == 0:
            return instructions
        
        # 如果当前在目标区域内且动量为0，标记为到达
        if self.is_in_goal(state.position) and state.momentum == 0:
            return instructions
        
        # 获取最优方向
        optimal_direction = self.get_optimal_direction(state.position, sensor_data, state.direction)
        
        if optimal_direction is None:
            # 没有可用方向，尝试转向
            if state.momentum == 0:
                instructions.append('L')  # 尝试左转
            else:
                instructions.append('BB')  # 刹车
            return instructions
        
        # 计算需要转向的角度
        current_dir = state.direction.value
        target_dir = optimal_direction.value
        
        # 计算转向角度
        turn_angle = (target_dir - current_dir) % 4
        
        # 如果不需要转向
        if turn_angle == 0:
            # 根据当前动量和目标优化速度
            if state.momentum < 1:
                instructions.append('F2')  # 加速
            elif state.momentum > 3:
                instructions.append('F0')  # 减速
            else:
                instructions.append('F1')  # 保持
        else:
            # 需要转向
            if state.momentum != 0:
                # 先刹车到0
                instructions.append('BB')
                return instructions
            
            # 执行转向
            if turn_angle == 1:
                instructions.append('R')
            elif turn_angle == 2:
                instructions.append('R')
                instructions.append('R')
            elif turn_angle == 3:
                instructions.append('L')
            
            # 转向后加速
            instructions.append('F2')
        
        return instructions

    def should_end_challenge(self, state: MouseState, total_time_ms: int) -> bool:
        """判断是否应该结束挑战"""
        # 如果已经到达目标且动量为0
        if state.reached_goal and state.momentum == 0:
            return True
        
        # 如果时间预算用完
        if total_time_ms >= 300000:  # 300秒
            return True
        
        return False

    def reset_exploration(self):
        """重置探索状态（新运行开始时调用）"""
        self.explored_cells.clear()
        self.wall_map.clear()
        self.visited_positions.clear()
        self.current_position = (0, 0)
        self.current_direction = Direction.NORTH

    def process_request(self, request_data: Dict) -> Dict:
        """处理API请求"""
        try:
            # 解析请求数据
            game_uuid = request_data.get('game_uuid', '')
            sensor_data = request_data.get('sensor_data', [])
            total_time_ms = request_data.get('total_time_ms', 0)
            goal_reached = request_data.get('goal_reached', False)
            best_time_ms = request_data.get('best_time_ms')
            run_time_ms = request_data.get('run_time_ms', 0)
            run = request_data.get('run', 0)
            momentum = request_data.get('momentum', 0)
            
            # 检查是否是新运行（run_time_ms为0且动量为0）
            if run_time_ms == 0 and momentum == 0:
                self.reset_exploration()
            
            # 创建状态对象
            state = MouseState(
                position=self.current_position,
                direction=self.current_direction,
                momentum=momentum,
                run_time_ms=run_time_ms,
                reached_goal=goal_reached,
                run=run,
                best_time_ms=best_time_ms
            )
            
            # 检查是否应该结束
            if self.should_end_challenge(state, total_time_ms):
                return {
                    'instructions': [],
                    'end': True
                }
            
            # 生成移动指令
            instructions = self.generate_movement_instructions(state, sensor_data)
            
            return {
                'instructions': instructions,
                'end': False
            }
            
        except Exception as e:
            # 发生错误时返回空指令
            return {
                'instructions': [],
                'end': True
            }


# 全局控制器实例
controller = MicroMouseController()


def process_micromouse_request(request_data: Dict) -> Dict:
    """处理微鼠请求的主函数"""
    return controller.process_request(request_data)
