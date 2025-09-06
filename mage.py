import json

def calculate_earliest_time(data):
    """
    计算法师克莱因击败所有不死生物所需的最早时间。

    Args:
        data (dict): 包含战斗信息的字典，键包括 'intel', 'reserve', 'fronts', 'stamina'。

    Returns:
        int: 完成所有战斗并进入冷却状态所需的最早时间（分钟）。
    """
    # 提取输入数据
    intel_sequence = data["intel"]
    mana_reserve = data["reserve"]
    stamina_limit = data["stamina"]

    # 初始化状态变量
    total_time = 0
    current_mana = mana_reserve
    current_stamina = stamina_limit
    last_front = -1  # 使用 -1 作为初始值，确保第一次攻击不被视为“扩展”AOE
    last_stage_rest = False

    # 遍历不死生物序列
    for front, mana_cost in intel_sequence:
        # 在施法前检查是否需要进入冷却状态
        # 冷却条件1: 法力不足以施展当前法术
        if current_mana < mana_cost:
            total_time += 10  # 增加10分钟的冷却时间
            current_mana = mana_reserve  # 法力恢复至最大值
            current_stamina = stamina_limit  # 体力恢复至最大值
            last_stage_rest = True
        # 冷却条件2: 体力耗尽
        elif current_stamina == 0:
            total_time += 10
            current_mana = mana_reserve
            current_stamina = stamina_limit
            last_stage_rest = True

        # 施展法术
        # 检查是否为“扩展”AOE攻击（与上一个法术在同一地点）
        if front == last_front and not last_stage_rest:
            # 如果是同一地点且不是刚休息完，不增加施法时间
            pass
        else:
            # 如果是新地点，增加10分钟的施法时间
            total_time+=10
        
        current_stamina-=1
        current_mana-=mana_cost
        last_front = front  # 更新上一个法术的地点
        last_stage_rest = False

    # 最后还要再休息一次
    total_time+=10
    result = {'time':total_time}
    return result

def process(json_data):
    ans = list()
    for d in json_data:
        ans.append(calculate_earliest_time(d))
    # json_output = json.dumps(ans, indent=2)
    return ans

# 示例输入数据
input_data = [
    {
      "intel": [[1,2],[2,3],[2,1],[2,4]],
      "reserve": 4,
      "fronts": 3,
      "stamina": 2
    },
    {
      "intel": [[2,1],[4,2],[4,2],[1,3]],
      "reserve": 3,
      "fronts": 5,
      "stamina": 4
    }
]

# 运行测试用例
for case in input_data:
    result = calculate_earliest_time(case)
    print(f"输入数据: {case}")
    print(f"最早时间: {result} 分钟")
    print("---")    