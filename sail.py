def merge_and_count_boats(bookings):
    """
    合并重叠的预订时段并计算所需的最小船只数量。

    该函数首先对预订进行排序，然后遍历它们以合并重叠的时段。
    同时，它使用一个扫描线算法来计算在任何给定时间点上最大
    的重叠数量，从而确定所需的最小船只数量。

    Args:
        bookings (list[list[int]]): 包含预订开始和结束时间的一系列列表。

    Returns:
        tuple: 包含两个元素的元组：
               - sorted_merged_slots (list[list[int]]): 合并后的、排序的无重叠时段列表。
               - min_boats_needed (int): 满足所有预订所需的最小船只数量。
    """
    if not bookings:
        return [], 0

    # Part 1: 合并重叠的预订时段
    # 按开始时间对预订进行排序
    bookings.sort(key=lambda x: x[0])
    
    merged_slots = []
    current_start, current_end = bookings[0]

    for next_start, next_end in bookings[1:]:
        # 如果当前时段与下一个时段重叠（或紧挨着），则扩展当前时段的结束时间
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            # 如果没有重叠，将当前时段添加到结果中，并开始一个新的时段
            merged_slots.append([current_start, current_end])
            current_start, current_end = next_start, next_end
    
    # 添加最后一个合并后的时段
    merged_slots.append([current_start, current_end])
    
    # Part 2: 计算所需的最小船只数量
    # 创建一个事件列表，其中包含开始和结束事件
    events = []
    for start, end in bookings:
        events.append((start, 1))  # 1 表示开始事件（增加一个预订）
        events.append((end, -1))   # -1 表示结束事件（减少一个预订）

    # 按时间对事件进行排序。如果时间相同，优先处理开始事件（+1）
    events.sort(key=lambda x: (x[0], x[1]))

    min_boats_needed = 0
    current_boats = 0
    for _, event_type in events:
        current_boats += event_type
        min_boats_needed = max(min_boats_needed, current_boats)
    
    return merged_slots, min_boats_needed

import json
class SailingClubHandler:
    """
    处理帆船俱乐部预订请求的类,用于模拟API的输入输出格式。
    """
    def handle_request(self, request_body_json):
        """
        根据给定的请求体JSON,处理所有测试用例并生成响应JSON。

        Args:
            request_body_json (str): 包含 'testCases' 键的JSON字符串。

        Returns:
            str: 包含 'solutions' 的响应JSON字符串。
        """
        # print(f"request_body_json:{request_body_json}")
        try:

            request_data = request_body_json
            test_cases = request_data.get("testCases", [])
            
            solutions = []
            for case in test_cases:
                case_id = case.get("id")
                input_bookings = case.get("input", [])

                sorted_merged_slots, min_boats_needed = merge_and_count_boats(input_bookings)

                solution = {
                    "id": case_id,
                    "sortedMergedSlots": sorted_merged_slots,
                    "minBoatsNeeded": min_boats_needed
                }
                solutions.append(solution)

            response_data = {"solutions": solutions}
            # return json.dumps(response_data, indent=2)
            return response_data

        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON format: {e}"})
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {e}"})


def run_tests():
    """
    运行 PDF 中提供的所有测试用例。
    """
    test_cases = [
        {
            "id": "0001",
            "input": [[1, 8], [17, 28], [5, 8], [8, 10]],
            "expected_merged": [[1, 10], [17, 28]],
            "expected_boats": 2
        },
        {
            "id": "0000",
            "input": [[15, 28], [49, 57], [8, 13], [51, 62], [16, 28], [66, 73], [83, 94], [44, 62], [69, 70]],
            "expected_merged": [[8, 13], [15, 28], [44, 62], [66, 73], [83, 94]],
            "expected_boats": 3
        },
        {
            "id": "0004",
            "input": [[45, 62], [53, 62], [53, 62], [46, 48], [78, 86], [72, 73], [80, 90], [47, 54], [77, 90]],
            "expected_merged": [[45, 62], [72, 73], [77, 90]],
            "expected_boats": 4
        },
        # 添加自定义测试用例以进一步验证
        {
            "id": "Custom 1",
            "input": [[1, 5], [2, 6], [8, 10], [15, 18], [12, 16]],
            "expected_merged": [[1, 6], [8, 10], [12, 18]],
            "expected_boats": 2
        },
        {
            "id": "Custom 2",
            "input": [[1, 10]],
            "expected_merged": [[1, 10]],
            "expected_boats": 1
        },
        {
            "id": "Custom 3",
            "input": [],
            "expected_merged": [],
            "expected_boats": 0
        },
        {
            "id": "Custom 4",
            "input": [[1, 2], [3, 4], [5, 6]],
            "expected_merged": [[1, 2], [3, 4], [5, 6]],
            "expected_boats": 1
        }
    ]

    print("--- 运行测试用例 ---")
    for case in test_cases:
        print(f"\n测试用例 ID: {case['id']}")
        input_data = case['input']
        print(f"输入: {input_data}")
        
        merged_slots, min_boats = merge_and_count_boats(input_data)
        
        print(f"合并后的时段: {merged_slots}")
        print(f"所需最小船只数: {min_boats}")
        
        # 验证结果
        is_merged_correct = (merged_slots == case["expected_merged"])
        is_boats_correct = (min_boats == case["expected_boats"])

        print(f"合并结果是否正确: {is_merged_correct}")
        print(f"船只数结果是否正确: {is_boats_correct}")
        
        if not is_merged_correct or not is_boats_correct:
            print("--- 测试失败 ---")
        else:
            print("--- 测试通过 ---")

if __name__ == "__main__":
    run_tests()