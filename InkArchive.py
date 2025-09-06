import math
import numpy as np
import json
from collections import deque

def find_max_product_cycle_optimized(graph):
    """
    优化的方法：使用BFS和剪枝
    """
    n = len(graph)
    max_product = -1
    best_cycle = []
    
    # 对每个节点作为起点
    for start in range(n):
        # BFS队列：(当前节点, 路径, 乘积, 访问过的节点掩码)
        queue = deque()
        queue.append((start, [start], 1.0, 1 << start))
        
        while queue:
            current, path, product, visited = queue.popleft()
            
            # 如果回到起点且路径长度大于1
            if len(path) > 1 and current == start:
                if product > max_product:
                    max_product = product
                    best_cycle = path.copy()
                continue
            
            # 探索下一节点
            for next_node in range(n):
                if next_node != current:  # 避免自环
                    # 如果下一个节点是起点或者还没访问过
                    if next_node == start or not (visited & (1 << next_node)):
                        new_product = product * graph[current][next_node]
                        new_visited = visited | (1 << next_node)
                        new_path = path + [next_node]
                        
                        # 剪枝：如果当前乘积已经不可能超过最大值，则跳过
                        if new_product > max_product or max_product == -1:
                            queue.append((next_node, new_path, new_product, new_visited))
    
    return best_cycle, max_product

def process(d):
    goods = d["goods"]
    n = len(goods)
    G = np.zeros((n, n)) + 1e-50
    for i in range(n):
        G[i, i] = 1.0
    r = d["ratios"]
    for rs in r:
        p1 = int(rs[0] + 0.5)
        p2 = int(rs[1] + 0.5)
        G[p1, p2] = rs[2]
    cycle, max_product = find_max_product_cycle_optimized(G)
    result = {"path": [goods[i] for i in cycle]}
    result["gain"] = 100 * (max_product - 1)
    return result

def final(json_data):
    ans = list()
    for d in json_data:
        ans.append(process(d))
    json_output = json.dumps(ans, indent=2)
    return json_output


if __name__ == "__main__":
    json_data =     [
  {
    "ratios": [
      [0.0, 1.0, 0.9],
      [1.0, 2.0, 120.0],
      [2.0, 0.0, 0.008],
      [0.0, 3.0, 0.00005],
      [3.0, 1.0, 18000.0],
      [1.0, 0.0, 1.11],
      [2.0, 3.0, 0.0000004],
      [3.0, 2.0, 2600000.0],
      [1.0, 3.0, 0.000055],
      [3.0, 0.0, 20000.0],
      [2.0, 1.0, 0.0075]
    ],
    "goods": [
      "Blue Moss",
      "Amberback Shells",
      "Kelp Silk",
      "Ventspice"
    ]
  },
  {
    "ratios": [
      [0.0, 1.0, 0.9],
      [1.0, 2.0, 1.1],
      [2.0, 0.0, 1.2],
      [1.0, 0.0, 1.1],
      [2.0, 1.0, 0.95],
      [0.0, 8.0, 0.8740536415671486],
      [15.0, 13.0, 0.9050437005117633],
      [19.0, 14.0, 0.883321839947665],
      [13.0, 2.0, 0.9484841540199809],
      [4.0, 15.0, 0.9441249179482114],
      [3.0, 8.0, 0.8628897150873777],
      [12.0, 0.0, 0.8523238122483889],
      [12.0, 3.0, 0.946448686067685],
      [17.0, 7.0, 0.9125146363465559],
      [5.0, 13.0, 0.9276312291274932],
      [0.0, 15.0, 0.8987232847030142],
      [18.0, 11.0, 0.9233152070194993],
      [12.0, 2.0, 0.9338890350047018],
      [16.0, 6.0, 0.9399335011611493],
      [8.0, 17.0, 0.8583062398224914],
      [7.0, 8.0, 0.9222357119188849],
      [11.0, 18.0, 0.8962957818422422],
      [18.0, 12.0, 0.8571498314879894],
      [6.0, 0.0, 0.8838769653535753],
      [15.0, 0.0, 0.9471546988851712],
      [17.0, 9.0, 0.9112581104709868],
      [9.0, 3.0, 0.8717570412209685],
      [3.0, 0.0, 0.8509673497300974],
      [16.0, 3.0, 0.927131296617068],
      [5.0, 11.0, 0.8711235374929895],
      [19.0, 13.0, 0.9445333238959629],
      [11.0, 17.0, 0.8894203552777331],
      [10.0, 5.0, 0.9286042450814552],
      [6.0, 1.0, 0.8670291530247703],
      [11.0, 14.0, 0.9224295033578869],
      [6.0, 4.0, 0.9304395417224636],
      [12.0, 10.0, 0.8962087990285994],
      [9.0, 5.0, 0.9001834850205734],
      [5.0, 2.0, 0.9469269909940415],
      [17.0, 11.0, 0.8547265869196129],
      [17.0, 18.0, 0.8529107512721635],
      [15.0, 1.0, 0.9471950120953745],
      [13.0, 17.0, 0.926744210301549],
      [16.0, 18.0, 0.8755525310896444],
      [15.0, 5.0, 0.9348280500272342],
      [13.0, 10.0, 0.8853852969708712],
      [18.0, 1.0, 0.85036769023557],
      [19.0, 11.0, 0.9031608556248797],
      [17.0, 1.0, 0.8777210026068711],
      [4.0, 3.0, 0.9109881513512763]
    ],
    "goods": [
      "Drift Kelp",
      "Sponge Flesh",
      "Saltbeads",
      "Stoneworm Paste",
      "Mudshrimp",
      "Algae Cakes",
      "Coral Pollen",
      "Rockmilk",
      "Kelp Tubers",
      "Shell Grain",
      "Inkbinding Resin",
      "Reef Clay",
      "Dry Coral Fiber",
      "Scale Sand",
      "Glowmoss Threads",
      "Filter Sponges",
      "Bubble Nets",
      "Siltstone Tools",
      "Shell Chits",
      "Crush Coral Blocks"
    ]
  }
]
    print(final(json_data))