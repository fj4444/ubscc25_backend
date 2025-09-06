def find_extra_channels(network):
    """
    找出间谍网络中的额外连接
    使用更智能的算法来识别冗余连接
    """
    # 构建边列表和邻接表
    edges = []
    adjacency = {}
    
    for connection in network:
        spy1 = connection["spy1"]
        spy2 = connection["spy2"]
        edges.append((spy1, spy2))
        
        # 构建邻接表（双向）
        if spy1 not in adjacency:
            adjacency[spy1] = set()
        if spy2 not in adjacency:
            adjacency[spy2] = set()
        adjacency[spy1].add(spy2)
        adjacency[spy2].add(spy1)
    
    # 找出所有间谍节点
    all_spies = set()
    for edge in edges:
        all_spies.add(edge[0])
        all_spies.add(edge[1])
    
    # 使用多种策略找冗余边
    redundant_edges = []
    
    # 策略1: 找最小生成树
    mst_edges = find_minimum_spanning_tree(edges, all_spies)
    
    # 策略2: 找可以通过其他路径到达的边
    redundant_by_path = find_redundant_by_alternative_paths(edges, adjacency, all_spies)
    
    # 合并结果
    all_redundant = set()
    
    # 添加不在MST中的边
    for edge in edges:
        if edge not in mst_edges and (edge[1], edge[0]) not in mst_edges:
            all_redundant.add(edge)
    
    # 添加通过其他路径可达的边
    for edge in redundant_by_path:
        all_redundant.add(edge)
    
    # 调试信息（可选）
    print(f"调试信息:")
    print(f"  总边数: {len(edges)}")
    print(f"  MST边数: {len(mst_edges)}")
    print(f"  通过路径找到的冗余边: {len(redundant_by_path)}")
    print(f"  最终冗余边数: {len(all_redundant)}")
    print(f"  所有边: {edges}")
    print(f"  MST边: {mst_edges}")
    print(f"  冗余边: {all_redundant}")
    
    # 转换为输出格式
    for edge in all_redundant:
        redundant_edges.append({"spy1": edge[0], "spy2": edge[1]})
    
    return redundant_edges

def find_minimum_spanning_tree(edges, nodes):
    """
    使用Kruskal算法找最小生成树
    这里我们假设所有边的权重都是1，所以实际上是在找连通所有节点的最小边集
    """
    # 初始化并查集
    parent = {node: node for node in nodes}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    # 按边的字典序排序（确保结果的一致性）
    sorted_edges = sorted(edges)
    
    mst_edges = set()
    for edge in sorted_edges:
        spy1, spy2 = edge
        if union(spy1, spy2):
            mst_edges.add(edge)
            # 如果已经连通了所有节点，可以提前结束
            if len(mst_edges) == len(nodes) - 1:
                break
    
    return mst_edges

def find_redundant_by_alternative_paths(edges, adjacency, nodes):
    """
    通过检查是否存在替代路径来找出冗余边
    """
    redundant = set()
    
    for edge in edges:
        spy1, spy2 = edge
        
        # 临时移除这条边
        adjacency[spy1].discard(spy2)
        adjacency[spy2].discard(spy1)
        
        # 检查是否仍然连通
        if is_connected_without_edge(spy1, spy2, adjacency, nodes):
            redundant.add(edge)
        
        # 恢复边
        adjacency[spy1].add(spy2)
        adjacency[spy2].add(spy1)
    
    return redundant

def is_connected_without_edge(start, end, adjacency, nodes):
    """
    检查在移除某条边后，两个节点是否仍然连通
    """
    if start == end:
        return True
    
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        
        if node == end:
            return True
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return False

def investigate(networks_data):
    """
    处理多个网络的数据
    """
    result = {"networks": []}
    
    # 检查输入数据的格式
    if isinstance(networks_data, list):
        # 如果直接是网络列表
        networks_list = networks_data
    elif isinstance(networks_data, dict) and 'networks' in networks_data:
        # 如果是包含networks键的字典
        networks_list = networks_data['networks']
    else:
        # 如果格式不正确，返回错误
        return {"error": "Invalid data format. Expected list of networks or dict with 'networks' key."}
    
    for network_data in networks_list:
        network_id = network_data['networkId']
        network = network_data['network']
        
        extra_channels = find_extra_channels(network)
        
        result["networks"].append({
            "networkId": network_id,
            "extraChannels": extra_channels
        })
    
    return result