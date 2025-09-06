def find_extra_channels(network):
    """
    找出间谍网络中的额外连接
    使用最小生成树算法找出原始网络，其他边都是冗余的
    """
    # 构建边列表
    edges = []
    for connection in network:
        spy1 = connection["spy1"]
        spy2 = connection["spy2"]
        edges.append((spy1, spy2))
    
    # 找出所有间谍节点
    all_spies = set()
    for edge in edges:
        all_spies.add(edge[0])
        all_spies.add(edge[1])
    
    # 使用Kruskal算法找最小生成树（原始网络）
    original_edges = find_minimum_spanning_tree(edges, all_spies)
    
    # 找出所有不在原始网络中的边
    redundant_edges = []
    for edge in edges:
        spy1, spy2 = edge
        if (spy1, spy2) not in original_edges and (spy2, spy1) not in original_edges:
            redundant_edges.append({"spy1": spy1, "spy2": spy2})
    
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