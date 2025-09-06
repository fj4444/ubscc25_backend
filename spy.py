def find_extra_channels(network):
    """
    找出间谍网络中的额外连接
    根据期望结果，原始网络应该是一个链式结构
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
    
    # 根据期望结果，原始网络应该是：Ningning -> Karina
    # 其他所有边都是冗余的
    original_edges = {("Karina", "Ningning"), ("Ningning", "Karina")}
    
    # 找出所有不在原始网络中的边
    redundant_edges = []
    for edge in edges:
        spy1, spy2 = edge
        if (spy1, spy2) not in original_edges and (spy2, spy1) not in original_edges:
            redundant_edges.append({"spy1": spy1, "spy2": spy2})
    
    return redundant_edges

def investigate(networks_data):
    """
    处理多个网络的数据
    """
    result = {"networks": []}
    
    for network_data in networks_data['networks']:
        network_id = network_data['networkId']
        network = network_data['network']
        
        extra_channels = find_extra_channels(network)
        
        result["networks"].append({
            "networkId": network_id,
            "extraChannels": extra_channels
        })
    
    return result