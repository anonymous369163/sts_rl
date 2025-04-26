"""
debug 文件
"""



import pandas as pd
import numpy as np
import sys
import time
import psutil
import os

def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_test_data():
    """创建测试数据：1000(时间) * 1000(空间) * 5(速度)个节点"""
    print("开始创建测试数据...")
    
    # 创建节点数据
    nodes = []
    for t in range(1000):
        for d in range(1000):
            for v in range(5):
                # 每个节点包含时间、空间、速度和一些随机值
                nodes.append((t, d, v, np.random.random()))
    
    print(f"创建了 {len(nodes)} 个节点")
    return nodes

def test_dict_storage(nodes):
    """测试字典存储方式的内存占用"""
    print("\n测试字典存储...")
    
    # 记录开始内存
    start_memory = get_memory_usage()
    print(f"开始内存占用: {start_memory:.2f} MB")
    
    start_time = time.time()
    
    # 创建字典存储
    node_dict = {}
    for t, d, v, value in nodes:
        node_dict[(t, d, v)] = value
    
    # 记录结束内存和时间
    end_memory = get_memory_usage()
    end_time = time.time()
    
    print(f"字典大小: {len(node_dict)}")
    print(f"结束内存占用: {end_memory:.2f} MB")
    print(f"内存增加: {end_memory - start_memory:.2f} MB")
    print(f"创建字典耗时: {end_time - start_time:.2f} 秒")
    
    # 测试访问速度
    start_time = time.time()
    for _ in range(10000):
        t = np.random.randint(0, 1000)
        d = np.random.randint(0, 1000)
        v = np.random.randint(0, 5)
        _ = node_dict.get((t, d, v), None)
    
    print(f"字典随机访问10000次耗时: {time.time() - start_time:.4f} 秒")
    
    # 测试范围查询速度
    start_time = time.time()
    for _ in range(100):
        t_min, t_max = sorted(np.random.randint(0, 1000, 2))
        d_min, d_max = sorted(np.random.randint(0, 1000, 2))
        v_min, v_max = sorted(np.random.randint(0, 5, 2))
        
        # 字典需要遍历所有键并筛选
        result = []
        for key, value in node_dict.items():
            t, d, v = key
            if (t_min <= t <= t_max and 
                d_min <= d <= d_max and 
                v_min <= v <= v_max):
                result.append(value)
    
    print(f"字典范围查询100次耗时: {time.time() - start_time:.4f} 秒")
    print(f"最后一次查询结果包含 {len(result)} 个项目")
    
    return end_memory - start_memory

def test_dataframe_storage(nodes):
    """测试DataFrame存储方式的内存占用"""
    print("\n测试DataFrame存储...")
    
    # 记录开始内存
    start_memory = get_memory_usage()
    print(f"开始内存占用: {start_memory:.2f} MB")
    
    start_time = time.time()
    
    # 创建DataFrame
    data = {
        'time': [node[0] for node in nodes],
        'distance': [node[1] for node in nodes],
        'speed': [node[2] for node in nodes],
        'value': [node[3] for node in nodes]
    }
    df = pd.DataFrame(data)
    # 设置MultiIndex以便于进行范围查询
    # 创建索引版本但保留原始列
    df_indexed = df.set_index(['time', 'distance', 'speed'], drop=False)
    df = df_indexed  # 将索引版本赋值回df，同时保留原始列便于范围查询
    
    # 记录结束内存和时间
    end_memory = get_memory_usage()
    end_time = time.time()
    
    print(f"DataFrame大小: {len(df)}")
    print(f"结束内存占用: {end_memory:.2f} MB")
    print(f"内存增加: {end_memory - start_memory:.2f} MB")
    print(f"创建DataFrame耗时: {end_time - start_time:.2f} 秒")
    
    # 测试访问速度
    start_time = time.time()
    for _ in range(10000):
        t = np.random.randint(0, 1000)
        d = np.random.randint(0, 1000)
        v = np.random.randint(0, 5)
        # 使用索引进行查询，更高效 
        _ = df.loc[(t, d, v), 'value'] 
    
    print(f"DataFrame随机访问10000次耗时: {time.time() - start_time:.4f} 秒")
    
    # 测试范围查询速度
    start_time = time.time()
    for _ in range(100):
        t_min, t_max = sorted(np.random.randint(0, 1000, 2))
        d_min, d_max = sorted(np.random.randint(0, 1000, 2))
        v_min, v_max = sorted(np.random.randint(0, 5, 2))
        
        # 使用DataFrame的范围查询功能
        result = df.query(f"time >= {t_min} and time <= {t_max} and "
                          f"distance >= {d_min} and distance <= {d_max} and "
                          f"speed >= {v_min} and speed <= {v_max}")
    
    print(f"DataFrame范围查询100次耗时: {time.time() - start_time:.4f} 秒")
    print(f"最后一次查询结果包含 {len(result)} 个项目")
    
    return end_memory - start_memory

if __name__ == "__main__":
    print("开始测试字典和DataFrame存储1000*1000*5个节点的内存占用和查询效率对比")
    print("=" * 80)
    
    # 创建测试数据
    nodes = create_test_data()
    
    # 测试字典存储
    dict_memory = test_dict_storage(nodes)
    
    # 测试DataFrame存储
    df_memory = test_dataframe_storage(nodes)

    # 测试集合存储
    print("\n测试集合(Set)存储...")
    print("=" * 80)
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # 创建集合存储
    nodes_set = set()
    for node in nodes:
        nodes_set.add(node)
    
    # 记录结束内存和时间
    end_memory = get_memory_usage() 
    end_time = time.time()
    
    set_memory = end_memory - start_memory
    
    print(f"集合大小: {len(nodes_set)}")
    print(f"结束内存占用: {end_memory:.2f} MB")
    print(f"内存增加: {set_memory:.2f} MB")
    print(f"创建集合耗时: {end_time - start_time:.2f} 秒")
    
    # 测试列表存储
    print("\n测试列表(List)存储...")
    print("=" * 80)
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # 创建列表存储
    nodes_list = []
    for node in nodes:
        nodes_list.append(node)
    
    # 记录结束内存和时间
    end_memory = get_memory_usage()
    end_time = time.time()
    
    list_memory = end_memory - start_memory
    
    print(f"列表大小: {len(nodes_list)}")
    print(f"结束内存占用: {end_memory:.2f} MB")
    print(f"内存增加: {list_memory:.2f} MB")
    print(f"创建列表耗时: {end_time - start_time:.2f} 秒")
    
    # 比较结果
    print("\n内存占用比较结果:")
    print("=" * 80)
    print(f"字典存储内存占用: {dict_memory:.2f} MB")
    print(f"DataFrame存储内存占用: {df_memory:.2f} MB")
    print(f"集合存储内存占用: {set_memory:.2f} MB")
    print(f"列表存储内存占用: {list_memory:.2f} MB")
    print(f"差异: {df_memory - dict_memory:.2f} MB")
    print(f"DataFrame比字典多用了 {(df_memory/dict_memory - 1)*100:.2f}% 的内存")
    print("=" * 80)


