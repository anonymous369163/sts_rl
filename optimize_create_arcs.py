"""
创建多种优化版本的create_valid_arcs函数
比较它们的性能以选择最适合的实现
"""

import numpy as np
import math
import heapq
import itertools
import time
import os
import concurrent.futures
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix
from functools import lru_cache
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba 未安装，相关优化无法使用")


def original_create_valid_arcs(valid_nodes, train_schedule, a_max):
    """
    原始的create_valid_arcs函数实现（假设的实现，根据需要替换为实际代码）
    
    参数:
        valid_nodes: 有效节点集合
        train_schedule: 列车时刻表
        a_max: 最大加速度约束
        
    返回:
        graph: 图的邻接表表示
        len_valid_arcs: 添加的有效弧数量
    """
    graph = {}
    valid_arcs_count = 0
    
    for node1 in valid_nodes:
        i1, t1, v1 = node1
        
        # 初始化当前节点的邻接表
        if node1 not in graph:
            graph[node1] = []
        
        for node2 in valid_nodes:
            if node1 == node2:
                continue
                
            i2, t2, v2 = node2
            
            # 检查时间是否递增
            if t2 <= t1:
                continue
                
            # 检查加速度约束
            delta_t = t2 - t1
            delta_v = v2 - v1
            delta_s = i2 - i1
            
            # 速度变化必须满足加速度约束
            if abs(delta_v) > a_max * delta_t:
                continue
                
            # 位移必须与平均速度相符
            avg_speed = (v1 + v2) / 2
            expected_s = avg_speed * delta_t
            
            if abs(delta_s - expected_s) > 0.5:  # 允许一定误差
                continue
                
            # 满足条件，添加弧
            cost = delta_t  # 使用时间作为成本
            graph[node1].append((node2, cost))
            valid_arcs_count += 1
    
    return graph, valid_arcs_count


def numba_create_valid_arcs(valid_nodes, train_schedule, a_max):
    """
    使用Numba加速的create_valid_arcs实现
    
    参数同上
    """
    if not NUMBA_AVAILABLE:
        print("无法使用Numba优化，回退到原始版本")
        return original_create_valid_arcs(valid_nodes, train_schedule, a_max)
    
    # 转换为numpy数组以便Numba处理
    nodes_array = np.array(list(valid_nodes))
    
    @jit(nopython=True)
    def check_valid_arc(i1, t1, v1, i2, t2, v2, a_max):
        """检查节点间的弧是否有效"""
        # 检查时间是否递增
        if t2 <= t1:
            return False
            
        # 检查加速度约束
        delta_t = t2 - t1
        delta_v = v2 - v1
        delta_s = i2 - i1
        
        # 速度变化必须满足加速度约束
        if abs(delta_v) > a_max * delta_t:
            return False
            
        # 位移必须与平均速度相符
        avg_speed = (v1 + v2) / 2
        expected_s = avg_speed * delta_t
        
        if abs(delta_s - expected_s) > 0.5:  # 允许一定误差
            return False
            
        return True
    
    @jit(nopython=True, parallel=True)
    def build_valid_arcs(nodes_array, a_max):
        """构建有效弧列表"""
        n = len(nodes_array)
        valid_arcs = []
        
        for i in prange(n):
            i1, t1, v1 = nodes_array[i]
            
            for j in range(n):
                if i == j:
                    continue
                    
                i2, t2, v2 = nodes_array[j]
                
                if check_valid_arc(i1, t1, v1, i2, t2, v2, a_max):
                    # 计算成本
                    cost = float(t2 - t1)
                    valid_arcs.append((i, j, cost))
        
        return valid_arcs
    
    # 使用Numba函数获取有效弧
    valid_arcs = build_valid_arcs(nodes_array, a_max)
    
    # 构建邻接表形式的图
    graph = {}
    for node in valid_nodes:
        graph[node] = []
    
    valid_arcs_count = len(valid_arcs)
    
    # 填充图结构
    for i, j, cost in valid_arcs:
        source = tuple(nodes_array[i])
        dest = tuple(nodes_array[j])
        graph[source].append((dest, cost))
    
    return graph, valid_arcs_count


def batch_create_valid_arcs(valid_nodes, train_schedule, a_max):
    """
    使用批量弧生成的create_valid_arcs实现
    
    参数同上
    """
    graph = {node: [] for node in valid_nodes}
    valid_nodes_list = list(valid_nodes)
    valid_arcs_count = 0
    
    # 一次性生成所有可能的节点对
    node_pairs = list(itertools.product(valid_nodes_list, valid_nodes_list))
    filtered_pairs = []
    
    # 批量预筛选
    for node1, node2 in node_pairs:
        i1, t1, v1 = node1
        i2, t2, v2 = node2
        
        # 快速筛选：时间必须递增
        if t2 <= t1 or node1 == node2:
            continue
            
        filtered_pairs.append((node1, node2))
    
    # 对预筛选后的对进行详细检查
    for node1, node2 in filtered_pairs:
        i1, t1, v1 = node1
        i2, t2, v2 = node2
        
        # 检查加速度约束
        delta_t = t2 - t1
        delta_v = v2 - v1
        delta_s = i2 - i1
        
        # 速度变化必须满足加速度约束
        if abs(delta_v) > a_max * delta_t:
            continue
            
        # 位移必须与平均速度相符
        avg_speed = (v1 + v2) / 2
        expected_s = avg_speed * delta_t
        
        if abs(delta_s - expected_s) > 0.5:  # 允许一定误差
            continue
            
        # 满足条件，添加弧
        cost = delta_t  # 使用时间作为成本
        graph[node1].append((node2, cost))
        valid_arcs_count += 1
    
    return graph, valid_arcs_count


def spatial_index_create_valid_arcs(valid_nodes, train_schedule, a_max):
    """
    使用空间索引优化的create_valid_arcs实现
    
    参数同上
    """
    graph = {node: [] for node in valid_nodes}
    valid_arcs_count = 0
    
    # 提取节点的空间和时间坐标
    valid_nodes_list = list(valid_nodes)
    node_indices = {node: idx for idx, node in enumerate(valid_nodes_list)}
    
    # 创建时间分组，将节点按时间排序
    time_groups = {}
    for node in valid_nodes:
        _, t, _ = node
        if t not in time_groups:
            time_groups[t] = []
        time_groups[t].append(node)
    
    # 排序时间点
    sorted_times = sorted(time_groups.keys())
    
    # 对于每个节点，只检查未来时间点的节点
    for t_idx, t1 in enumerate(sorted_times[:-1]):  # 跳过最后一个时间点
        for node1 in time_groups[t1]:
            i1, _, v1 = node1
            
            # 只检查未来时间点
            for t2 in sorted_times[t_idx+1:]:
                # 计算时间差
                delta_t = t2 - t1
                
                # 根据最大加速度和时间差，计算可能的位置范围
                max_delta_v = a_max * delta_t
                avg_speed_min = max(0, v1 - max_delta_v/2)
                avg_speed_max = v1 + max_delta_v/2
                
                min_dist = avg_speed_min * delta_t - 0.5
                max_dist = avg_speed_max * delta_t + 0.5
                
                # 寻找可能的目标节点
                for node2 in time_groups[t2]:
                    i2, _, v2 = node2
                    delta_s = i2 - i1
                    
                    # 检查位置是否在范围内
                    if not (min_dist <= delta_s <= max_dist):
                        continue
                    
                    # 检查加速度约束
                    delta_v = v2 - v1
                    if abs(delta_v) > max_delta_v:
                        continue
                    
                    # 检查平均速度与位移的关系
                    avg_speed = (v1 + v2) / 2
                    expected_s = avg_speed * delta_t
                    
                    if abs(delta_s - expected_s) > 0.5:
                        continue
                    
                    # 添加有效弧
                    cost = delta_t
                    graph[node1].append((node2, cost))
                    valid_arcs_count += 1
    
    return graph, valid_arcs_count


def parallel_create_valid_arcs(valid_nodes, train_schedule, a_max, num_workers=None):
    """
    使用并行子任务处理的create_valid_arcs实现
    
    参数:
        valid_nodes: 有效节点集合
        train_schedule: 列车时刻表
        a_max: 最大加速度约束
        num_workers: 工作线程数，默认为None（使用CPU核心数）
        
    返回:
        graph: 图的邻接表表示
        len_valid_arcs: 添加的有效弧数量
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    
    # 将节点分成多个批次
    valid_nodes_list = list(valid_nodes)
    chunk_size = max(1, len(valid_nodes_list) // num_workers)
    node_chunks = [valid_nodes_list[i:i+chunk_size] for i in range(0, len(valid_nodes_list), chunk_size)]
    
    def process_chunk(source_nodes, all_nodes, a_max):
        """处理一个节点批次"""
        chunk_graph = {}
        arc_count = 0
        
        for node1 in source_nodes:
            i1, t1, v1 = node1
            chunk_graph[node1] = []
            
            for node2 in all_nodes:
                if node1 == node2:
                    continue
                    
                i2, t2, v2 = node2
                
                # 检查时间是否递增
                if t2 <= t1:
                    continue
                    
                # 检查加速度约束
                delta_t = t2 - t1
                delta_v = v2 - v1
                delta_s = i2 - i1
                
                # 速度变化必须满足加速度约束
                if abs(delta_v) > a_max * delta_t:
                    continue
                    
                # 位移必须与平均速度相符
                avg_speed = (v1 + v2) / 2
                expected_s = avg_speed * delta_t
                
                if abs(delta_s - expected_s) > 0.5:  # 允许一定误差
                    continue
                    
                # 满足条件，添加弧
                cost = delta_t  # 使用时间作为成本
                chunk_graph[node1].append((node2, cost))
                arc_count += 1
        
        return chunk_graph, arc_count
    
    # 使用线程池并行处理所有批次
    graph = {node: [] for node in valid_nodes}
    valid_arcs_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk in node_chunks:
            future = executor.submit(process_chunk, chunk, valid_nodes_list, a_max)
            futures.append(future)
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            chunk_graph, chunk_arc_count = future.result()
            # 合并到最终图
            for node, arcs in chunk_graph.items():
                graph[node].extend(arcs)
            valid_arcs_count += chunk_arc_count
    
    return graph, valid_arcs_count


def sparse_matrix_create_valid_arcs(valid_nodes, train_schedule, a_max):
    """
    使用稀疏矩阵表示的create_valid_arcs实现
    
    参数同上
    """
    valid_nodes_list = list(valid_nodes)
    n = len(valid_nodes_list)
    node_indices = {node: idx for idx, node in enumerate(valid_nodes_list)}
    
    # 使用稀疏矩阵表示图结构
    # 矩阵中(i,j)的值表示从节点i到节点j的弧的成本，0表示没有连接
    adj_matrix = lil_matrix((n, n), dtype=float)
    valid_arcs_count = 0
    
    for i, node1 in enumerate(valid_nodes_list):
        i1, t1, v1 = node1
        
        for j, node2 in enumerate(valid_nodes_list):
            if i == j:
                continue
                
            i2, t2, v2 = node2
            
            # 检查时间是否递增
            if t2 <= t1:
                continue
                
            # 检查加速度约束
            delta_t = t2 - t1
            delta_v = v2 - v1
            delta_s = i2 - i1
            
            # 速度变化必须满足加速度约束
            if abs(delta_v) > a_max * delta_t:
                continue
                
            # 位移必须与平均速度相符
            avg_speed = (v1 + v2) / 2
            expected_s = avg_speed * delta_t
            
            if abs(delta_s - expected_s) > 0.5:  # 允许一定误差
                continue
                
            # 满足条件，添加弧
            cost = delta_t  # 使用时间作为成本
            adj_matrix[i, j] = cost
            valid_arcs_count += 1
    
    # 将稀疏矩阵转换为邻接表形式
    graph = {node: [] for node in valid_nodes}
    
    # 转为更高效的CSR格式进行迭代
    adj_matrix = adj_matrix.tocsr()
    
    for i in range(n):
        node1 = valid_nodes_list[i]
        # 获取非零元素的列索引和值
        row_start, row_end = adj_matrix.indptr[i], adj_matrix.indptr[i+1]
        col_indices = adj_matrix.indices[row_start:row_end]
        data = adj_matrix.data[row_start:row_end]
        
        for col_idx, cost in zip(col_indices, data):
            node2 = valid_nodes_list[col_idx]
            graph[node1].append((node2, cost))
    
    return graph, valid_arcs_count


def cached_create_valid_arcs(valid_nodes, train_schedule, a_max):
    """
    使用函数缓存的create_valid_arcs实现
    
    参数同上
    """
    graph = {node: [] for node in valid_nodes}
    valid_arcs_count = 0
    
    @lru_cache(maxsize=10000)  # 缓存结果提高重复计算效率
    def is_valid_arc(i1, t1, v1, i2, t2, v2, a_max):
        """检查弧是否有效"""
        # 检查时间是否递增
        if t2 <= t1:
            return False
            
        # 检查加速度约束
        delta_t = t2 - t1
        delta_v = v2 - v1
        delta_s = i2 - i1
        
        # 速度变化必须满足加速度约束
        if abs(delta_v) > a_max * delta_t:
            return False
            
        # 位移必须与平均速度相符
        avg_speed = (v1 + v2) / 2
        expected_s = avg_speed * delta_t
        
        if abs(delta_s - expected_s) > 0.5:  # 允许一定误差
            return False
            
        return True
    
    for node1 in valid_nodes:
        i1, t1, v1 = node1
        
        for node2 in valid_nodes:
            if node1 == node2:
                continue
                
            i2, t2, v2 = node2
            
            # 使用缓存函数检查弧是否有效
            if is_valid_arc(i1, t1, v1, i2, t2, v2, a_max):
                # 满足条件，添加弧
                cost = t2 - t1  # 使用时间作为成本
                graph[node1].append((node2, cost))
                valid_arcs_count += 1
    
    return graph, valid_arcs_count


def hybrid_create_valid_arcs(valid_nodes, train_schedule, a_max, num_workers=None):
    """
    结合多种优化方法的create_valid_arcs实现
    
    参数同上，增加num_workers参数控制并行度
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    
    # 转换为数组并获取索引映射
    valid_nodes_list = list(valid_nodes)
    node_indices = {node: idx for idx, node in enumerate(valid_nodes_list)}
    
    # 按时间分组
    time_groups = {}
    for node in valid_nodes:
        _, t, _ = node
        if t not in time_groups:
            time_groups[t] = []
        time_groups[t].append(node)
    
    # 排序时间点
    sorted_times = sorted(time_groups.keys())
    
    # 定义一个处理时间批次的函数
    def process_time_batch(time_batch_idx, a_max):
        """处理一个时间批次的节点"""
        local_graph = {}
        arc_count = 0
        
        # 获取当前批次的时间范围
        start_idx = time_batch_idx * time_batch_size
        end_idx = min(start_idx + time_batch_size, len(sorted_times))
        
        if start_idx >= len(sorted_times):
            return {}, 0
        
        current_times = sorted_times[start_idx:end_idx]
        
        # 对该批次中每个时间点的节点进行处理
        for t_idx, t1 in enumerate(current_times):
            for node1 in time_groups[t1]:
                i1, _, v1 = node1
                
                if node1 not in local_graph:
                    local_graph[node1] = []
                
                # 只检查未来时间点
                future_times = sorted_times[sorted_times.index(t1)+1:]
                
                for t2 in future_times:
                    # 计算时间差
                    delta_t = t2 - t1
                    
                    # 根据最大加速度和时间差，计算可能的位置范围
                    max_delta_v = a_max * delta_t
                    avg_speed_min = max(0, v1 - max_delta_v/2)
                    avg_speed_max = v1 + max_delta_v/2
                    
                    min_dist = avg_speed_min * delta_t - 0.5
                    max_dist = avg_speed_max * delta_t + 0.5
                    
                    # 寻找可能的目标节点
                    for node2 in time_groups[t2]:
                        i2, _, v2 = node2
                        delta_s = i2 - i1
                        
                        # 检查位置是否在范围内
                        if not (min_dist <= delta_s <= max_dist):
                            continue
                        
                        # 检查加速度约束
                        delta_v = v2 - v1
                        if abs(delta_v) > max_delta_v:
                            continue
                        
                        # 检查平均速度与位移的关系
                        avg_speed = (v1 + v2) / 2
                        expected_s = avg_speed * delta_t
                        
                        if abs(delta_s - expected_s) > 0.5:
                            continue
                        
                        # 添加有效弧
                        cost = delta_t
                        local_graph[node1].append((node2, cost))
                        arc_count += 1
        
        return local_graph, arc_count
    
    # 计算时间批次大小
    time_batch_size = max(1, len(sorted_times) // num_workers)
    
    # 创建时间批次
    time_batch_indices = list(range(0, (len(sorted_times) + time_batch_size - 1) // time_batch_size))
    
    # 使用线程池并行处理所有批次
    graph = {node: [] for node in valid_nodes}
    valid_arcs_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch_idx in time_batch_indices:
            future = executor.submit(process_time_batch, batch_idx, a_max)
            futures.append(future)
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            batch_graph, batch_arc_count = future.result()
            # 合并到最终图
            for node, arcs in batch_graph.items():
                if node in graph:  # 确保节点在图中
                    graph[node].extend(arcs)
            valid_arcs_count += batch_arc_count
    
    return graph, valid_arcs_count


def benchmark_functions(valid_nodes, train_schedule, a_max, num_workers=None):
    """
    对所有优化函数进行基准测试
    
    参数:
        valid_nodes: 有效节点集合
        train_schedule: 列车时刻表
        a_max: 最大加速度约束
        num_workers: 工作线程数（用于并行方法）
        
    返回:
        results: 各方法的性能结果字典
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)  # 保留一个核心给系统使用
    
    functions = [
        ("原始实现", original_create_valid_arcs),
        ("Numba加速", numba_create_valid_arcs),
        ("批量处理", batch_create_valid_arcs),
        ("空间索引", spatial_index_create_valid_arcs),
        ("稀疏矩阵", sparse_matrix_create_valid_arcs),
        ("LRU缓存", cached_create_valid_arcs)
    ]
    
    # 添加并行处理的函数
    parallel_funcs = [
        ("并行子任务", lambda vn, ts, am: parallel_create_valid_arcs(vn, ts, am, num_workers)),
        ("混合优化", lambda vn, ts, am: hybrid_create_valid_arcs(vn, ts, am, num_workers))
    ]
    
    functions.extend(parallel_funcs)
    
    results = {}
    
    print(f"开始基准测试...使用 {len(valid_nodes)} 个节点，{num_workers} 个工作线程")
    
    for name, func in functions:
        print(f"测试 {name} 方法...")
        
        try:
            start_time = time.time()
            graph, arcs_count = func(valid_nodes, train_schedule, a_max)
            end_time = time.time()
            
            duration = end_time - start_time
            
            results[name] = {
                "时间": duration,
                "弧数": arcs_count,
                "成功": True
            }
            
            print(f"  - 耗时: {duration:.4f} 秒")
            print(f"  - 创建的弧: {arcs_count}")
            
        except Exception as e:
            print(f"  - 错误: {str(e)}")
            results[name] = {
                "时间": float('inf'),
                "弧数": 0,
                "成功": False,
                "错误": str(e)
            }
    
    # 对结果排序（按时间）
    sorted_results = sorted(
        [(name, info) for name, info in results.items() if info["成功"]],
        key=lambda x: x[1]["时间"]
    )
    
    if sorted_results:
        fastest_name, fastest_info = sorted_results[0]
        slowest_name, slowest_info = sorted_results[-1]
        
        speedup = slowest_info["时间"] / fastest_info["时间"]
        
        print("\n性能对比结果:")
        print(f"最快的方法: {fastest_name}, 耗时 {fastest_info['时间']:.4f} 秒")
        print(f"最慢的方法: {slowest_name}, 耗时 {slowest_info['时间']:.4f} 秒")
        print(f"加速比: {speedup:.2f}x")
        
        # 完整结果表格
        print("\n所有方法的耗时（秒）:")
        for name, info in sorted_results:
            print(f"{name:15s}: {info['时间']:.4f} ({info['弧数']} 弧)")
    
    return results


if __name__ == "__main__":
    # 测试不同方法的性能
    print("请确保已导入实际的create_valid_arcs函数所需的依赖")
    print("使用示例:")
    print("from optimize_create_arcs import benchmark_functions")
    print("results = benchmark_functions(valid_nodes, train_schedule, a_max)")
    print("也可以单独调用每个优化函数，比如:")
    print("from optimize_create_arcs import numba_create_valid_arcs")
    print("graph, arc_count = numba_create_valid_arcs(valid_nodes, train_schedule, a_max)") 