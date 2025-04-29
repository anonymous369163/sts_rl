"""
使用拉格朗日乘子法求解多列车时刻表的STS网格模型
基于原始STS网格模型，添加列车间耦合约束，并使用拉格朗日松弛法求解
单线程版本
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import math
import heapq
import copy
import time
import concurrent.futures
from utils import plot_sts_grid_nodes, draw_plan_on_sts_grid, finalize_plot, plot_optimal_path, export_train_schedule
from main import (convert_time_to_units, Arrow3D, convert_time_to_minutes, check_time_nodes_and_space_segments,
                  trans_df_grid, filter_valid_nodes, filter_nodes_near_plan, create_valid_arcs)
from optimize_create_arcs import batch_create_valid_arcs



# 验证更新后的乘子是否能够影响所选弧
def verify_multiplier_impact(new_multipliers, all_selected_arcs, headway):
    """
    验证更新后的乘子是否能够影响所选弧
    
    参数:
        new_multipliers: 更新后的拉格朗日乘子字典
        all_selected_arcs: 所有列车选择的弧
        headway: 最小间隔时间
        
    返回:
        has_impact: 布尔值，表示乘子是否能够影响所选弧
        impact_info: 字典，包含影响的详细信息
    """
    print(f"验证更新后的乘子是否能够影响所选弧...")
    
    # 检查乘子与所选弧的交集
    multiplier_keys_set = set(new_multipliers.keys())
    arc_keys_set = set()
    
    # 收集所有列车选择的弧的时空位置键
    for train_idx, selected_arcs in enumerate(all_selected_arcs):
        print(f"列车 {train_idx} 选择了 {len(selected_arcs)} 条弧")
        for arc in selected_arcs:
            i, j, t, s, v_speed, u_speed = arc
            i_pos, j_pos = min(i, j), max(i, j)
            
            # 添加起点和终点时间的键
            arc_keys_set.add((i_pos, j_pos, t))
            arc_keys_set.add((i_pos, j_pos, s))
            
            # 添加headway范围内的时间点
            for h in range(1, headway + 1):
                for time_offset in [-h, h]:
                    arc_keys_set.add((i_pos, j_pos, t + time_offset))
                    arc_keys_set.add((i_pos, j_pos, s + time_offset))
    
    # 检查乘子与所选弧的交集
    intersection = multiplier_keys_set.intersection(arc_keys_set)
    print(f"乘子与所选弧的交集大小: {len(intersection)}")
    
    impact_info = {
        "intersection_size": len(intersection),
        "has_intersection": len(intersection) > 0,
        "multiplier_keys_sample": list(multiplier_keys_set)[:5] if multiplier_keys_set else [],
        "arc_keys_sample": list(arc_keys_set)[:5] if arc_keys_set else [],
        "intersection_sample": list(intersection)[:5] if intersection else []
    }
    
    # 如果没有交集，输出警告和调试信息
    if len(intersection) == 0:
        print("警告: 乘子与所选弧没有交集，这可能导致乘子无法影响路径选择")
        print(f"乘子样本: {impact_info['multiplier_keys_sample'] if impact_info['multiplier_keys_sample'] else '空'}")
        print(f"弧键样本: {impact_info['arc_keys_sample'] if impact_info['arc_keys_sample'] else '空'}")
        
        # 检查键的格式是否一致
        if multiplier_keys_set and arc_keys_set:
            multiplier_key_example = list(multiplier_keys_set)[0]
            arc_key_example = list(arc_keys_set)[0]
            print(f"乘子键格式示例: {multiplier_key_example}, 类型: {[type(x) for x in multiplier_key_example]}")
            print(f"弧键格式示例: {arc_key_example}, 类型: {[type(x) for x in arc_key_example]}")
            
            impact_info["multiplier_key_example"] = multiplier_key_example
            impact_info["arc_key_example"] = arc_key_example
        
        return False, impact_info
    else:
        print(f"乘子与所选弧有 {len(intersection)} 个交集点，乘子可以影响路径选择")
        # 输出一些交集样本
        print(f"交集样本: {impact_info['intersection_sample']}")
        
        # 检查这些交集点的乘子值
        intersection_values = {key: new_multipliers[key] for key in intersection if key in new_multipliers}
        non_zero_values = {k: v for k, v in intersection_values.items() if v > 0}
        print(f"交集中非零乘子数量: {len(non_zero_values)}")
        
        impact_info["non_zero_count"] = len(non_zero_values)
        impact_info["non_zero_sample"] = list(non_zero_values.items())[:5] if non_zero_values else []
        
        if non_zero_values:
            print(f"非零乘子样本: {impact_info['non_zero_sample']}")
        
        return len(non_zero_values) > 0, impact_info

def find_optimal_path_with_lagrangian(valid_nodes, graph, start_node, end_node, 
                                     multipliers, train_idx, headway, ax=None, drawn_arcs=set()):
    """
    使用带有拉格朗日乘子的Dijkstra算法寻找最优路径
    
    参数:
        valid_nodes: 有效节点集合
        graph: 图的邻接表表示
        start_node: 起点节点 (位置, 时间, 速度)
        end_node: 终点节点 (位置, 时间, 速度)
        multipliers: 拉格朗日乘子字典，键为(i,j,τ)，τ可以是起点时间t或终点时间s
        incompatible_arcs_dict: 不兼容弧集合字典，键为(i,j,τ)，值为与该时空位置冲突的弧集合
        train_idx: 当前列车的索引
        ax: 可选的绘图对象，用于可视化
    
    返回:
        path: 最优路径列表
        dist: 距离字典
        prev: 前驱节点字典
        selected_arcs: 列车所选择的弧
    """
    all_graph_nodes = valid_nodes
    selected_arcs = []  # 记录该列车选择的所有弧

    dist = {node: math.inf for node in all_graph_nodes}   # 记录每个节点的最短距离
    prev = {node: None for node in all_graph_nodes}       # 记录每个节点的前驱节点
    
    dist[start_node] = 0
    
    # 使用优先队列（最小堆）存储待访问节点 (距离, 节点)
    pq = [(0, start_node)]
    
    visited_nodes = set() # 用于优化，避免重复处理已确定最短路径的节点

    print(f"列车 {train_idx}：开始使用带拉格朗日乘子的Dijkstra算法寻找最短路径...") 
    
    while pq:
        # 取出当前距离最小的节点
        d, u = heapq.heappop(pq)
        
        # 如果取出的距离大于已记录的距离，说明找到了更短的路径，跳过
        if d > dist[u]:
            continue
            
        # 如果节点已经访问过
        if u in visited_nodes:
            continue
        visited_nodes.add(u)

        # 如果到达终点，则结束搜索
        if u == end_node:
            print(f"列车 {train_idx}：已找到终点节点 {end_node}，最短距离（代价）为 {dist[u]:.2f}")
            break
            
        # 遍历当前节点的邻居
        if u in graph: # 确保当前节点有出边
            for v, base_cost in graph[u]:
                # 检查邻居节点是否有效
                if v in dist:
                    # 构建当前弧 (i,j,t,s,u,v)
                    i, t, v_speed = u
                    j, s, u_speed = v
                    arc = (i, j, t, s, v_speed, u_speed)
                    
                    # 计算拉格朗日惩罚项
                    penalty = 0.0
                    
                    # 检查起点时间及其headway范围内的不兼容集合
                    i_pos, j_pos = min(i, j), max(i, j)
                    
                    # 只根据当前弧的起点和终点时间添加惩罚
                    start_key = (i_pos, j_pos, t)
                    end_key = (i_pos, j_pos, s)
                    
                    # 检查并添加惩罚
                    if start_key in multipliers:
                        penalty += multipliers[start_key]
                    if end_key in multipliers:
                        penalty += multipliers[end_key]

                    # 记录加入惩罚的弧
                    if penalty > 0:
                        drawn_arcs.add(arc)
                        
                    # 计算总成本 = 基础成本 + 拉格朗日惩罚
                    total_cost = base_cost + penalty
                    
                    # 计算通过当前节点 u 到达邻居 v 的新距离
                    alt = dist[u] + total_cost
                    # 如果找到了更短的路径
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                        # 将邻居节点加入优先队列
                        heapq.heappush(pq, (alt, v)) 

    # 检查是否找到了到达终点的路径
    if dist.get(end_node, math.inf) == math.inf:
        print(f"列车 {train_idx}：错误：无法从起点 {start_node} 找到到达终点 {end_node} 的路径。")
        # 可视化处理（类似原函数）
        if ax is not None:
            print(f"列车 {train_idx}：终点不可达，正在标记从起点可达的节点...")
            reachable_nodes_labeled = False
            for node, distance in dist.items():
                if distance != math.inf:
                    if isinstance(node, tuple) and len(node) == 3:
                        i_node, t_node, v_node = node
                        label_to_add = ""
                        if not reachable_nodes_labeled:
                            label_to_add = f'列车{train_idx}可达节点'
                            reachable_nodes_labeled = True
                        ax.scatter(i_node, t_node, v_node, color=f'C{train_idx}', s=60, marker='p', alpha=0.8, label=label_to_add)
            
            if not reachable_nodes_labeled and start_node in dist and dist[start_node] == 0:
                print(f"列车 {train_idx}：除了起点外，没有其他可达节点。")
            plt.title(f"列车{train_idx} STS网格模型（未找到路径）")
            plt.show()
        
        plt.show(block=True)
        return None, dist, prev, []

    # 重建最短路径
    path = []
    current = end_node
    
    while current and prev[current]:
        # 记录弧 (i,j,t,s,u,v)
        next_node = current
        prev_node = prev[current]
        i, t, v_speed = prev_node
        j, s, u_speed = next_node
        arc = (i, j, t, s, v_speed, u_speed)
        selected_arcs.append(arc)
        
        path.append(current)
        current = prev[current]
    
    if current:  # 添加起点
        path.append(current)
    
    path.reverse()  # 反转路径，从起点到终点
    selected_arcs.reverse()  # 反转弧，与路径匹配
    
    return path, dist, prev, selected_arcs, ax, drawn_arcs

from main import enhanced_filter_valid_nodes
# 定义处理单个列车的函数
def process_single_train(train_idx, train_schedule, all_nodes, train_max_speed, select_near_plan, station_names, max_distance, a_max):
    # 筛选单个列车的有效节点
    station_positions = [int(round(station[0])) for station in train_schedule]
    valid_nodes = filter_valid_nodes(all_nodes, station_positions, train_schedule, train_max_speed)
    
    if select_near_plan:
        print(f"列车 {train_idx}：根据计划时刻表筛选节点...")
        valid_nodes = filter_nodes_near_plan(valid_nodes, train_schedule, station_names, max_distance)
    
    # 创建图结构
    graph, len_valid_arcs = create_valid_arcs(valid_nodes, train_schedule, a_max)
    # graph, len_valid_arcs = batch_create_valid_arcs(valid_nodes, train_schedule, a_max)
    print(f"列车 {train_idx}：添加的有效弧数量: {len_valid_arcs}")
    
    return train_idx, valid_nodes, graph, len_valid_arcs


def identify_incompatible_arcs(graph, headway):
    """
    识别不兼容弧集合，每个弧生成两个不兼容集合：
    1. 以起点时间为键 (i,j,t)
    2. 以终点时间为键 (i,j,s)
    
    参数: 
        graph: 图的邻接表表示
        headway: 最小间隔时间
        
    返回:
        incompatible_arcs_dict: 不兼容弧集合字典，键为(i,j,τ)，值为与该时空位置冲突的弧集合
    """
    # 初始化不兼容弧集合字典
    incompatible_arcs_dict = {}
    
    # 收集所有有效弧 (i,j,t,s,u,v)
    all_arcs = []
    for from_node in graph:
        for to_node, _ in graph[from_node]:
            i, t, v = from_node
            j, s, u = to_node
            all_arcs.append((i, j, t, s, v, u))
    
    print(f"总共有 {len(all_arcs)} 条有效弧")
    
    # 为每个弧的起点和终点时间创建不兼容集合
    for arc in all_arcs:
        i, j, t, s, v, u = arc
        
        # 确保物理链接的位置顺序一致(i < j)
        i_pos, j_pos = min(i, j), max(i, j)
        
        # 创建起点时间的键
        start_key = (i_pos, j_pos, t)
        if start_key not in incompatible_arcs_dict:
            incompatible_arcs_dict[start_key] = set()
        incompatible_arcs_dict[start_key].add(arc)
        
        # 创建终点时间的键
        end_key = (i_pos, j_pos, s)
        if end_key not in incompatible_arcs_dict:
            incompatible_arcs_dict[end_key] = set()
        incompatible_arcs_dict[end_key].add(arc)
        
        # 添加headway范围内的时间点
        for h in range(1, headway + 1):
            # 起点时间前后headway范围内的时间点
            for time_offset in [-h, h]:
                t_offset = t + time_offset
                t_key = (i_pos, j_pos, t_offset)
                if t_key not in incompatible_arcs_dict:
                    incompatible_arcs_dict[t_key] = set()
                incompatible_arcs_dict[t_key].add(arc)
            
            # 终点时间前后headway范围内的时间点
            for time_offset in [-h, h]:
                s_offset = s + time_offset
                s_key = (i_pos, j_pos, s_offset)
                if s_key not in incompatible_arcs_dict:
                    incompatible_arcs_dict[s_key] = set()
                incompatible_arcs_dict[s_key].add(arc)
    
    print(f"构建了 {len(incompatible_arcs_dict)} 个时空位置的不兼容弧集合")
    return incompatible_arcs_dict


def update_multipliers(multipliers, train_paths, step_size, headway):
    """
    使用次梯度法更新拉格朗日乘子
    
    参数:
        multipliers: 当前拉格朗日乘子字典，键为(i,j,τ)
        train_paths: 所有列车的路径弧列表
        incompatible_arcs_dict: 不兼容弧集合字典，键为(i,j,τ)，值为与该时空位置冲突的弧集合
        step_size: 次梯度更新步长
        
    返回:
        new_multipliers: 更新后的拉格朗日乘子
        total_violations: 约束违反总数
    """
    # new_multipliers = copy.deepcopy(multipliers)
    new_multipliers = multipliers
    total_violations = 0
    
    # 创建每个时空位置(i,j,τ)上的列车使用情况
    space_time_usage = {}
    
    # 收集所有列车所使用的弧
    for train_idx, train_arcs in enumerate(train_paths):
        for arc in train_arcs:
            i, j, t, s, _, _ = arc
            i_pos, j_pos = min(i, j), max(i, j)
            
            # 记录起点时间位置
            start_key = (i_pos, j_pos, t)
            if start_key not in space_time_usage:
                space_time_usage[start_key] = []
            space_time_usage[start_key].append(train_idx)
            
            # 记录终点时间位置
            end_key = (i_pos, j_pos, s)
            if end_key not in space_time_usage:
                space_time_usage[end_key] = []
            space_time_usage[end_key].append(train_idx)
            
            # 添加headway范围内的时间点 - 确保与构建不兼容弧集合时使用相同的headway值
            for h in range(1, headway + 1):  # 使用传入的headway参数
                # 起点时间前后的时间点
                for time_offset in [-h, h]:
                    t_offset = t + time_offset
                    t_key = (i_pos, j_pos, t_offset)
                    if t_key not in space_time_usage:
                        space_time_usage[t_key] = []
                    space_time_usage[t_key].append(train_idx)
                
                # 终点时间前后的时间点
                for time_offset in [-h, h]:
                    s_offset = s + time_offset
                    s_key = (i_pos, j_pos, s_offset)
                    if s_key not in space_time_usage:
                        space_time_usage[s_key] = []
                    space_time_usage[s_key].append(train_idx)
    
    # 检查冲突并更新乘子
    conflict_zones = []
    for space_time_key, trains in space_time_usage.items():
        # 如果同一时空位置有多列车使用，则存在冲突
        if len(set(trains)) > 1:
            violation = len(set(trains)) - 1  # 冲突的列车数减1
            total_violations += violation
            
            # 更新该时空位置的乘子
            if space_time_key not in new_multipliers:
                new_multipliers[space_time_key] = 0
            new_multipliers[space_time_key] = max(0, new_multipliers[space_time_key] + step_size * violation)

            i, j, t = space_time_key
            conflict_zones.append((i, t, len(set(trains))))
    
    # 打印调试信息，检查乘子是否有效影响路径
    print(f"总违反约束数: {total_violations}")  
    print(f"更新的乘子数量: {len(new_multipliers)}") 
    
    return new_multipliers, total_violations, conflict_zones


def visualize_penalized_arcs(graph, multipliers, headway, ax):
    """
    可视化三维空间中已经加入惩罚的弧
    
    参数:
        graph: 图的邻接表表示
        multipliers: 拉格朗日乘子字典
        ax: 3D绘图对象
    """
    penalized_arcs = []
    
    # 遍历所有弧，检查是否有惩罚
    for u in graph:
        for v, base_cost in graph[u]:
            i, t, v_speed = u
            j, s, u_speed = v
            
            # 检查弧是否与乘子位置相关（考虑headway）
            for h in range(headway + 1):
                for time_offset in range(-h, h+1):
                    t_key = (min(i, j), max(i, j), t + time_offset)
                    s_key = (min(i, j), max(i, j), s + time_offset)
                    
                    if t_key in multipliers and multipliers[t_key] > 0:
                        # 标记为惩罚弧
                        penalized_arcs.append((u, v, multipliers[t_key], 'start'))
                    if s_key in multipliers and multipliers[s_key] > 0:
                        # 标记为惩罚弧
                        penalized_arcs.append((u, v, multipliers[s_key], 'end'))
    
    print(f"找到 {len(penalized_arcs)} 条带惩罚的弧")
    
    if not penalized_arcs:
        print("没有找到带惩罚的弧，可能是因为乘子全为零或者位置键与弧不匹配")
        return
    
    # 绘制带惩罚的弧
    for arc_info in penalized_arcs:
        u, v, penalty, pos_type = arc_info
        i, t, v_speed = u
        j, s, u_speed = v
        
        # 设置弧的颜色，强度由惩罚值决定
        alpha = min(1.0, penalty / 10)  # 将惩罚值映射到0-1的透明度
        line_width = 1 + penalty / 5  # 惩罚值越大，线越粗
        
        # 绘制3D弧
        ax.plot([i, j], [t, s], [v_speed, u_speed], 
                color='red', alpha=alpha, linewidth=line_width)
        
        # 在弧的中点添加标记表示惩罚值
        mid_i = (i + j) / 2
        mid_t = (t + s) / 2
        mid_v = (v_speed + u_speed) / 2
        
        # 使用文本标签显示惩罚值
        if penalty > 1:  # 只显示显著的惩罚
            ax.text(mid_i, mid_t, mid_v, f"{penalty:.1f}", 
                    color='black', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))


def plot_penalty_distribution(multipliers_history, ax=None):
    """
    绘制迭代过程中惩罚分布的变化
    
    参数:
        multipliers_history: 迭代过程中记录的乘子历史
        ax: 可选的绘图对象
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    
    # 获取所有迭代中的最大乘子值
    max_multiplier = 0
    for mult_dict in multipliers_history:
        if mult_dict:
            max_multiplier = max(max_multiplier, max(mult_dict.values()))
    
    # 创建颜色映射
    cmap = plt.cm.get_cmap('viridis')
    
    # 为每次迭代绘制一条线
    for i, mult_dict in enumerate(multipliers_history):
        if not mult_dict:
            continue
        
        # 按值排序的乘子
        sorted_values = sorted(mult_dict.values(), reverse=True)
        
        # 绘制分布曲线
        color = cmap(i / len(multipliers_history))
        ax.plot(range(len(sorted_values)), sorted_values, 
                color=color, alpha=0.7, label=f'迭代 {i+1}')
    
    ax.set_xlabel('乘子索引 (按值降序排列)')
    ax.set_ylabel('乘子值')
    ax.set_title('惩罚分布的迭代变化')
    ax.legend()
    
    return ax


def visualize_path_violations(best_paths, best_selected_arcs, headway, ax):
    """
    可视化最优路径中违反约束的部分
    
    参数:
        best_paths: 最优路径列表
        best_selected_arcs: 最优路径对应的弧列表
        headway: 最小间隔时间
        ax: 3D绘图对象
    """
    # 创建时空位置使用字典
    space_time_usage = {}
    
    # 记录每个列车的弧使用情况
    train_arcs_dict = {}
    
    # 收集所有列车使用的时空位置
    for train_idx, arcs in enumerate(best_selected_arcs):
        train_arcs_dict[train_idx] = arcs
        for arc in arcs:
            i, j, t, s, v_speed, u_speed = arc
            i_pos, j_pos = min(i, j), max(i, j)
            
            # 记录起点时间位置
            start_key = (i_pos, j_pos, t)
            if start_key not in space_time_usage:
                space_time_usage[start_key] = []
            space_time_usage[start_key].append(train_idx)
            
            # 记录终点时间位置
            end_key = (i_pos, j_pos, s)
            if end_key not in space_time_usage:
                space_time_usage[end_key] = []
            space_time_usage[end_key].append(train_idx)
            
            # 添加headway范围内的时间点
            for h in range(1, headway + 1):
                # 起点时间前后的时间点
                for time_offset in [-h, h]:
                    t_offset = t + time_offset
                    t_key = (i_pos, j_pos, t_offset)
                    if t_key not in space_time_usage:
                        space_time_usage[t_key] = []
                    space_time_usage[t_key].append(train_idx)
                
                # 终点时间前后的时间点
                for time_offset in [-h, h]:
                    s_offset = s + time_offset
                    s_key = (i_pos, j_pos, s_offset)
                    if s_key not in space_time_usage:
                        space_time_usage[s_key] = []
                    space_time_usage[s_key].append(train_idx)
    
    # 找出冲突的时空位置
    conflict_positions = {}
    for space_time_key, trains in space_time_usage.items():
        if len(set(trains)) > 1:
            conflict_positions[space_time_key] = list(set(trains))
    
    # 找出对应的弧并可视化
    visualized_conflicts = set()  # 避免重复可视化
    
    if not conflict_positions:
        print("没有找到路径冲突！所有列车路径满足约束。")
        return
    
    print(f"找到 {len(conflict_positions)} 个冲突时空位置")
    
    # 绘制冲突区域和相应的列车弧
    for space_time_key, conflicting_trains in conflict_positions.items():
        i, j, t = space_time_key
        
        # 在图中标记冲突位置
        conflict_intensity = len(conflicting_trains)
        ax.scatter(i, t, 0, 
                   c='red', s=100+conflict_intensity*20, 
                   marker='x', alpha=0.7, zorder=10)
        
        # 为每个冲突列车找到对应的弧
        for train_idx in conflicting_trains:
            if train_idx not in train_arcs_dict:
                continue
                
            for arc in train_arcs_dict[train_idx]:
                arc_i, arc_j, arc_t, arc_s, v_speed, u_speed = arc
                arc_i_pos, arc_j_pos = min(arc_i, arc_j), max(arc_i, arc_j)
                
                # 检查弧是否与冲突位置相关
                if ((arc_i_pos == i and arc_j_pos == j) or (arc_i_pos == j and arc_j_pos == i)) and \
                   ((abs(arc_t - t) <= headway) or (abs(arc_s - t) <= headway)):
                    
                    # 创建唯一标识以避免重复绘制
                    conflict_id = (arc_i, arc_j, arc_t, arc_s, train_idx)
                    if conflict_id in visualized_conflicts:
                        continue
                    visualized_conflicts.add(conflict_id)
                    
                    # 高亮显示冲突弧
                    ax.plot([arc_i, arc_j], [arc_t, arc_s], [v_speed, u_speed], 
                            color=f'C{train_idx}', linewidth=4, linestyle=':', alpha=0.9,
                            zorder=5)
                    
                    # 添加冲突说明
                    mid_i = (arc_i + arc_j) / 2
                    mid_t = (arc_t + arc_s) / 2
                    mid_v = (v_speed + u_speed) / 2
                    
                    ax.text(mid_i, mid_t, mid_v, f"列车{train_idx}冲突", 
                            color=f'C{train_idx}', fontsize=10, 
                            bbox=dict(facecolor='white', alpha=0.7))
    
    # 添加冲突图例
    ax.scatter([], [], [], c='red', s=120, marker='x', 
               label='冲突时空位置')
    
    for train_idx in range(len(best_paths)):
        ax.plot([], [], [], color=f'C{train_idx}', linewidth=4, linestyle=':', 
                label=f'列车{train_idx}冲突弧')
    
    ax.legend(loc='upper left')

import os
def solve_multi_train_with_lagrangian(train_schedules, station_names, delta_d=5, delta_t=5, 
                                      speed_levels=5, time_diff_minutes=5*60, total_distance=50,
                                      max_distance=30, select_near_plan=True, a_max=5,
                                      train_max_speed=5, headway=2, max_iterations=50, debug_mode=False):
    """
    使用拉格朗日松弛法求解多列车时刻表问题
    
    参数:
        train_schedules: 列车时刻表列表，每个元素是一个列车的时刻表
        station_names: 车站名称字典，键为位置，值为站名
        delta_d: 空间单位长度，默认为5km
        delta_t: 时间单位长度，默认为5分钟
        speed_levels: 速度级别数量，默认为5
        time_diff_minutes: 时间窗口长度，默认为5小时
        total_distance: 线路总长度，默认为50km
        max_distance: 节点到计划时刻表直线的最大距离，默认为30
        select_near_plan: 是否使用计划表完成有效点的筛选，默认为True
        a_max: 最大加速度约束，默认为5
        train_max_speed: 列车最大速度，默认为5
        headway: 列车之间的最小间隔时间，默认为2
        max_iterations: 拉格朗日迭代的最大次数，默认为50
        
    返回:
        all_train_paths: 所有列车的最优路径
    """
    start_time = time.time()
    
    # 初步计算
    time_nodes = math.ceil(time_diff_minutes / delta_t)
    space_segments = math.ceil(total_distance / delta_d)
    
    time_nodes, space_segments, delta_t, delta_d = check_time_nodes_and_space_segments(
        time_nodes, space_segments, delta_t, delta_d, time_diff_minutes, total_distance)
    
    # 转换站点时刻表为单位值
    train_schedules, station_names = trans_df_grid(train_schedules, station_names, delta_t, delta_d)
    
    # 确保train_schedules是列表的列表
    if not isinstance(train_schedules[0], list):
        train_schedules = [train_schedules]
    
    # 获取所有站点位置
    all_station_positions = []
    for train_schedule in train_schedules:
        station_positions = [int(round(station[0])) for station in train_schedule]
        all_station_positions.extend(station_positions)
    all_station_positions = list(set(all_station_positions))  # 去重
    
    # 设置坐标范围
    space_range = np.arange(0, space_segments + 1)
    time_range = np.arange(0, time_nodes + 1)
    speed_range = np.arange(0, speed_levels + 1)
    
    # 创建节点集合
    all_nodes = set()
    for i in space_range:
        for t in time_range:
            for v in speed_range:
                all_nodes.add((i, t, v))
    print(f"初始网格节点总数: {len(all_nodes)}")
    
    # 获取所有列车的有效节点和图
    train_valid_nodes = []
    train_graphs = []
    
    # 创建适合当前屏幕的图形
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    
    # 获取屏幕分辨率信息并据此设置图形大小
    fig = plt.figure(figsize=(10, 8))  # 使用更适合标准屏幕的尺寸
    
    # 调整DPI以确保图形清晰且适合屏幕
    fig.set_dpi(100)  # 设置适当的DPI值
    
    # 创建3D子图
    ax = fig.add_subplot(111, projection='3d')
    
    # 调整视角使图形更容易查看
    ax.view_init(elev=30, azim=45)  # 设置仰角和方位角
    
    # 使用线程池并行处理所有列车 根据系统CPU核心数自动调整线程数，但最大不超过8个线程
    num_workers = min(os.cpu_count(), 16) if os.cpu_count() else 4
    print(f"当前系统使用 {num_workers} 个线程进行并行计算")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务到线程池
        future_to_train = {
            executor.submit(
                process_single_train, 
                train_idx, 
                train_schedule, 
                all_nodes, 
                train_max_speed, 
                select_near_plan, 
                station_names, 
                max_distance, 
                a_max
            ): train_idx 
            for train_idx, train_schedule in enumerate(train_schedules)
        }
        
        # 收集结果并保持原有顺序
        results = [None] * len(train_schedules)
        for future in concurrent.futures.as_completed(future_to_train):
            train_idx, valid_nodes, graph, _ = future.result()
            train_valid_nodes.append(valid_nodes)
            train_graphs.append(graph)
            results[train_idx] = (valid_nodes, graph)
        
        # 确保结果按列车索引顺序排列
        train_valid_nodes = []
        train_graphs = []
        for valid_nodes, graph in results:
            train_valid_nodes.append(valid_nodes)
            train_graphs.append(graph)
    
    # region 单线程处理
    # for train_idx, train_schedule in enumerate(train_schedules):
    #     # 筛选单个列车的有效节点
    #     station_positions = [int(round(station[0])) for station in train_schedule]
    #     valid_nodes = filter_valid_nodes(all_nodes, station_positions, train_schedule, train_max_speed)
        
    #     if select_near_plan:
    #         print(f"列车 {train_idx}：根据计划时刻表筛选节点...")
    #         valid_nodes = filter_nodes_near_plan(valid_nodes, train_schedule, station_names, max_distance)
        
    #     train_valid_nodes.append(valid_nodes)
        
    #     # 创建图结构
    #     graph, len_valid_arcs = spatial_index_create_valid_arcs(valid_nodes, train_schedule, a_max)
    #     print(f"列车 {train_idx}：添加的有效弧数量: {len_valid_arcs}")
    #     train_graphs.append(graph) 
    # endregion

    # 设置图例和标题
    ax.legend()
    ax.set_title('多列车STS网格模型可视化')
    ax.set_xlabel('空间维度 (站点位置/km)')
    ax.set_ylabel('时间维度 (时间/min)')
    ax.set_zlabel('速度维度 (速度/km/min)')
    
    # region 识别所有列车的不兼容弧集合
    # all_incompatible_arcs_dict = []
    # for train_idx, graph in enumerate(train_graphs):
    #     print(f"计算列车 {train_idx} 的不兼容弧集合...")
    #     incompatible_arcs_dict = identify_incompatible_arcs(graph, headway)
    #     all_incompatible_arcs_dict.append(incompatible_arcs_dict)
    # endregion

    # 拉格朗日乘子迭代求解
    multipliers = {}  # 初始化拉格朗日乘子为0
    step_size = 100.  # 增大初始步长加快乘子增长
    
    best_paths = None
    best_violations = float('inf')
    
    # 创建每个列车的起点和终点
    train_start_end_nodes = []
    for train_idx, train_schedule in enumerate(train_schedules):
        start_node = (train_schedule[0][0], train_schedule[0][2], 0)
        end_node = (train_schedule[-1][0], train_schedule[-1][2], 0)
        train_start_end_nodes.append((start_node, end_node))
    
    print("开始拉格朗日松弛迭代...")
    
    # 初始化乘子历史记录 
    drawn_arcs = set()
    multipliers_history = []

    for iteration in range(max_iterations):
        print(f"\n迭代 {iteration + 1}/{max_iterations}")
        
        # 为每个列车找最短路径
        all_train_paths = []
        all_selected_arcs = []
        
        for train_idx, (valid_nodes, graph) in enumerate(zip(train_valid_nodes, train_graphs)):
            start_node, end_node = train_start_end_nodes[train_idx]
            
            # 使用拉格朗日乘子寻找最短路径
            path, dist, prev, selected_arcs, ax, drawn_arcs = find_optimal_path_with_lagrangian(
                valid_nodes, graph, start_node, end_node, multipliers, train_idx, headway, ax, drawn_arcs)
            
            if path is None:
                print(f"列车 {train_idx} 无法找到路径，跳过当前迭代")
                all_train_paths = None
                break
            
            all_train_paths.append(path)
            all_selected_arcs.append(selected_arcs)
        
        if all_train_paths is None:
            # 如果任一列车无法找到路径，减小步长重试
            step_size *= 0.5
            print(f"减小步长至 {step_size}，继续迭代")
            if step_size < 0.01:
                print("步长过小，停止迭代")
                break
            continue

        if iteration == max_iterations - 1 and debug_mode:  # 最后一次迭代时可视化
            print("可视化带惩罚的弧...") 
            
            # 遍历所有已记录的弧
            penalty_arcs_count = 0
            for arc_key in drawn_arcs:
                # 获取弧的起点和终点坐标
                i, j, t, s, v_speed, u_speed = arc_key
                start_point = (i, t, v_speed)
                end_point = (j, s, u_speed)
                # 绘制紫色线条表示带惩罚的弧
                ax.plot([start_point[0], end_point[0]], 
                        [start_point[1], end_point[1]], 
                        [start_point[2], end_point[2]], 
                        color='purple', linewidth=4, alpha=0.7)
                penalty_arcs_count += 1 
                    
            print(f"共绘制了 {penalty_arcs_count} 条带惩罚的弧")
        
        # 更新拉格朗日乘子
        new_multipliers, violations, conflict_zones = update_multipliers(multipliers, all_selected_arcs, step_size, headway)
        
        if iteration == max_iterations - 1 and debug_mode:
            # 在绘图时标记冲突区域
            if conflict_zones:
                conflict_data = np.array(conflict_zones)
                sc = ax.scatter(conflict_data[:, 0], conflict_data[:, 1], 
                            np.zeros_like(conflict_data[:, 2]), 
                            c=conflict_data[:, 2], cmap='plasma',
                            s=100, marker='o', edgecolors='black',
                            label='列车冲突区域')
                plt.colorbar(sc, ax=ax, label='冲突列车数')
        
        print(f"约束违反数: {violations}")
        print(f"乘子数量: {len(new_multipliers)}")
        
        # 保存最佳解
        if violations < best_violations:
            best_violations = violations
            best_paths = all_train_paths
            print(f"找到更好的解，违反数: {violations}")
        
        # 检查终止条件
        if violations == 0:
            print("找到无冲突解，但继续迭代以优化解")
            # 当没有冲突时，减少乘子的罚值，使算法能够探索更多可能的解
            for key in multipliers:
                multipliers[key] *= 0.5  # 将所有乘子的值减半
            print("已将所有乘子的罚值减半，以便探索更优解")
        
        # 更新乘子和步长
        multipliers = new_multipliers
        if iteration % 10 == 9:  # 每10次迭代调整步长
            step_size *= 0.8
            print(f"调整步长至 {step_size}")
        
        # 更新乘子后记录到历史
        multipliers_history.append(copy.deepcopy(new_multipliers))
    
    end_time = time.time()
    print(f"\n拉格朗日松弛求解完成，耗时 {end_time - start_time:.2f} 秒")
    
    # 绘制最终结果
    if best_paths:
        print("绘制最终结果...")
        for train_idx, path in enumerate(best_paths):
            train_color = f'C{train_idx}'
            path_space = [node[0] for node in path]
            path_time = [node[1] for node in path]
            path_speed = [node[2] for node in path]
            
            # 绘制路径线
            ax.plot(path_space, path_time, path_speed, color=train_color, linewidth=3, 
                    linestyle='-', marker='o', markersize=5, label=f'列车{train_idx}最优路径')
            
            # 添加路径起点和终点的特殊标记
            ax.scatter(path_space[0], path_time[0], path_speed[0], 
                    color=train_color, s=150, marker='*', edgecolors='black')
            ax.scatter(path_space[-1], path_time[-1], path_speed[-1], 
                    color=train_color, s=150, marker='*', edgecolors='black')
            
            # 绘制在空间-时间平面的投影
            ax.plot(path_space, path_time, np.zeros_like(path_speed), 
                    color=train_color, linewidth=2, linestyle='--', alpha=0.7)
    if debug_mode:
        # 绘制惩罚分布变化
        fig_penalty = plt.figure(figsize=(10, 6))
        ax_penalty = fig_penalty.add_subplot(111)
        plot_penalty_distribution(multipliers_history, ax_penalty)
        plt.savefig('penalty_distribution.png')
    
    # 绘制最终结果后，可视化路径违反情况
    # if best_paths and best_violations > 0:
    #     print("可视化路径冲突...")
    #     # 获取最佳路径对应的弧
    #     best_selected_arcs = []
    #     for train_idx, path in enumerate(best_paths):
    #         # 重建路径对应的弧
    #         selected_arcs = []
    #         for i in range(len(path) - 1):
    #             prev_node = path[i]
    #             next_node = path[i+1]
    #             i, t, v_speed = prev_node
    #             j, s, u_speed = next_node
    #             arc = (i, j, t, s, v_speed, u_speed)
    #             selected_arcs.append(arc)
    #         best_selected_arcs.append(selected_arcs)
        
    #     # 可视化路径冲突
    #     visualize_path_violations(best_paths, best_selected_arcs, headway, ax)
            # 完善图形显示
    finalize_plot(ax, space_segments, time_nodes, speed_levels, delta_d, delta_t)
    
    return best_paths


if __name__ == "__main__":
    # 定义两个列车的时刻表
    # 定义更复杂的列车时刻表，增加列车数量和站点数量
    train_schedule1 = [
        [0, 0, '8:00', '8:00', 4],      # 北京南站(始发站) 
        [40, 1, '8:12', '8:12', 4],     # 固安站(通过站)
        [70, 2, '8:20', '8:25', 4],     # 廊坊站(停靠站) 
        [100, 1, '8:35', '8:35', 4],    # 胜芳站(通过站)
        [140, 2, '8:50', '8:55', 4],    # 天津站(停靠站)
        [170, 1, '9:10', '9:10', 4],    # 塘沽站(通过站)
        [200, 0, '9:30', '9:30', 4],    # 滨海站(终点站)
    ]
    
    train_schedule2 = [
        [0, 0, '8:10', '8:10', 4],      # 北京南站(始发站) 
        [40, 1, '8:20', '8:20', 4],     # 固安站(通过站)
        [70, 2, '8:28', '8:33', 4],     # 廊坊站(停靠站) 
        [100, 1, '8:45', '8:45', 4],    # 胜芳站(通过站)
        [140, 2, '9:00', '9:05', 4],    # 天津站(停靠站)
        [170, 1, '9:18', '9:18', 4],    # 塘沽站(通过站)
        [200, 0, '9:40', '9:40', 4],    # 滨海站(终点站)
    ]
    
    train_schedule3 = [
        [0, 0, '8:20', '8:20', 4],      # 北京南站(始发站) 
        [40, 1, '8:35', '8:35', 4],     # 固安站(通过站)
        [70, 2, '8:43', '8:48', 4],     # 廊坊站(停靠站) 
        [100, 1, '8:55', '8:55', 4],    # 胜芳站(通过站)
        [140, 2, '9:10', '9:15', 4],    # 天津站(停靠站)
        [170, 1, '9:32', '9:32', 4],    # 塘沽站(通过站)
        [200, 0, '9:50', '9:50', 4],    # 滨海站(终点站)
    ]
    
    train_schedule4 = [
        [0, 0, '8:30', '8:30', 4],      # 北京南站(始发站) 
        [40, 1, '8:40', '8:40', 4],     # 固安站(通过站)
        [70, 2, '8:48', '8:53', 4],     # 廊坊站(停靠站) 
        [100, 1, '9:05', '9:05', 4],    # 胜芳站(通过站)
        [140, 2, '9:20', '9:25', 4],    # 天津站(停靠站)
        [170, 1, '9:38', '9:38', 4],    # 塘沽站(通过站)
        [200, 0, '10:00', '10:00', 4],  # 滨海站(终点站)
    ]
    
    train_schedule5 = [
        [0, 0, '8:40', '8:40', 4],      # 北京南站(始发站) 
        [40, 1, '8:55', '8:55', 4],     # 固安站(通过站)
        [70, 2, '9:03', '9:08', 4],     # 廊坊站(停靠站) 
        [100, 1, '9:15', '9:15', 4],    # 胜芳站(通过站)
        [140, 2, '9:30', '9:35', 4],    # 天津站(停靠站)
        [170, 1, '9:52', '9:52', 4],    # 塘沽站(通过站)
        [200, 0, '10:10', '10:10', 4],  # 滨海站(终点站)
    ]
    
    train_schedule6 = [
        [0, 0, '8:50', '8:50', 4],      # 北京南站(始发站) 
        [40, 1, '9:00', '9:00', 4],     # 固安站(通过站)
        [70, 2, '9:08', '9:13', 4],     # 廊坊站(停靠站) 
        [100, 1, '9:25', '9:25', 4],    # 胜芳站(通过站)
        [140, 2, '9:40', '9:45', 4],    # 天津站(停靠站)
        [170, 1, '9:58', '9:58', 4],    # 塘沽站(通过站)
        [200, 0, '10:20', '10:20', 4],  # 滨海站(终点站)
    ]
    
    train_schedule7 = [
        [0, 0, '9:00', '9:00', 4],      # 北京南站(始发站) 
        [40, 1, '9:15', '9:15', 4],     # 固安站(通过站)
        [70, 2, '9:23', '9:28', 4],     # 廊坊站(停靠站) 
        [100, 1, '9:35', '9:35', 4],    # 胜芳站(通过站)
        [140, 2, '9:50', '9:55', 4],    # 天津站(停靠站)
        [170, 1, '10:12', '10:12', 4],  # 塘沽站(通过站)
        [200, 0, '10:30', '10:30', 4],  # 滨海站(终点站)
    ]
    
    train_schedule8 = [
        [0, 0, '9:10', '9:10', 4],      # 北京南站(始发站) 
        [40, 1, '9:20', '9:20', 4],     # 固安站(通过站)
        [70, 2, '9:28', '9:33', 4],     # 廊坊站(停靠站) 
        [100, 1, '9:45', '9:45', 4],    # 胜芳站(通过站)
        [140, 2, '10:00', '10:05', 4],  # 天津站(停靠站)
        [170, 1, '10:18', '10:18', 4],  # 塘沽站(通过站)
        [200, 0, '10:40', '10:40', 4],  # 滨海站(终点站)
    ]
    
    train_schedule9 = [
        [0, 0, '9:20', '9:20', 4],      # 北京南站(始发站) 
        [40, 1, '9:35', '9:35', 4],     # 固安站(通过站)
        [70, 2, '9:43', '9:48', 4],     # 廊坊站(停靠站) 
        [100, 1, '9:55', '9:55', 4],    # 胜芳站(通过站)
        [140, 2, '10:10', '10:15', 4],  # 天津站(停靠站)
        [170, 1, '10:32', '10:32', 4],  # 塘沽站(通过站)
        [200, 0, '10:50', '10:50', 4],  # 滨海站(终点站)
    ]
    
    train_schedule10 = [
        [0, 0, '9:30', '9:30', 4],      # 北京南站(始发站) 
        [40, 1, '9:40', '9:40', 4],     # 固安站(通过站)
        [70, 2, '9:48', '9:53', 4],     # 廊坊站(停靠站) 
        [100, 1, '10:05', '10:05', 4],  # 胜芳站(通过站)
        [140, 2, '10:20', '10:25', 4],  # 天津站(停靠站)
        [170, 1, '10:38', '10:38', 4],  # 塘沽站(通过站)
        [200, 0, '11:00', '11:00', 4],  # 滨海站(终点站)
    ]
    
    # 增加更多列车以测试算法性能
    train_schedule11 = [
        [0, 0, '9:40', '9:40', 4],      # 北京南站(始发站) 
        [40, 1, '9:55', '9:55', 4],     # 固安站(通过站)
        [70, 2, '10:03', '10:08', 4],   # 廊坊站(停靠站) 
        [100, 1, '10:15', '10:15', 4],  # 胜芳站(通过站)
        [140, 2, '10:30', '10:35', 4],  # 天津站(停靠站)
        [170, 1, '10:52', '10:52', 4],  # 塘沽站(通过站)
        [200, 0, '11:10', '11:10', 4],  # 滨海站(终点站)
    ]
    
    train_schedule12 = [
        [0, 0, '9:50', '9:50', 4],      # 北京南站(始发站) 
        [40, 1, '10:00', '10:00', 4],   # 固安站(通过站)
        [70, 2, '10:08', '10:13', 4],   # 廊坊站(停靠站) 
        [100, 1, '10:25', '10:25', 4],  # 胜芳站(通过站)
        [140, 2, '10:40', '10:45', 4],  # 天津站(停靠站)
        [170, 1, '10:58', '10:58', 4],  # 塘沽站(通过站)
        [200, 0, '11:20', '11:20', 4],  # 滨海站(终点站)
    ]
    
    train_schedule13 = [
        [0, 0, '10:00', '10:00', 4],    # 北京南站(始发站) 
        [40, 1, '10:15', '10:15', 4],   # 固安站(通过站)
        [70, 2, '10:23', '10:28', 4],   # 廊坊站(停靠站) 
        [100, 1, '10:35', '10:35', 4],  # 胜芳站(通过站)
        [140, 2, '10:50', '10:55', 4],  # 天津站(停靠站)
        [170, 1, '11:12', '11:12', 4],  # 塘沽站(通过站)
        [200, 0, '11:30', '11:30', 4],  # 滨海站(终点站)
    ]
    
    train_schedule14 = [
        [0, 0, '10:10', '10:10', 4],    # 北京南站(始发站) 
        [40, 1, '10:20', '10:20', 4],   # 固安站(通过站)
        [70, 2, '10:28', '10:33', 4],   # 廊坊站(停靠站) 
        [100, 1, '10:45', '10:45', 4],  # 胜芳站(通过站)
        [140, 2, '11:00', '11:05', 4],  # 天津站(停靠站)
        [170, 1, '11:18', '11:18', 4],  # 塘沽站(通过站)
        [200, 0, '11:40', '11:40', 4],  # 滨海站(终点站)
    ]
    
    train_schedule15 = [
        [0, 0, '10:20', '10:20', 4],    # 北京南站(始发站) 
        [40, 1, '10:35', '10:35', 4],   # 固安站(通过站)
        [70, 2, '10:43', '10:48', 4],   # 廊坊站(停靠站) 
        [100, 1, '10:55', '10:55', 4],  # 胜芳站(通过站)
        [140, 2, '11:10', '11:15', 4],  # 天津站(停靠站)
        [170, 1, '11:32', '11:32', 4],  # 塘沽站(通过站)
        [200, 0, '11:50', '11:50', 4],  # 滨海站(终点站)
    ]
    
    # 更新站点名称字典，增加新增站点
    station_names = {
        0: "北京南",
        40: "固安",
        70: "廊坊",
        100: "胜芳",
        140: "天津",
        170: "塘沽",
        200: "滨海"
    }
    
    # 使用拉格朗日松弛法求解多列车问题
    # 记录算法开始时间
    start_time = time.time()
    
    # 执行算法
    best_paths = solve_multi_train_with_lagrangian(
        [train_schedule1, train_schedule2, train_schedule3, train_schedule4, train_schedule5, train_schedule6, train_schedule7, train_schedule8, train_schedule9, train_schedule10], 
        # [train_schedule2, train_schedule3], 
        station_names,
        delta_d=0.5,      # 500米
        delta_t=0.5,      # 0.5分钟 30秒
        speed_levels=5, 
        time_diff_minutes=4*60,   # 调度范围是2个小时
        total_distance=300,    # 300km的线路
        max_distance=10,       # 25个格子
        headway=2,             # 2个时间单位的最小间隔
        max_iterations=10,      # 最多迭代50次
        select_near_plan=True,  # 是否使用计划表完成有效点的筛选
        debug_mode=False,        # 是否开启调试模式
    )
    
    # 计算并输出算法运行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"算法总运行时间: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")