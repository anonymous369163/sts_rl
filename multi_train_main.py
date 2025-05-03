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
    使用带有拉格朗日乘子的Dijkstra算法寻找最优路径，支持新的节点结构和股道约束
    
    参数:
        valid_nodes: 有效节点集合
        graph: 图的邻接表表示
        start_node: 起点节点 (位置, 时间, 速度, 是否进站, 股道)
        end_node: 终点节点 (位置, 时间, 速度, 是否进站, 股道)
        multipliers: 拉格朗日乘子字典，键为(i,j,τ,track_id)
        train_idx: 当前列车的索引
        headway: 最小间隔时间
        ax: 可选的绘图对象，用于可视化
        drawn_arcs: 已绘制的弧集合
    
    返回:
        path: 最优路径列表
        dist: 距离字典
        prev: 前驱节点字典
        selected_arcs: 列车所选择的弧
        ax: 更新后的绘图对象
        drawn_arcs: 更新后的已绘制弧集合
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
                    # 构建当前弧，支持新的节点结构
                    if len(u) >= 5 and len(v) >= 5:  # 新结构
                        i, t, v_speed, entry_flag_i, track_i = u
                        j, s, u_speed, entry_flag_j, track_j = v
                        
                        # 是否为车站内部弧
                        is_station_arc = (i == j) and (entry_flag_i == 0) and (entry_flag_j == 1) and (track_j is not None)
                        
                        # 构建增强的弧表示
                        arc = (i, j, t, s, v_speed, u_speed, entry_flag_i, entry_flag_j, track_j if is_station_arc else None)
                    else:  # 兼容原有结构
                        i, t, v_speed = u[:3]
                        j, s, u_speed = v[:3]
                        arc = (i, j, t, s, v_speed, u_speed)
                    
                    # 计算拉格朗日惩罚项
                    penalty = 0.0
                    
                    # 检查是否为车站内部弧，如果是则使用股道约束
                    if len(u) >= 5 and len(v) >= 5 and i == j and entry_flag_i == 0 and entry_flag_j == 1 and track_j is not None:
                        # 车站内部弧，使用股道信息
                        i_pos, j_pos = min(i, j), max(i, j)
                        
                        # 考虑当前弧的起点和终点时间及其附近的时间点
                        for time_offset in range(-headway, headway + 1):
                            # 起点附近的时间点
                            start_time = t + time_offset
                            start_key = (i_pos, j_pos, start_time, track_j)
                            
                            # 终点附近的时间点
                            end_time = s + time_offset
                            end_key = (i_pos, j_pos, end_time, track_j)
                            
                            # 检查并添加惩罚
                            if start_key in multipliers:
                                penalty += multipliers[start_key]
                            if end_key in multipliers:
                                penalty += multipliers[end_key]
                    else:
                        # 非车站内部弧，使用原有约束
                        i_pos, j_pos = min(i, j), max(i, j)
                        
                        # 考虑当前弧的起点和终点时间及其附近的时间点
                        for time_offset in range(-headway, headway + 1):
                            # 起点附近的时间点
                            start_time = t + time_offset
                            start_key = (i_pos, j_pos, start_time, None)
                            
                            # 终点附近的时间点
                            end_time = s + time_offset
                            end_key = (i_pos, j_pos, end_time, None)
                            
                            # 检查并添加惩罚
                            if start_key in multipliers:
                                penalty += multipliers[start_key]
                            if end_key in multipliers:
                                penalty += multipliers[end_key]
                            
                            # 检查旧格式的键
                            old_start_key = (i_pos, j_pos, start_time)
                            old_end_key = (i_pos, j_pos, end_time)
                            
                            if old_start_key in multipliers:
                                penalty += multipliers[old_start_key]
                            if old_end_key in multipliers:
                                penalty += multipliers[old_end_key]
                    
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
                    node_pos, node_time = node[0], node[1]  # 提取位置和时间，适用于新旧节点结构
                    node_speed = node[2]  # 提取速度
                    label_to_add = ""
                    if not reachable_nodes_labeled:
                        label_to_add = f'列车{train_idx}可达节点'
                        reachable_nodes_labeled = True
                    ax.scatter(node_pos, node_time, node_speed, color=f'C{train_idx}', s=60, marker='p', alpha=0.8, label=label_to_add)
            
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
        # 记录弧，支持新的节点结构
        next_node = current
        prev_node = prev[current]
        
        if len(prev_node) >= 5 and len(next_node) >= 5:  # 新结构
            i, t, v_speed, entry_flag_i, track_i = prev_node
            j, s, u_speed, entry_flag_j, track_j = next_node
            
            # 是否为车站内部弧
            is_station_arc = (i == j) and (entry_flag_i == 0) and (entry_flag_j == 1) and (track_j is not None)
            
            # 构建增强的弧表示
            arc = (i, j, t, s, v_speed, u_speed, entry_flag_i, entry_flag_j, track_j if is_station_arc else None)
        else:  # 兼容原有结构
            i, t, v_speed = prev_node[:3]
            j, s, u_speed = next_node[:3]
            arc = (i, j, t, s, v_speed, u_speed)
        
        selected_arcs.append(arc)
        
        path.append(current)
        current = prev[current]
    
    if current:  # 添加起点
        path.append(current)
    
    path.reverse()  # 反转路径，从起点到终点
    selected_arcs.reverse()  # 反转弧，与路径匹配
    
    return path, dist, prev, selected_arcs, ax, drawn_arcs

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

def filter_valid_nodes(all_nodes, station_positions, train_schedule, train_max_speed):
    """
    筛选有效节点，适配新的节点结构
    
    参数:
        all_nodes: 所有节点集合
        station_positions: 站点位置列表
        train_schedule: 列车时刻表
        train_max_speed: 列车最大速度
        
    返回:
        valid_nodes: 有效节点集合
    """
    valid_nodes = set()
    
    start_pos = train_schedule[0][0]
    end_pos = train_schedule[-1][0]
    start_time = train_schedule[0][2]
    end_time = train_schedule[-1][2]
    
    # 获取所有节点的位置范围和时间范围
    all_positions = set()
    all_times = set()
    all_speeds = set()
    for node in all_nodes:
        if len(node) >= 5:  # 适配新的节点结构
            i, t, v, entry_flag, track = node
            all_positions.add(i)
            all_times.add(t)
            all_speeds.add(v)
    
    min_pos, max_pos = min(all_positions), max(all_positions)
    min_time, max_time = min(all_times), max(all_times)
    
    # 筛选节点
    for node in all_nodes:
        if len(node) >= 5:  # 确保节点是新格式
            i, t, v, entry_flag, track = node
            
            # 检查位置是否在列车运行范围内
            if min(start_pos, end_pos) <= i <= max(start_pos, end_pos):
                # 检查时间是否在列车运行时间范围内
                if start_time <= t <= end_time:
                    # 检查速度是否不超过最大速度
                    if v <= train_max_speed:
                        valid_nodes.add(node)
    
    print(f"筛选出 {len(valid_nodes)} 个有效节点")
    return valid_nodes

def filter_nodes_near_plan(valid_nodes, train_schedule, station_names, max_distance):
    """
    根据计划时刻表筛选节点，适配新的节点结构
    
    参数:
        valid_nodes: 有效节点集合
        train_schedule: 列车时刻表
        station_names: 车站名称字典
        max_distance: 最大距离
        
    返回:
        filtered_nodes: 筛选后的节点集合
    """
    filtered_nodes = set()
    
    # 创建计划点字典，键为位置，值为时间
    plan_points = {}
    for station in train_schedule:
        pos, _, time = station[:3]
        plan_points[pos] = time
    
    # 处理车站点和非车站点
    for node in valid_nodes:
        if len(node) >= 5:  # 适配新的节点结构
            i, t, v, entry_flag, track = node
            
            # 检查节点位置是否在计划点中
            if i in plan_points:
                # 对于车站点，计算与计划时间的距离
                time_diff = abs(t - plan_points[i])
                if time_diff <= max_distance:
                    filtered_nodes.add(node)
            else:
                # 对于非车站点，找最近的左右计划点进行插值
                left_pos = None
                right_pos = None
                
                for pos in sorted(plan_points.keys()):
                    if pos < i:
                        left_pos = pos
                    elif pos > i:
                        right_pos = pos
                        break
                
                # 线性插值计算预期时间
                if left_pos is not None and right_pos is not None:
                    left_time = plan_points[left_pos]
                    right_time = plan_points[right_pos]
                    
                    # 插值计算预期时间
                    expected_time = left_time + (right_time - left_time) * (i - left_pos) / (right_pos - left_pos)
                    
                    # 计算与预期时间的距离
                    time_diff = abs(t - expected_time)
                    if time_diff <= max_distance:
                        filtered_nodes.add(node)
    
    print(f"根据计划时刻表筛选出 {len(filtered_nodes)} 个节点")
    return filtered_nodes

def create_valid_arcs(valid_nodes, train_schedule, a_max):
    """
    创建有效弧，适配新的节点结构并考虑车站内股道约束
    
    参数:
        valid_nodes: 有效节点集合
        train_schedule: 列车时刻表
        a_max: 最大加速度约束
        
    返回:
        graph: 图的邻接表表示
        len_valid_arcs: 有效弧数量
    """
    graph = {}
    valid_arcs_count = 0
    
    # 获取站点位置
    station_positions = [int(round(station[0])) for station in train_schedule]
    
    # 将节点按照位置和时间分组，方便后续查找
    nodes_by_pos_time = {}
    for node in valid_nodes:
        i, t, v, entry_flag, track = node
        if (i, t) not in nodes_by_pos_time:
            nodes_by_pos_time[(i, t)] = []
        nodes_by_pos_time[(i, t)].append(node)
    
    # 遍历所有节点创建弧
    for from_node in valid_nodes:
        i, t, v, entry_flag, track_from = from_node
        
        # 初始化图结构
        if from_node not in graph:
            graph[from_node] = []
        
        # 处理不同类型的节点
        if i in station_positions:
            # 车站节点
            if entry_flag == 0:  # 进站节点
                # 连接到同一车站的出站节点
                for to_node in valid_nodes:
                    j, s, u, to_entry_flag, track_to = to_node
                    
                    # 只连接同一车站的进站节点到出站节点
                    if j == i and to_entry_flag == 1 and s > t:
                        # 在车站内部，从进站节点到出站节点需要考虑股道
                        # 计算停站时间成本
                        stopping_time = s - t
                        min_stopping_time = 2  # 最小停站时间单位
                        
                        # 确保停站时间满足最小要求
                        if stopping_time >= min_stopping_time:
                            # 创建站内弧，成本基于停站时间
                            cost = stopping_time * 0.5  # 停站时间乘以权重因子
                            graph[from_node].append((to_node, cost))
                            valid_arcs_count += 1
            
            elif entry_flag == 1:  # 出站节点
                # 连接到其他地点的节点(非车站或其他车站的进站节点)
                for to_node in valid_nodes:
                    j, s, u, to_entry_flag, track_to = to_node
                    
                    # 只连接到非本站的节点或其他车站的进站节点
                    if j != i and s > t:
                        # 如果目标是另一个车站，确保连接到进站节点
                        if j in station_positions:
                            if to_entry_flag != 0:  # 不是进站节点，跳过
                                continue
                        
                        # 计算弧的成本
                        distance = abs(j - i)
                        time_diff = s - t
                        
                        # 检查速度变化是否符合加速度约束
                        if time_diff > 0:
                            speed_change = abs(u - v)
                            acceleration = speed_change / time_diff
                            
                            if acceleration <= a_max:
                                # 创建弧，成本基于距离和时间
                                cost = distance + time_diff * 0.1
                                graph[from_node].append((to_node, cost))
                                valid_arcs_count += 1
        
        else:
            # 非车站节点
            for to_node in valid_nodes:
                j, s, u, to_entry_flag, track_to = to_node
                
                # 只连接到时间更晚的节点
                if s > t:
                    # 如果目标是车站，确保连接到进站节点
                    if j in station_positions and to_entry_flag != 0:
                        continue
                    
                    # 计算弧的成本
                    distance = abs(j - i)
                    time_diff = s - t
                    
                    # 检查速度变化是否符合加速度约束
                    if time_diff > 0:
                        speed_change = abs(u - v)
                        acceleration = speed_change / time_diff
                        
                        if acceleration <= a_max:
                            # 创建弧，成本基于距离和时间
                            cost = distance + time_diff * 0.1
                            graph[from_node].append((to_node, cost))
                            valid_arcs_count += 1
    
    return graph, valid_arcs_count

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


def detect_train_overtaking(train_paths):
    """
    检测列车之间的越行情况
    
    参数:
        train_paths: 所有列车的路径弧列表
        
    返回:
        overtaking_events: 越行事件列表
    """
    # 用于检测列车越行的数据结构
    train_positions = {}  # 记录每个列车在每个时间点的位置
    overtaking_events = []  # 记录越行事件
    
    # 首先收集每个列车在每个时间点的位置信息
    for train_idx, train_arcs in enumerate(train_paths):
        for arc in train_arcs:
            i, j, t, s, _, _ = arc
            # 确保i是较小的位置，j是较大的位置
            i_pos, j_pos = min(i, j), max(i, j)
            
            # 记录列车在起点时间t的位置i
            if t not in train_positions:
                train_positions[t] = {}
            train_positions[t][train_idx] = i_pos
            
            # 记录列车在终点时间s的位置j
            if s not in train_positions:
                train_positions[s] = {}
            train_positions[s][train_idx] = j_pos
    
    # 按时间顺序排序
    sorted_times = sorted(train_positions.keys())
    
    # 检测越行情况
    if len(train_paths) >= 2:  # 至少有两列车才可能发生越行
        # 记录每列车的前进方向（1表示向前，-1表示向后）
        train_directions = {}
        
        # 确定每列车的运行方向
        for train_idx, train_arcs in enumerate(train_paths):
            if train_arcs:
                first_arc = train_arcs[0]
                last_arc = train_arcs[-1]
                start_pos = min(first_arc[0], first_arc[1])
                end_pos = min(last_arc[0], last_arc[1])
                train_directions[train_idx] = 1 if end_pos > start_pos else -1
        
        # 检查相邻时间点之间的位置变化
        for t_idx in range(1, len(sorted_times)):
            prev_time = sorted_times[t_idx-1]
            curr_time = sorted_times[t_idx]
            
            # 获取在这两个时间点都有记录的列车
            common_trains = set(train_positions[prev_time].keys()) & set(train_positions[curr_time].keys())
            
            # 如果至少有两列车在这两个时间点都有记录
            if len(common_trains) >= 2:
                train_list = list(common_trains)
                
                # 比较每对列车
                for i in range(len(train_list)):
                    for j in range(i+1, len(train_list)):
                        train_a = train_list[i]
                        train_b = train_list[j]
                        
                        # 确保两列车运行方向相同
                        if train_a in train_directions and train_b in train_directions:
                            if train_directions[train_a] == train_directions[train_b]:
                                # 获取两列车在前后时间点的位置
                                a_prev_pos = train_positions[prev_time][train_a]
                                a_curr_pos = train_positions[curr_time][train_a]
                                b_prev_pos = train_positions[prev_time][train_b]
                                b_curr_pos = train_positions[curr_time][train_b]
                                
                                # 检测越行：如果列车A在前一时刻在列车B前面，但在当前时刻在列车B后面
                                if (a_prev_pos < b_prev_pos and a_curr_pos > b_curr_pos) or \
                                   (a_prev_pos > b_prev_pos and a_curr_pos < b_curr_pos):
                                    # 记录越行事件
                                    overtaking_event = {
                                        "time_interval": (prev_time, curr_time),
                                        "space_interval": (min(a_prev_pos, a_curr_pos, b_prev_pos, b_curr_pos),
                                                          max(a_prev_pos, a_curr_pos, b_prev_pos, b_curr_pos)),
                                        "trains": (train_a, train_b),
                                        "location": (a_prev_pos, a_curr_pos, b_prev_pos, b_curr_pos)
                                    }
                                    overtaking_events.append(overtaking_event)
    return overtaking_events


def update_multipliers(multipliers, train_paths, step_size, headway, debug_mode=False, ax=None):
    """
    使用次梯度法更新拉格朗日乘子，考虑股道维度
    
    参数:
        multipliers: 当前拉格朗日乘子字典，键为(i,j,τ,track_id)
        train_paths: 所有列车的路径弧列表
        step_size: 次梯度更新步长
        headway: 最小间隔时间
        debug_mode: 是否开启调试模式
        ax: 绘图对象
        
    返回:
        new_multipliers: 更新后的拉格朗日乘子
        total_violations: 约束违反总数
        conflict_zones: 冲突区域
    """
    # new_multipliers = copy.deepcopy(multipliers)
    new_multipliers = multipliers
    total_violations = 0
    
    # 创建每个时空位置(i,j,τ,track_id)上的列车使用情况
    space_time_usage = {} 

    if debug_mode:
        # 检测列车越行情况
        overtaking_events = detect_train_overtaking(train_paths)
        
        # 输出越行检测结果
        if overtaking_events:
            print("\n检测到以下越行情况:")
            for idx, event in enumerate(overtaking_events):
                print(f"越行事件 {idx+1}:")
                print(f"  时间区间: {event['time_interval'][0]} - {event['time_interval'][1]}")
                print(f"  空间区间: {event['space_interval'][0]} - {event['space_interval'][1]}")
                print(f"  涉及列车: {event['trains'][0]} 和 {event['trains'][1]}")
        else:
            print("\n未检测到列车越行情况")
    
    # 检测越行位置的时间间隔是否满足headway约束
    if debug_mode and 'overtaking_events' in locals() and overtaking_events:
        print("\n检查越行位置的时间间隔约束:")
        for idx, event in enumerate(overtaking_events):
            train_a, train_b = event['trains']
            a_prev_pos, a_curr_pos, b_prev_pos, b_curr_pos = event['location']
            prev_time, curr_time = event['time_interval']
            
            # 查找两列车在越行位置的具体时间
            train_a_times = {}
            train_b_times = {}
            
            # 收集列车A的时间信息
            for arc in train_paths[train_a]:
                i, j, t, s, _, _, entry_flag_i, entry_flag_j, track_id = arc if len(arc) >= 9 else (*arc, None, None, None)
                if i == a_prev_pos and j == a_curr_pos:
                    train_a_times[i] = t
                    train_a_times[j] = s
                elif i == a_curr_pos and j == a_prev_pos:
                    train_a_times[i] = t
                    train_a_times[j] = s
            
            # 收集列车B的时间信息
            for arc in train_paths[train_b]:
                i, j, t, s, _, _, entry_flag_i, entry_flag_j, track_id = arc if len(arc) >= 9 else (*arc, None, None, None)
                if i == b_prev_pos and j == b_curr_pos:
                    train_b_times[i] = t
                    train_b_times[j] = s
                elif i == b_curr_pos and j == b_prev_pos:
                    train_b_times[i] = t
                    train_b_times[j] = s
            
            # 检查越行位置的时间间隔
            headway_violations = []
            for pos in set([a_prev_pos, a_curr_pos, b_prev_pos, b_curr_pos]):
                if pos in train_a_times and pos in train_b_times:
                    time_diff = abs(train_a_times[pos] - train_b_times[pos])
                    if time_diff < headway:
                        headway_violations.append((pos, time_diff))
            
            # 输出结果
            print(f"越行事件 {idx+1} (列车 {train_a} 和 {train_b}):")
            if headway_violations:
                print(f"  发现时间间隔违反 (要求间隔>={headway}):")
                for pos, time_diff in headway_violations:
                    print(f"    位置 {pos}: 时间间隔为 {time_diff}")
            else:
                print(f"所有位置的时间间隔均满足headway>={headway}的约束")
    
    # 在三维空间中标记越行位置的时间信息
    if debug_mode and 'overtaking_events' in locals() and overtaking_events and ax is not None:
        print("\n在三维空间中标记越行位置的时间信息...")
        for idx, event in enumerate(overtaking_events):
            train_a, train_b = event['trains']
            a_prev_pos, a_curr_pos, b_prev_pos, b_curr_pos = event['location']
            prev_time, curr_time = event['time_interval']
            
            # 收集列车A的时间和速度信息
            train_a_info = {}  # 格式: {位置: (时间, 速度)}
            train_b_info = {}
            
            # 收集列车A的信息
            for arc in train_paths[train_a]:
                i, j, t, s, v_speed, u_speed, entry_flag_i, entry_flag_j, track_id = arc if len(arc) >= 9 else (*arc, None, None, None)
                if i == a_prev_pos and j == a_curr_pos:
                    train_a_info[i] = (t, v_speed)
                    train_a_info[j] = (s, u_speed)
                elif i == a_curr_pos and j == a_prev_pos:
                    train_a_info[i] = (t, v_speed)
                    train_a_info[j] = (s, u_speed)
            
            # 收集列车B的信息
            for arc in train_paths[train_b]:
                i, j, t, s, v_speed, u_speed, entry_flag_i, entry_flag_j, track_id = arc if len(arc) >= 9 else (*arc, None, None, None)
                if i == b_prev_pos and j == b_curr_pos:
                    train_b_info[i] = (t, v_speed)
                    train_b_info[j] = (s, u_speed)
                elif i == b_curr_pos and j == b_prev_pos:
                    train_b_info[i] = (t, v_speed)
                    train_b_info[j] = (s, u_speed)
            
            # 在三维空间中标记这些特殊点
            special_positions = set([a_prev_pos, a_curr_pos, b_prev_pos, b_curr_pos])
            for pos in special_positions:
                # 标记列车A的点
                if pos in train_a_info:
                    time_a, speed_a = train_a_info[pos]
                    ax.scatter(pos, time_a, speed_a, 
                              color=f'C{train_a}', s=200, marker='D', 
                              edgecolors='black', linewidth=2, 
                              label=f'列车{train_a}越行点' if idx == 0 and pos == a_prev_pos else "")
                    ax.text(pos, time_a, speed_a + 0.5, 
                           f'列车{train_a}\n位置{pos}\n时间{time_a}', 
                           color=f'C{train_a}', fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.7))
                
                # 标记列车B的点
                if pos in train_b_info:
                    time_b, speed_b = train_b_info[pos]
                    ax.scatter(pos, time_b, speed_b, 
                              color=f'C{train_b}', s=200, marker='D', 
                              edgecolors='black', linewidth=2, 
                              label=f'列车{train_b}越行点' if idx == 0 and pos == b_prev_pos else "")
                    ax.text(pos, time_b, speed_b + 0.5, 
                           f'列车{train_b}\n位置{pos}\n时间{time_b}', 
                           color=f'C{train_b}', fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.7))
                
                # 如果两列车在同一位置有时间记录，则连接它们以显示时间差
                if pos in train_a_info and pos in train_b_info:
                    time_a, speed_a = train_a_info[pos]
                    time_b, speed_b = train_b_info[pos]
                    time_diff = abs(time_a - time_b)
                    
                    # 用虚线连接两个点，显示时间差
                    ax.plot([pos, pos], [time_a, time_b], [speed_a, speed_b], 
                           color='red' if time_diff < headway else 'green', 
                           linestyle='--', linewidth=2, alpha=0.8)
                    
                    # 在连线中间标记时间差
                    mid_time = (time_a + time_b) / 2
                    mid_speed = (speed_a + speed_b) / 2
                    ax.text(pos, mid_time, mid_speed, 
                           f'Δt={time_diff}', 
                           color='red' if time_diff < headway else 'green', 
                           fontsize=12, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.9))
            
            print(f"已标记越行事件 {idx+1} 的特殊点 (列车 {train_a} 和 {train_b})")


    # 收集所有列车所使用的弧
    for train_idx, train_arcs in enumerate(train_paths):
        for arc in train_arcs:
            # 检查弧的格式，支持新旧格式
            if len(arc) >= 9:  # 新格式：包含进出站标志和股道信息
                i, j, t, s, v_speed, u_speed, entry_flag_i, entry_flag_j, track_id = arc
            else:  # 旧格式：不包含进出站标志和股道信息
                i, j, t, s, v_speed, u_speed = arc
                entry_flag_i = entry_flag_j = track_id = None
            
            i_pos, j_pos = min(i, j), max(i, j)
            
            # 检查是否是车站内部弧（进站到出站）
            is_station_arc = entry_flag_i == 0 and entry_flag_j == 1 and i == j and track_id is not None
            
            # 计算弧上所有可能的空间位置点
            space_positions = list(range(i_pos, j_pos + 1))
            
            # 计算时间的线性插值
            if i_pos != j_pos:  # 避免除以零
                time_step = (s - t) / (j_pos - i_pos)
                for pos in space_positions:
                    # 计算当前位置对应的时间（线性插值）
                    curr_time = int(t + (pos - i_pos) * time_step)
                    
                    # 记录当前空间-时间位置
                    # 对于车站内部弧，添加股道信息
                    if is_station_arc:
                        curr_key = (pos, pos, curr_time, track_id)
                    else:
                        curr_key = (pos, pos, curr_time, None)
                    
                    if curr_key not in space_time_usage:
                        space_time_usage[curr_key] = []
                    space_time_usage[curr_key].append(train_idx)
                    
                    # 添加headway范围内的时间点
                    for h in range(1, headway + 1):
                        for time_offset in [-h, h]:
                            offset_time = curr_time + time_offset
                            if is_station_arc:
                                offset_key = (pos, pos, offset_time, track_id)
                            else:
                                offset_key = (pos, pos, offset_time, None)
                            
                            if offset_key not in space_time_usage:
                                space_time_usage[offset_key] = []
                            space_time_usage[offset_key].append(train_idx)
            else:
                # 如果是同一空间位置（垂直弧）
                # 记录起点和终点，以及它们之间的所有时间点
                start_time, end_time = min(t, s), max(t, s)
                for curr_time in range(start_time, end_time + 1):
                    if is_station_arc:
                        curr_key = (i_pos, j_pos, curr_time, track_id)
                    else:
                        curr_key = (i_pos, j_pos, curr_time, None)
                    
                    if curr_key not in space_time_usage:
                        space_time_usage[curr_key] = []
                    space_time_usage[curr_key].append(train_idx)
                    
                    # 添加headway范围内的时间点
                    for h in range(1, headway + 1):
                        for time_offset in [-h, h]:
                            offset_time = curr_time + time_offset
                            if is_station_arc:
                                offset_key = (i_pos, j_pos, offset_time, track_id)
                            else:
                                offset_key = (i_pos, j_pos, offset_time, None)
                            
                            if offset_key not in space_time_usage:
                                space_time_usage[offset_key] = []
                            space_time_usage[offset_key].append(train_idx)
    
    # 检查冲突并更新乘子
    conflict_zones = []
    for space_time_key, trains in space_time_usage.items():
        # 解析空间-时间键
        if len(space_time_key) == 4:
            i, j, t, track_id = space_time_key
        else:
            i, j, t = space_time_key
            track_id = None
        
        # 判断是否是车站节点
        is_station_node = track_id is not None
        
        # 如果同一时空位置有多列车使用，则存在冲突
        # 对于不同股道的站内弧，不视为冲突
        if len(set(trains)) > 1:
            # 只计算非车站节点或者同股道车站节点的冲突
            if not is_station_node or (is_station_node and track_id is not None):
                violation = len(set(trains)) - 1  # 冲突的列车数减1
                total_violations += violation
                
                # 更新该时空位置的乘子
                if space_time_key not in new_multipliers:
                    new_multipliers[space_time_key] = 0
                new_multipliers[space_time_key] = max(0, new_multipliers[space_time_key] + step_size * violation)

                conflict_zones.append((i, t, len(set(trains))))
    
    # 打印调试信息
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
def solve_multi_train_with_lagrangian(train_schedules, station_names, train_reschedules=None, delta_d=5, delta_t=5, 
                                     speed_levels=5, time_diff_minutes=5*60, total_distance=50,
                                     max_distance=30, select_near_plan=True, a_max=5,
                                     train_max_speed=5, headway=5, max_iterations=50, debug_mode=False, draw_plan=True):
    """
    使用拉格朗日松弛法求解多列车时刻表问题
    
    参数:
        train_schedules: 列车计划时刻表列表，每个元素是一个列车的时刻表，用于路径规划和约束
        train_reschedules: 列车重调度方案列表，用于节点筛选，默认为None(使用train_schedules)
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

    # 将headway从分钟转换为时间单位
    headway = math.ceil(headway / delta_t)   
    
    # 如果未提供重调度方案，则使用计划时刻表
    if train_reschedules is None:
        train_reschedules = train_schedules
    
    # 初步计算
    time_nodes = math.ceil(time_diff_minutes / delta_t)
    space_segments = math.ceil(total_distance / delta_d)
    
    time_nodes, space_segments, delta_t, delta_d = check_time_nodes_and_space_segments(
        time_nodes, space_segments, delta_t, delta_d, time_diff_minutes, total_distance)
    
    # 转换计划时刻表和重调度方案为单位值
    train_schedules, station_names = trans_df_grid(train_schedules, station_names, delta_t, delta_d)
    train_reschedules, _ = trans_df_grid(train_reschedules, station_names, delta_t, delta_d)
    
    # 确保train_schedules和train_reschedules是列表的列表
    if not isinstance(train_schedules[0], list):
        train_schedules = [train_schedules]
    if not isinstance(train_reschedules[0], list):
        train_reschedules = [train_reschedules]
        
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
                # 检查该位置是否为车站
                is_station = i in all_station_positions
                
                if is_station:
                    # 对于车站节点，创建进站和出站两种节点
                    # 进站节点: (位置, 时间, 速度, 0, None)，其中0表示进站
                    all_nodes.add((i, t, v, 0, None))
                    
                    # 对于出站节点，为每个可能的股道创建节点
                    # 根据车站规模设置股道数量，这里假设所有车站都有相同数量的股道
                    num_tracks = 3  # 可以根据实际情况调整每个车站的股道数
                    for track in range(num_tracks):
                        # 出站节点: (位置, 时间, 速度, 1, 股道编号)，其中1表示出站
                        all_nodes.add((i, t, v, 1, track))
                else:
                    # 对于非车站节点，保持原有结构，但增加标记和默认股道值
                    # 非车站节点: (位置, 时间, 速度, None, None)
                    all_nodes.add((i, t, v, None, None))
    print(f"初始网格节点总数: {len(all_nodes)}")
    
    # 获取所有列车的有效节点和图
    train_valid_nodes = []
    train_graphs = []
    
    # 创建适合当前屏幕的图形
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    
    # 获取屏幕分辨率信息并据此设置图形大小
    fig = plt.figure(figsize=(10, 8))
    fig.set_dpi(100)
    
    # 创建3D子图
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)
    
    # 使用线程池并行处理所有列车
    num_workers = min(os.cpu_count(), 16) if os.cpu_count() else 4
    print(f"当前系统使用 {num_workers} 个线程进行并行计算")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务到线程池，使用重调度方案进行节点筛选
        future_to_train = {
            executor.submit(
                process_single_train, 
                train_idx, 
                train_reschedule,  # 使用重调度方案进行节点筛选 
                all_nodes, 
                train_max_speed, 
                select_near_plan, 
                station_names, 
                max_distance, 
                a_max
            ): train_idx 
            for train_idx, train_reschedule in enumerate(train_reschedules)
        }
        
        # 收集结果并保持原有顺序
        results = [None] * len(train_reschedules)
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
    step_size = 10000.  # 增大初始步长加快乘子增长
    
    best_paths = None
    best_violations = float('inf')
    
    # 创建每个列车的起点和终点（使用计划时刻表）
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
        draw_debug_mode = iteration == max_iterations - 1 and debug_mode
        new_multipliers, violations, conflict_zones = update_multipliers(multipliers, all_selected_arcs, step_size, headway, draw_debug_mode, ax)
        
        if draw_debug_mode:
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
            # 绘制在空间-速度平面的投影
            ax.plot(path_space, np.zeros_like(path_time), path_speed, 
                    color=train_color, linewidth=2, linestyle='-.', alpha=0.7)
            # 绘制在时间-速度平面的投影
            ax.plot(np.zeros_like(path_space), path_time, path_speed, 
                    color=train_color, linewidth=2, linestyle=':', alpha=0.7)
        print("绘制出计划时刻表")
        for each_schedule in train_schedules:
            draw_plan_on_sts_grid(ax, each_schedule, space_segments, time_nodes, draw_plan=draw_plan)
    if debug_mode:
        # 绘制惩罚分布变化
        fig_penalty = plt.figure(figsize=(10, 6))
        ax_penalty = fig_penalty.add_subplot(111)
        plot_penalty_distribution(multipliers_history, ax_penalty)
        plt.savefig('penalty_distribution.png')
    
    # region 绘制最终结果后，可视化路径违反情况
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
    # endregion
            # 完善图形显示
    finalize_plot(ax, space_segments, time_nodes, speed_levels, delta_d, delta_t)
    
    return best_paths


if __name__ == "__main__":
    # 从文件导入重调度的列车时刻表
    from train_schedules import (
        train_schedule1, train_schedule2, train_schedule3, train_schedule4,
        train_schedule5, train_schedule6, train_schedule7, train_schedule8,
        train_schedule9, train_schedule10, train_schedule11, train_schedule12,
        train_schedule13, train_schedule14, train_schedule15
    )

    # 从文件导入计划列车时刻表
    from train_schedules import (
        train_timetable1, train_timetable2, train_timetable3, train_timetable4,
        train_timetable5, train_timetable6, train_timetable7, train_timetable8,
        train_timetable9, train_timetable10, train_timetable11, train_timetable12,
        train_timetable13, train_timetable14, train_timetable15
    )
    
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
        [train_timetable1, train_timetable2],  # 计划时刻表  
        station_names,
        train_reschedules=[train_schedule1, train_schedule2],  # 重调度方案
        delta_d=1,
        delta_t=1,
        speed_levels=5, 
        time_diff_minutes=4*60,   # 调度范围是2个小时
        total_distance=300,    # 300km的线路
        max_distance=15,       # 25个格子
        headway=3,             # 3个时间单位的最小间隔
        max_iterations=100,      # 最多迭代50次
        select_near_plan=True,  # 是否使用计划表完成有效点的筛选
        debug_mode=False,        # 是否开启调试模式
        draw_plan=True,         # 是否绘制计划时刻表
    )
    
    # 计算并输出算法运行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"算法总运行时间: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")