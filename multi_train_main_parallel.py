"""
使用拉格朗日乘子法求解多列车时刻表的STS网格模型
基于原始STS网格模型，添加列车间耦合约束，并使用拉格朗日松弛法求解
并行处理版本
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


def find_optimal_path_with_lagrangian(valid_nodes, graph, start_node, end_node, 
                                     multipliers, incompatible_arcs_dict, train_idx, ax=None):
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
    
    # 统计乘子影响
    penalty_applied_count = 0
    total_penalty = 0.0
    
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
                    
                    # 检查起点时间的不兼容集合
                    start_key = (min(i, j), max(i, j), t)
                    if start_key in multipliers:
                        penalty += multipliers[start_key]
                    
                    # 检查终点时间的不兼容集合
                    end_key = (min(i, j), max(i, j), s)
                    if end_key in multipliers:
                        penalty += multipliers[end_key]
                    
                    # 使用更强的惩罚系数，确保乘子能有效影响路径选择
                    penalty_factor = 5.0
                    penalty *= penalty_factor
                    
                    # 统计乘子影响
                    if penalty > 0:
                        penalty_applied_count += 1
                        total_penalty += penalty
                    
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

    # 输出乘子影响统计
    if penalty_applied_count > 0:
        print(f"列车 {train_idx}：应用了 {penalty_applied_count} 次乘子惩罚，平均惩罚值: {total_penalty/penalty_applied_count:.2f}")
    else:
        print(f"列车 {train_idx}：未应用任何乘子惩罚，可能导致路径不变")

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
    
    return path, dist, prev, selected_arcs


def identify_incompatible_arcs(valid_nodes, graph, headway):
    """
    识别不兼容弧集合，每个弧生成两个不兼容集合：
    1. 以起点时间为键 (i,j,t)
    2. 以终点时间为键 (i,j,s)
    
    参数:
        valid_nodes: 有效节点集合
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


def update_multipliers(multipliers, train_paths, incompatible_arcs_dict, step_size, headway):
    """
    使用次梯度法更新拉格朗日乘子
    
    参数:
        multipliers: 当前拉格朗日乘子字典，键为(i,j,τ)
        train_paths: 所有列车的路径弧列表
        incompatible_arcs_dict: 不兼容弧集合字典，键为(i,j,τ)，值为与该时空位置冲突的弧集合
        step_size: 次梯度更新步长
        headway: 列车间隔时间，与identify_incompatible_arcs函数中使用的参数相同
        
    返回:
        new_multipliers: 更新后的拉格朗日乘子
        total_violations: 约束违反总数
    """
    new_multipliers = copy.deepcopy(multipliers)
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
    
    # 检查冲突并更新乘子
    for space_time_key, trains in space_time_usage.items():
        # 如果同一时空位置有多列车使用，则存在冲突
        if len(set(trains)) > 1:
            violation = len(set(trains)) - 1  # 冲突的列车数减1
            total_violations += violation
            
            # 更新该时空位置的乘子，使用更大的增长因子
            if space_time_key not in new_multipliers:
                new_multipliers[space_time_key] = 0
            new_multipliers[space_time_key] = max(0, new_multipliers[space_time_key] + step_size * violation)
    
    # 打印调试信息，检查乘子是否有效影响路径
    print(f"总违反约束数: {total_violations}")
    print(f"更新的乘子数量: {len(new_multipliers)}")
    
    # 检查乘子值的分布，帮助判断是否足够大来影响路径选择
    if new_multipliers:
        multiplier_values = list(new_multipliers.values())
        print(f"乘子最小值: {min(multiplier_values):.2f}, 最大值: {max(multiplier_values):.2f}, 平均值: {sum(multiplier_values)/len(multiplier_values):.2f}")
    
    return new_multipliers, total_violations


# 并行处理函数 - 处理单个列车的有效节点和图结构创建
def process_train_nodes_graph(train_idx, train_schedule, all_nodes, station_positions, 
                              train_max_speed, select_near_plan, max_distance, 
                              station_names, a_max):
    """并行处理单个列车的节点和图结构"""
    print(f"列车 {train_idx}：处理有效节点和图结构...")
    
    # 筛选单个列车的有效节点
    valid_nodes = filter_valid_nodes(all_nodes, station_positions, train_schedule, train_max_speed)
    
    if select_near_plan:
        print(f"列车 {train_idx}：根据计划时刻表筛选节点...")
        valid_nodes = filter_nodes_near_plan(valid_nodes, train_schedule, station_names, max_distance)
    
    # 创建图结构
    graph, len_valid_arcs = create_valid_arcs(valid_nodes, train_schedule, a_max)
    print(f"列车 {train_idx}：添加的有效弧数量: {len_valid_arcs}")
    
    return train_idx, valid_nodes, graph


# 并行处理函数 - 计算不兼容弧集合
def process_incompatible_arcs(train_idx, valid_nodes, graph, headway):
    """并行计算不兼容弧集合"""
    print(f"计算列车 {train_idx} 的不兼容弧集合...")
    incompatible_arcs_dict = identify_incompatible_arcs(valid_nodes, graph, headway)
    return train_idx, incompatible_arcs_dict


# 并行处理函数 - 寻找最短路径
def process_shortest_path(train_idx, valid_nodes, graph, start_node, end_node, 
                         multipliers, incompatible_arcs_dict):
    """并行寻找最短路径"""
    path, dist, prev, selected_arcs = find_optimal_path_with_lagrangian(
        valid_nodes, graph, start_node, end_node, multipliers, incompatible_arcs_dict, train_idx)
    return train_idx, path, dist, prev, selected_arcs


def check_path_multiplier_interaction(multipliers, all_selected_arcs, headway):
    """
    检查路径和乘子之间的交互关系，诊断乘子是否有效影响路径
    
    参数:
        multipliers: 当前拉格朗日乘子字典
        all_selected_arcs: 所有列车选择的弧
        headway: 列车间隔时间
        
    返回:
        has_interaction: 是否存在有效交互
        interaction_info: 交互信息描述
    """
    if not multipliers:
        return False, "乘子字典为空，无法影响路径"
    
    # 收集所有路径使用的时空位置
    path_space_time_keys = set()
    
    for train_idx, train_arcs in enumerate(all_selected_arcs):
        for arc in train_arcs:
            i, j, t, s, _, _ = arc
            i_pos, j_pos = min(i, j), max(i, j)
            
            # 添加起点和终点时间位置
            path_space_time_keys.add((i_pos, j_pos, t))
            path_space_time_keys.add((i_pos, j_pos, s))
            
            # 添加headway范围内的时间点
            for h in range(1, headway + 1):
                for time_offset in [-h, h]:
                    path_space_time_keys.add((i_pos, j_pos, t + time_offset))
                    path_space_time_keys.add((i_pos, j_pos, s + time_offset))
    
    # 检查路径位置和乘子位置的交集
    multiplier_keys = set(multipliers.keys())
    intersection = multiplier_keys.intersection(path_space_time_keys)
    
    interaction_count = len(intersection)
    total_multipliers = len(multiplier_keys)
    total_path_keys = len(path_space_time_keys)
    
    # 检查交集中的乘子值
    if interaction_count > 0:
        avg_value = sum(multipliers[key] for key in intersection) / interaction_count
        max_value = max(multipliers[key] for key in intersection)
        
        info = (f"路径与乘子有 {interaction_count} 个交集位置 "
                f"({interaction_count/total_multipliers*100:.1f}% 的乘子与路径相关), "
                f"交集平均乘子值: {avg_value:.2f}, 最大值: {max_value:.2f}")
        
        return True, info
    else:
        return False, f"路径与乘子没有交集，乘子总数: {total_multipliers}, 路径位置总数: {total_path_keys}"


def solve_multi_train_with_lagrangian_parallel(train_schedules, station_names, delta_d=5, delta_t=5, 
                                              speed_levels=5, time_diff_minutes=5*60, total_distance=50,
                                              max_distance=30, select_near_plan=True, a_max=5,
                                              train_max_speed=5, headway=2, max_iterations=50,
                                              max_workers=None):
    """
    使用拉格朗日松弛法并行求解多列车时刻表问题
    
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
        max_workers: 并行处理的最大工作线程数，默认为None（使用系统默认值）
        
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
    
    # 创建每个列车的起点和终点
    train_start_end_nodes = []
    for train_idx, train_schedule in enumerate(train_schedules):
        start_node = (train_schedule[0][0], train_schedule[0][2], 0)
        end_node = (train_schedule[-1][0], train_schedule[-1][2], 0)
        train_start_end_nodes.append((start_node, end_node))
    
    # 并行处理每个列车的有效节点和图结构
    print("并行处理列车的有效节点和图结构...")
    train_valid_nodes = [None] * len(train_schedules)
    train_graphs = [None] * len(train_schedules)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 准备并行任务
        futures = []
        for train_idx, train_schedule in enumerate(train_schedules):
            station_positions = [int(round(station[0])) for station in train_schedule]
            futures.append(
                executor.submit(
                    process_train_nodes_graph,
                    train_idx, train_schedule, all_nodes, station_positions,
                    train_max_speed, select_near_plan, max_distance,
                    station_names, a_max
                )
            )
        
        # 获取结果
        for future in concurrent.futures.as_completed(futures):
            try:
                train_idx, valid_nodes, graph = future.result()
                train_valid_nodes[train_idx] = valid_nodes
                train_graphs[train_idx] = graph
            except Exception as exc:
                print(f'生成列车节点和图结构时出现异常: {exc}')
    
    # 设置图例和标题
    ax.legend()
    ax.set_title('多列车STS网格模型可视化')
    ax.set_xlabel('空间维度 (站点位置/km)')
    ax.set_ylabel('时间维度 (时间/min)')
    ax.set_zlabel('速度维度 (速度/km/min)')
    
    # 并行识别所有列车的不兼容弧集合
    print("并行计算不兼容弧集合...")
    all_incompatible_arcs_dict = [None] * len(train_schedules)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 准备并行任务
        futures = []
        for train_idx, (valid_nodes, graph) in enumerate(zip(train_valid_nodes, train_graphs)):
            futures.append(
                executor.submit(
                    process_incompatible_arcs,
                    train_idx, valid_nodes, graph, headway
                )
            )
        
        # 获取结果
        for future in concurrent.futures.as_completed(futures):
            try:
                train_idx, incompatible_arcs_dict = future.result()
                all_incompatible_arcs_dict[train_idx] = incompatible_arcs_dict
            except Exception as exc:
                print(f'计算不兼容弧集合时出现异常: {exc}')
    
    # 拉格朗日乘子迭代求解
    multipliers = {}  # 初始化拉格朗日乘子为0
    step_size = 10.0   # 增大初始步长，使乘子更容易影响路径选择
    
    best_paths = None
    best_violations = float('inf')
    no_improvement_count = 0   # 记录连续没有改进的次数
    previous_violations = float('inf')  # 上一次迭代的违反数
    
    print("开始拉格朗日松弛迭代...")
    
    for iteration in range(max_iterations):
        print(f"\n迭代 {iteration + 1}/{max_iterations}")
        
        # 并行为每个列车找最短路径
        all_train_paths = [None] * len(train_schedules)
        all_selected_arcs = [None] * len(train_schedules)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 准备并行任务
            futures = []
            for train_idx, (valid_nodes, graph) in enumerate(zip(train_valid_nodes, train_graphs)):
                start_node, end_node = train_start_end_nodes[train_idx]
                futures.append(
                    executor.submit(
                        process_shortest_path,
                        train_idx, valid_nodes, graph, start_node, end_node,
                        multipliers, all_incompatible_arcs_dict[train_idx]
                    )
                )
            
            # 获取结果
            for future in concurrent.futures.as_completed(futures):
                try:
                    train_idx, path, dist, prev, selected_arcs = future.result()
                    all_train_paths[train_idx] = path
                    all_selected_arcs[train_idx] = selected_arcs
                except Exception as exc:
                    print(f'寻找最短路径时出现异常: {exc}')
        
        # 检查是否所有列车都找到了路径
        if None in all_train_paths:
            print("存在列车无法找到路径，减小步长重试")
            step_size *= 0.5
            print(f"减小步长至 {step_size}，继续迭代")
            if step_size < 0.01:
                print("步长过小，停止迭代")
                break
            continue
        
        # 更新拉格朗日乘子
        new_multipliers, violations = update_multipliers(
            multipliers, all_selected_arcs, all_incompatible_arcs_dict[0], step_size, headway)
        
        print(f"约束违反数: {violations}")
        print(f"乘子数量: {len(new_multipliers)}")
        
        # 检查乘子和路径的交互情况
        has_interaction, interaction_info = check_path_multiplier_interaction(
            new_multipliers, all_selected_arcs, headway)
        print(f"乘子与路径的交互: {interaction_info}")
        
        # 如果没有交互，可能需要调整参数
        if not has_interaction and len(new_multipliers) > 0:
            print("警告：乘子与路径没有交互，尝试增加步长或检查headway参数")
            # 增加步长以增强影响
            step_size *= 2.0
            print(f"增加步长至 {step_size}")
        
        # 检查是否有改进
        if violations < best_violations:
            best_violations = violations
            best_paths = copy.deepcopy(all_train_paths)  # 深拷贝以保证不会被后续迭代修改
            print(f"找到更好的解，违反数: {violations}")
            no_improvement_count = 0  # 重置计数器
        else:
            no_improvement_count += 1
            print(f"未找到更好的解，当前连续未改进次数: {no_improvement_count}")
        
        # 检查是否与上次迭代结果相同
        if violations == previous_violations:
            print("警告: 违反数与上次迭代相同，可能陷入局部最优")
            
            # 如果连续5次没有改进，调整步长
            if no_improvement_count >= 5:
                if step_size < 100:  # 如果步长还不够大，增加步长
                    step_size *= 2.0
                    print(f"连续{no_improvement_count}次未改进，增大步长至 {step_size}")
                    no_improvement_count = 0
                else:  # 如果步长已经很大，转为减小步长
                    step_size *= 0.5
                    print(f"连续{no_improvement_count}次未改进，减小步长至 {step_size}")
                    no_improvement_count = 0
        
        previous_violations = violations
        
        # 检查终止条件
        if violations == 0:
            print("找到无冲突解，停止迭代")
            break
        
        # 更新乘子和步长
        multipliers = new_multipliers
        
        # 修改步长调整策略
        if iteration % 5 == 4:  # 每5次迭代调整步长
            if violations > previous_violations:
                # 如果状况变差，减小步长
                step_size *= 0.8
                print(f"违反数增加，减小步长至 {step_size}")
            elif violations == previous_violations and no_improvement_count >= 3:
                # 如果多次没改进，增大步长以跳出局部最优
                step_size *= 1.5
                print(f"连续{no_improvement_count}次无改进，增大步长至 {step_size}")
            else:
                # 正常情况，小幅减小步长
                step_size *= 0.95
                print(f"正常调整步长至 {step_size}")
    
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
    
    # 完善图形显示
    finalize_plot(ax, space_segments, time_nodes, speed_levels, delta_d, delta_t)
    
    return best_paths


if __name__ == "__main__":
    # 定义四个列车的时刻表
    train_schedule1 = [
        [0, 0, '8:00', '8:00', 4],  # 北京南站(始发站) 
        [70, 2, '8:20', '8:25', 4],  # 廊坊站(停靠站) 
        [140, 1, '9:00', '9:00', 4],  # 天津站(通过站)
        [200, 0, '9:30', '9:30', 4],  # 滨海站(终点站)
    ]
    
    train_schedule2 = [
        [0, 0, '8:10', '8:10', 4],  # 北京南站(始发站) 
        [70, 1, '8:30', '8:30', 4],  # 廊坊站(通过站) 
        [140, 2, '9:05', '9:10', 4],  # 天津站(停靠站)
        [200, 0, '9:40', '9:40', 4],  # 滨海站(终点站)
    ]

    train_schedule3 = [
        [0, 0, '8:20', '8:20', 4],  # 北京南站(始发站) 
        [70, 2, '8:40', '8:45', 4],  # 廊坊站(停靠站) 
        [140, 1, '9:15', '9:15', 4],  # 天津站(通过站)
        [200, 0, '9:50', '9:50', 4],  # 滨海站(终点站)
    ]
    
    train_schedule4 = [
        [0, 0, '8:30', '8:30', 4],  # 北京南站(始发站) 
        [70, 1, '8:50', '8:50', 4],  # 廊坊站(通过站) 
        [140, 2, '9:25', '9:30', 4],  # 天津站(停靠站)
        [200, 0, '10:00', '10:00', 4],  # 滨海站(终点站)
    ]
    
    station_names = {
        0: "北京南",
        70: "廊坊",
        140: "天津",
        200: "滨海"
    }
    
    # 使用拉格朗日松弛法并行求解多列车问题
    # 记录算法开始时间
    start_time = time.time()
    
    # 执行算法 - 使用四列车场景测试
    best_paths = solve_multi_train_with_lagrangian_parallel(
        [train_schedule1, train_schedule2, train_schedule3, train_schedule4], 
        station_names,
        delta_d=1,      # 1km
        delta_t=1,      # 1分钟
        speed_levels=5, 
        time_diff_minutes=2*60,   # 调度范围是2个小时
        total_distance=300,    # 300km的线路
        max_distance=10,       # 最大距离为10个单位
        headway=2,             # 2个时间单位的最小间隔
        max_iterations=30,     # 最多迭代30次
        select_near_plan=True,  # 使用计划表完成有效点的筛选
        max_workers=4,          # 并行处理的最大工作线程数
    )
    
    # 计算并输出算法运行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"算法总运行时间: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)") 