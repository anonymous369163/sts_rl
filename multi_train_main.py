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
    for space_time_key, trains in space_time_usage.items():
        # 如果同一时空位置有多列车使用，则存在冲突
        if len(set(trains)) > 1:
            violation = len(set(trains)) - 1  # 冲突的列车数减1
            total_violations += violation
            
            # 更新该时空位置的乘子
            if space_time_key not in new_multipliers:
                new_multipliers[space_time_key] = 0
            new_multipliers[space_time_key] = max(0, new_multipliers[space_time_key] + step_size * violation)
    
    # 打印调试信息，检查乘子是否有效影响路径
    print(f"总违反约束数: {total_violations}")
    print(f"更新的乘子数量: {len(new_multipliers)}")
    
    # 检查乘子是否与路径有交集
    multiplier_keys_set = set(new_multipliers.keys())
    path_keys_set = set()
    
    # 收集所有路径中使用的时空位置键
    for train_idx, train_arcs in enumerate(train_paths):
        for arc in train_arcs:
            i, j, t, s, _, _ = arc
            i_pos, j_pos = min(i, j), max(i, j)
            path_keys_set.add((i_pos, j_pos, t))
            path_keys_set.add((i_pos, j_pos, s))
            
            # 添加与不兼容弧集合构建时相同的headway范围
            for h in range(1, headway + 1):
                for time_offset in [-h, h]:
                    path_keys_set.add((i_pos, j_pos, t + time_offset))
                    path_keys_set.add((i_pos, j_pos, s + time_offset))
    
    # 检查交集
    intersection = multiplier_keys_set.intersection(path_keys_set)
    print(f"乘子与路径的交集大小: {len(intersection)}")
    if len(intersection) == 0:
        print("警告: 乘子与路径没有交集，这可能导致乘子无法影响最优路径选择")
        # 输出一些乘子和路径的样本以便调试
        print(f"乘子样本: {list(multiplier_keys_set)[:5] if multiplier_keys_set else '空'}")
        print(f"路径键样本: {list(path_keys_set)[:5] if path_keys_set else '空'}")
        
        # 检查键的格式是否一致
        if multiplier_keys_set and path_keys_set:
            multiplier_key_example = list(multiplier_keys_set)[0]
            path_key_example = list(path_keys_set)[0]
            print(f"乘子键格式示例: {multiplier_key_example}, 类型: {[type(x) for x in multiplier_key_example]}")
            print(f"路径键格式示例: {path_key_example}, 类型: {[type(x) for x in path_key_example]}")
            
            # 如果发现类型不匹配，尝试转换
            if any(type(a) != type(b) for a, b in zip(multiplier_key_example, path_key_example)):
                print("发现类型不匹配，尝试转换...")
                # 转换乘子键类型
                converted_multipliers = {}
                for key, value in new_multipliers.items():
                    converted_key = tuple(int(k) if isinstance(k, (int, float)) else k for k in key)
                    converted_multipliers[converted_key] = value
                new_multipliers = converted_multipliers
                
                # 重新检查交集
                multiplier_keys_set = set(new_multipliers.keys())
                converted_path_keys = set(tuple(int(k) if isinstance(k, (int, float)) else k for k in key) for key in path_keys_set)
                intersection = multiplier_keys_set.intersection(converted_path_keys)
                print(f"类型转换后的交集大小: {len(intersection)}")
    
    return new_multipliers, total_violations


def solve_multi_train_with_lagrangian(train_schedules, station_names, delta_d=5, delta_t=5, 
                                      speed_levels=5, time_diff_minutes=5*60, total_distance=50,
                                      max_distance=30, select_near_plan=True, a_max=5,
                                      train_max_speed=5, headway=2, max_iterations=50):
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
    
    for train_idx, train_schedule in enumerate(train_schedules):
        # 筛选单个列车的有效节点
        station_positions = [int(round(station[0])) for station in train_schedule]
        valid_nodes = filter_valid_nodes(all_nodes, station_positions, train_schedule, train_max_speed)
        
        if select_near_plan:
            print(f"列车 {train_idx}：根据计划时刻表筛选节点...")
            valid_nodes = filter_nodes_near_plan(valid_nodes, train_schedule, station_names, max_distance)
        
        train_valid_nodes.append(valid_nodes)
        
        # 创建图结构
        graph, len_valid_arcs = create_valid_arcs(valid_nodes, train_schedule, a_max)
        print(f"列车 {train_idx}：添加的有效弧数量: {len_valid_arcs}")
        train_graphs.append(graph)
        
        # region 绘制有效节点
        # node_color = f'C{train_idx}'
        # valid_x = [node[0] for node in valid_nodes]
        # valid_y = [node[1] for node in valid_nodes]
        # valid_z = [node[2] for node in valid_nodes]
        # ax.scatter(valid_x, valid_y, valid_z, color=node_color, s=10, alpha=0.3, label=f'列车{train_idx}有效节点')
        # endregion
    
    # 设置图例和标题
    ax.legend()
    ax.set_title('多列车STS网格模型可视化')
    ax.set_xlabel('空间维度 (站点位置/km)')
    ax.set_ylabel('时间维度 (时间/min)')
    ax.set_zlabel('速度维度 (速度/km/min)')
    
    # 识别所有列车的不兼容弧集合
    all_incompatible_arcs_dict = []
    for train_idx, graph in enumerate(train_graphs):
        print(f"计算列车 {train_idx} 的不兼容弧集合...")
        incompatible_arcs_dict = identify_incompatible_arcs(train_valid_nodes[train_idx], graph, headway)
        all_incompatible_arcs_dict.append(incompatible_arcs_dict)
    
    # 拉格朗日乘子迭代求解
    multipliers = {}  # 初始化拉格朗日乘子为0
    step_size = 10.0   # 初始步长
    
    best_paths = None
    best_violations = float('inf')
    
    # 创建每个列车的起点和终点
    train_start_end_nodes = []
    for train_idx, train_schedule in enumerate(train_schedules):
        start_node = (train_schedule[0][0], train_schedule[0][2], 0)
        end_node = (train_schedule[-1][0], train_schedule[-1][2], 0)
        train_start_end_nodes.append((start_node, end_node))
    
    print("开始拉格朗日松弛迭代...")
    
    for iteration in range(max_iterations):
        print(f"\n迭代 {iteration + 1}/{max_iterations}")
        
        # 为每个列车找最短路径
        all_train_paths = []
        all_selected_arcs = []
        
        for train_idx, (valid_nodes, graph) in enumerate(zip(train_valid_nodes, train_graphs)):
            start_node, end_node = train_start_end_nodes[train_idx]
            
            # 使用拉格朗日乘子寻找最短路径
            path, dist, prev, selected_arcs = find_optimal_path_with_lagrangian(
                valid_nodes, graph, start_node, end_node, multipliers, all_incompatible_arcs_dict[train_idx], train_idx)
            
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
        
        # 更新拉格朗日乘子
        new_multipliers, violations = update_multipliers(
            multipliers, all_selected_arcs, all_incompatible_arcs_dict[0], step_size, headway)
        
        # 标记multipliers所包含的节点
        if iteration == max_iterations - 1:  # 只在最后一次迭代时绘制
            print(f"正在标记第 {iteration + 1} 次迭代中的乘子节点...")
            multiplier_nodes_marked = False
            
            # 收集所有乘子节点
            multiplier_nodes = []
            for key, value in new_multipliers.items():
                if value > 0:  # 只标记正值乘子
                    i, j, t = key
                    # 在空间-时间平面上标记
                    multiplier_nodes.append((i, t, 0))  # 速度维度设为0以便在底部平面显示
            
            if multiplier_nodes:
                # 绘制乘子节点
                multiplier_nodes = np.array(multiplier_nodes)
                ax.scatter(
                    multiplier_nodes[:, 0], 
                    multiplier_nodes[:, 1], 
                    multiplier_nodes[:, 2],
                    color='red', s=80, marker='x', alpha=0.7,
                    label=f'迭代{iteration+1}乘子节点({len(multiplier_nodes)}个)'
                )
                multiplier_nodes_marked = True
                print(f"已标记 {len(multiplier_nodes)} 个乘子节点")
            
            if not multiplier_nodes_marked:
                print("没有找到需要标记的乘子节点")
        
        print(f"约束违反数: {violations}")
        print(f"乘子数量: {len(new_multipliers)}")
        
        
        # 保存最佳解
        if violations < best_violations:
            best_violations = violations
            best_paths = all_train_paths
            print(f"找到更好的解，违反数: {violations}")
        
        # 检查终止条件
        if violations == 0:
            print("找到无冲突解，停止迭代")
            break
        
        # 更新乘子和步长
        multipliers = new_multipliers
        if iteration % 10 == 9:  # 每10次迭代调整步长
            step_size *= 0.8
            print(f"调整步长至 {step_size}")
    
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
    # 定义两个列车的时刻表
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
    
    # 使用拉格朗日松弛法求解多列车问题
    # 记录算法开始时间
    start_time = time.time()
    
    # 执行算法
    best_paths = solve_multi_train_with_lagrangian(
        [train_schedule1, train_schedule2, train_schedule3, train_schedule4], 
        station_names,
        delta_d=1,      # 200米
        delta_t=1,      # 0.2分钟 12秒
        speed_levels=5, 
        time_diff_minutes=2*60,   # 调度范围是2个小时
        total_distance=300,    # 300km的线路
        max_distance=10,       # 25个格子
        headway=2,             # 2个时间单位的最小间隔
        max_iterations=2,      # 最多迭代50次
        select_near_plan=True,  # 是否使用计划表完成有效点的筛选
    )
    
    # 计算并输出算法运行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"算法总运行时间: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")