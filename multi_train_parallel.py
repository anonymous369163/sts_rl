"""
使用拉格朗日乘子法求解多列车时刻表的STS网格模型
基于原始STS网格模型，添加列车间耦合约束，并使用拉格朗日松弛法求解
多线程并行版本
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
                                     multipliers, train_idx, headway, ax=None, drawn_arcs=None):
    """
    使用带有拉格朗日乘子的Dijkstra算法寻找最优路径
    
    参数:
        valid_nodes: 有效节点集合
        graph: 图的邻接表表示
        start_node: 起点节点 (位置, 时间, 速度)
        end_node: 终点节点 (位置, 时间, 速度)
        multipliers: 拉格朗日乘子字典，键为(i,j,τ)，τ可以是起点时间t或终点时间s
        train_idx: 当前列车的索引
        headway: 最小间隔时间
        ax: 可选的绘图对象，用于可视化
        drawn_arcs: 已经绘制的弧集合
    
    返回:
        path: 最优路径列表
        dist: 距离字典
        prev: 前驱节点字典
        selected_arcs: 列车所选择的弧
    """
    all_graph_nodes = valid_nodes
    selected_arcs = []  # 记录该列车选择的所有弧
    
    if drawn_arcs is None:
        drawn_arcs = set()

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

                    arc_keys_set = set()
                    # 添加起点和终点时间的键
                    arc_keys_set.add((i_pos, j_pos, t))
                    arc_keys_set.add((i_pos, j_pos, s))
                    
                    # 添加headway范围内的时间点
                    for h in range(1, headway + 1):
                        for time_offset in [-h, h]:
                            arc_keys_set.add((i_pos, j_pos, t + time_offset))
                            arc_keys_set.add((i_pos, j_pos, s + time_offset))

                    for arc_key in arc_keys_set:
                        if arc_key in multipliers:
                            penalty += multipliers[arc_key] 

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
        return None, dist, prev, [], ax, drawn_arcs

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


def process_train_path(train_idx, valid_nodes, graph, start_node, end_node, multipliers, headway, drawn_arcs_shared):
    """
    处理单个列车的最优路径查找，用于多线程执行
    
    参数与find_optimal_path_with_lagrangian相同，但移除了绘图相关参数
    
    返回:
        results: 字典，包含路径查找结果
    """
    try:
        path, dist, prev, selected_arcs, _, drawn_arcs_local = find_optimal_path_with_lagrangian(
            valid_nodes, graph, start_node, end_node, multipliers, train_idx, headway, None, drawn_arcs_shared.copy())
        
        # 更新共享的drawn_arcs集合
        drawn_arcs_shared.update(drawn_arcs_local)
        
        return {
            'train_idx': train_idx,
            'path': path,
            'selected_arcs': selected_arcs,
            'success': path is not None,
            'dist': dist[end_node] if path is not None else float('inf')
        }
    except Exception as e:
        print(f"列车 {train_idx} 处理出错：{str(e)}")
        return {
            'train_idx': train_idx,
            'path': None,
            'selected_arcs': [],
            'success': False,
            'error': str(e)
        }


def update_multipliers(multipliers, train_paths, step_size, headway):
    """
    使用次梯度法更新拉格朗日乘子
    
    参数:
        multipliers: 当前拉格朗日乘子字典，键为(i,j,τ)
        train_paths: 所有列车的路径弧列表
        step_size: 次梯度更新步长
        headway: 最小间隔时间
        
    返回:
        new_multipliers: 更新后的拉格朗日乘子
        total_violations: 约束违反总数
    """
    # 改用浅拷贝以提高效率
    new_multipliers = multipliers.copy()
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
    
    # 检查冲突并更新乘子
    conflict_zones = []
    for space_time_key, trains in space_time_usage.items():
        # 使用集合操作更高效地检测冲突
        unique_trains = set(trains)
        trains_count = len(unique_trains)
        
        # 如果同一时空位置有多列车使用，则存在冲突
        if trains_count > 1:
            violation = trains_count - 1  # 冲突的列车数减1
            total_violations += violation
            
            # 更新该时空位置的乘子
            if space_time_key not in new_multipliers:
                new_multipliers[space_time_key] = 0
            new_multipliers[space_time_key] = max(0, new_multipliers[space_time_key] + step_size * violation)

            i, j, t = space_time_key
            conflict_zones.append((i, t, trains_count))
    
    # 打印调试信息
    print(f"总违反约束数: {total_violations}")  
    print(f"更新的乘子数量: {len(new_multipliers)}") 
    
    return new_multipliers, total_violations, conflict_zones


def solve_multi_train_with_lagrangian_parallel(train_schedules, station_names, delta_d=5, delta_t=5, 
                                     speed_levels=5, time_diff_minutes=5*60, total_distance=50,
                                     max_distance=30, select_near_plan=True, a_max=5,
                                     train_max_speed=5, headway=2, max_iterations=50, 
                                     num_workers=None, debug_mode=False):
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
        num_workers: 并行工作线程数，默认为None(使用系统默认值)
        debug_mode: 是否启用调试模式，包括额外的绘图和输出
        
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
    
    # 创建适合当前屏幕的图形，仅在需要可视化时执行
    if debug_mode:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
        fig = plt.figure(figsize=(10, 8))  
        fig.set_dpi(100)  
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=45)
    else:
        ax = None
    
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
        print(f"列车 {train_idx}：添加的有效弧数量: {len_valid_arcs}")
        
        return train_idx, valid_nodes, graph, len_valid_arcs
    
    # 使用线程池并行处理所有列车
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

    # 仅在调试模式下设置图例和标题
    if debug_mode and ax is not None:
        ax.legend()
        ax.set_title('多列车STS网格模型可视化')
        ax.set_xlabel('空间维度 (站点位置/km)')
        ax.set_ylabel('时间维度 (时间/min)')
        ax.set_zlabel('速度维度 (速度/km/min)')

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
    
    print("开始拉格朗日松弛迭代(并行版)...")
    
    # 初始化共享变量
    drawn_arcs = set()
    multipliers_history = []

    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for iteration in range(max_iterations):
            print(f"\n迭代 {iteration + 1}/{max_iterations}")
            
            # 准备每个列车的任务
            future_to_train = {}
            for train_idx, (valid_nodes, graph) in enumerate(zip(train_valid_nodes, train_graphs)):
                start_node, end_node = train_start_end_nodes[train_idx]
                
                # 提交任务到线程池
                future = executor.submit(
                    process_train_path, 
                    train_idx, valid_nodes, graph, start_node, end_node, multipliers, headway, drawn_arcs.copy()
                )
                future_to_train[future] = train_idx
            
            # 收集所有完成的任务结果
            all_train_paths = []
            all_selected_arcs = []
            all_results = [None] * len(train_schedules)  # 预分配结果列表
            
            # 等待所有任务完成并收集结果
            for future in concurrent.futures.as_completed(future_to_train):
                train_idx = future_to_train[future]
                try:
                    result = future.result()
                    all_results[train_idx] = result  # 按列车索引存储结果
                except Exception as exc:
                    print(f"列车 {train_idx} 处理出错: {exc}")
                    all_results[train_idx] = {
                        'train_idx': train_idx,
                        'path': None,
                        'selected_arcs': [],
                        'success': False
                    }
            
            # 检查是否所有列车都找到了路径
            all_paths_found = True
            for result in all_results:
                if not result['success']:
                    all_paths_found = False
                    break
            
            if not all_paths_found:
                # 如果任一列车无法找到路径，减小步长重试
                step_size *= 0.5
                print(f"减小步长至 {step_size}，继续迭代")
                if step_size < 0.01:
                    print("步长过小，停止迭代")
                    break
                continue
            
            # 整理结果
            for result in all_results:
                all_train_paths.append(result['path'])
                all_selected_arcs.append(result['selected_arcs'])
                # 更新drawn_arcs集合
                for arc in result['selected_arcs']:
                    if any(arc_key in multipliers and multipliers[arc_key] > 0 for arc_key in [
                        (min(arc[0], arc[1]), max(arc[0], arc[1]), arc[2]),  # 起点
                        (min(arc[0], arc[1]), max(arc[0], arc[1]), arc[3])   # 终点
                    ]):
                        drawn_arcs.add(arc)
            
            # 绘制最后一次迭代的惩罚弧
            if iteration == max_iterations - 1 and debug_mode and ax is not None:
                print("可视化带惩罚的弧...") 
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
            
            # 在最后一次迭代的调试模式下，标记冲突区域
            if iteration == max_iterations - 1 and debug_mode and ax is not None and conflict_zones:
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
            if debug_mode:
                multipliers_history.append(copy.deepcopy(new_multipliers))
            else:
                multipliers_history.append(new_multipliers.copy())
    
    end_time = time.time()
    print(f"\n拉格朗日松弛求解完成，耗时 {end_time - start_time:.2f} 秒")
    
    # 绘制最终结果
    if best_paths and debug_mode and ax is not None:
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
            ax_penalty.plot(range(len(sorted_values)), sorted_values, 
                    color=color, alpha=0.7, label=f'迭代 {i+1}')
        
        ax_penalty.set_xlabel('乘子索引 (按值降序排列)')
        ax_penalty.set_ylabel('乘子值')
        ax_penalty.set_title('惩罚分布的迭代变化')
        ax_penalty.legend()
        plt.savefig('penalty_distribution.png')
        
        # 完善图形显示
        if ax is not None:
            finalize_plot(ax, space_segments, time_nodes, speed_levels, delta_d, delta_t)
    
    return best_paths


if __name__ == "__main__":
    # 定义列车的时刻表
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
    
    # 执行多线程并行算法
    best_paths = solve_multi_train_with_lagrangian_parallel(
        [train_schedule1, train_schedule2, train_schedule3, train_schedule4], 
        station_names,
        delta_d=1,            # 1km
        delta_t=1,            # 1分钟
        speed_levels=5, 
        time_diff_minutes=2*60,   # 调度范围是2个小时
        total_distance=300,       # 300km的线路
        max_distance=30,          # 30个格子
        headway=2,                # 2个时间单位的最小间隔
        max_iterations=10,        # 最多迭代10次
        select_near_plan=True,    # 使用计划表完成有效点的筛选
        num_workers=4,            # 4个线程并行处理
        debug_mode=False,         # 关闭调试模式
    )
    
    # 计算并输出算法运行时间
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"算法总运行时间: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")
    
    # 单线程版本比较（可选）
    print("\n开始单线程版本对照测试...")
    from multi_train_main import solve_multi_train_with_lagrangian
    
    start_time_single = time.time()
    _ = solve_multi_train_with_lagrangian(
        [train_schedule1, train_schedule2, train_schedule3, train_schedule4], 
        station_names,
        delta_d=1, delta_t=1, speed_levels=5, 
        time_diff_minutes=2*60, total_distance=300, max_distance=30,
        headway=2, max_iterations=10, select_near_plan=True,
        debug_mode=False,
    )
    end_time_single = time.time()
    execution_time_single = end_time_single - start_time_single
    
    print(f"单线程版本运行时间: {execution_time_single:.2f} 秒")
    print(f"多线程性能提升: {execution_time_single/execution_time:.2f}x") 