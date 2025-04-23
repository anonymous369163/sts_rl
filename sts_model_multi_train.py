"""
本代码用于生成多列车的运行区间STS网格模型，根据给定的多个列车计划时刻表，完成多个列车的最优速度位置曲线的生成
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import copy
import heapq
import math

# 复用单列车模型中的辅助函数
class Arrow3D(FancyArrowPatch):
    """用于在3D空间中绘制箭头的类"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def convert_time_to_units(time_str, delta_t=1):
    """
    将时间字符串(格式为'HH:MM')转换为以delta_t为单位的时间值
    """
    if time_str == '8:00':
        return 0  # 基准时间点
    
    hours, minutes = map(int, time_str.split(':'))
    # 计算与基准时间(8:00)的差值，单位为分钟
    time_diff = (hours - 8) * 60 + minutes
    # 转换为delta_t单位
    return int(time_diff / delta_t)

def convert_time_to_minutes(time_str):
    """
    将时间字符串(格式为'HH:MM')转换为分钟数值 
    """
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def check_end_node_in_graph(graph, stations):
    """检查图中是否包含起始节点和是否有弧通向终点节点"""
    start_node_in_graph = False
    end_node_reachable = False

    # 定义起始节点和终点节点
    start_node = (stations[0][0], stations[0][2], 0)  # (位置, 到达/出发时间, 速度)
    end_node = (stations[-1][0], stations[-1][2], 0)  # (位置, 到达时间, 速度)

    # 检查起始节点是否存在于图中
    if start_node in graph:
        start_node_in_graph = True
        print(f"起始节点 {start_node} 存在于图中。")
    else:
        print(f"警告：起始节点 {start_node} 不在图中或没有出边，无法开始路径搜索。")

    # 检查是否有弧通向终点节点
    for from_node, neighbors in graph.items():
        for to_node, _ in neighbors:
            if to_node == end_node:
                end_node_reachable = True
                print(f"找到通向终点节点 {end_node} 的弧: {from_node} -> {to_node}")
                break
        if end_node_reachable:
            break

    # 综合判断
    if start_node_in_graph and end_node_reachable:
        print("起始节点和终点节点检查通过，可以继续寻找最短路径。")
        return True
    else:
        print("错误：由于起始节点缺失或终点节点不可达，无法保证找到有效路径。")
        return False

def create_multi_train_schedule_sts_grid(train_schedules, station_names_input, delta_d=0.5,
                                       delta_t=20/60, speed_levels=5, time_diff_minutes=5*60+1,
                                       total_distance=50, draw_plan=False, draw_line=False,
                                       max_distance=30, select_near_plan=True, a_max=10,
                                       min_headway=2):
    """
    创建包含多列车时刻表约束的STS网格模型
    
    参数:
        train_schedules: 多个列车时刻表的列表，每个元素是一个列车的时刻表列表
        station_names_input: 车站名称字典，键为位置，值为站名
        delta_d: 空间单位长度，默认为0.5km
        delta_t: 时间单位长度，默认为20/60分钟
        speed_levels: 速度级别数量，默认为5
        time_diff_minutes: 时间窗口长度，默认为5小时1分钟
        total_distance: 线路总长度，默认为50km
        draw_plan: 是否绘制计划时刻表，默认为False
        draw_line: 是否绘制有效弧，默认为False
        max_distance: 节点到计划时刻表直线的最大距离，默认为30
        select_near_plan: 是否选择靠近计划时刻表的节点，默认为True
        a_max: 最大加速度约束，默认为10
        min_headway: 最小列车间隔（时间单位），默认为2
        
    返回:
        多个列车的最优路径和状态信息
    """
    # 检查输入参数
    if not train_schedules:
        print("错误：未提供列车时刻表")
        return None

    # 对站点名称字典进行深拷贝
    station_names = copy.deepcopy(station_names_input)
    
    # 计算时间节点数和空间段数
    time_nodes = int(round(time_diff_minutes / delta_t))
    space_segments = int(round(total_distance / delta_d))
    
    print(f"时间节点数: {time_nodes}, 空间段数: {space_segments}")
    print(f"时间间隔: {delta_t*60:.2f}秒, 空间间隔: {delta_d*1000:.2f}米")
    print(f"总时间: {time_nodes * delta_t:.2f} 分钟, 总距离: {space_segments * delta_d:.2f} km")
    
    # 预处理每个列车的时刻表
    processed_schedules = []
    for train_idx, train_schedule in enumerate(train_schedules):
        stations = copy.deepcopy(train_schedule)
        
        # 转换站点时刻表为单位值，位置为栅格索引
        for i in range(len(stations)):
            arrive_time = stations[i][2]
            depart_time = stations[i][3]
            # 转换到达和出发时间
            stations[i][2] = convert_time_to_units(arrive_time, delta_t)
            stations[i][3] = convert_time_to_units(depart_time, delta_t) 
            stations[i][0] = int(round(stations[i][0] / delta_d))
        
        processed_schedules.append(stations)
    
    # 修改station_names中的键，使其与栅格化后的站点位置对应
    new_station_names = {}
    for pos, name in station_names.items():
        # 栅格化站点位置
        new_pos = int(round(pos / delta_d))
        new_station_names[new_pos] = name 
    station_names = new_station_names
    
    # 列车信息
    train_max_speed = 20  # 最高速度级别
    
    # 创建图形
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标范围
    space_range = np.arange(0, space_segments + 1)
    time_range = np.arange(0, time_nodes+1)
    speed_range = np.arange(0, speed_levels+1)
    
    # 存储多列车的最优路径结果
    train_paths = []
    train_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    
    # 标记车站位置和名称（只需执行一次）
    print("正在标记车站信息...")
    # 获取所有唯一的站点位置
    all_station_positions = set()
    for schedule in processed_schedules:
        for station in schedule:
            all_station_positions.add(station[0])
    
    # 为每个车站添加标记和标签
    for position in all_station_positions:
        # 确定站点类型（从第一个包含该站点的列车时刻表中获取）
        station_type = None
        for schedule in processed_schedules:
            for station in schedule:
                if station[0] == position:
                    station_type = station[1]
                    break
            if station_type is not None:
                break
        
        # 获取车站名称
        station_name = station_names.get(position, f"站点{position}")
        
        # 根据站点类型选择不同的标记样式
        if station_type == 0:  # 始发/终点站
            marker_color = 'red'
            marker_style = '^'
            marker_size = 150
        elif station_type == 2:  # 停靠站
            marker_color = 'purple'
            marker_style = 's'
            marker_size = 120
        else:  # 通过站
            marker_color = 'green'
            marker_style = 'o'
            marker_size = 100
        
        # 在空间轴上标记车站位置（在z=0平面上）
        ax.scatter(position, 0, 0, color=marker_color, s=marker_size, marker=marker_style)
        
        # 添加具体车站名称标签
        ax.text(position, -1, -0.5, station_name, color=marker_color, 
                fontsize=12, ha='center', va='top', weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=marker_color, boxstyle='round,pad=0.3'))
    
    # 添加车站类型说明到图例
    ax.scatter([], [], color='red', s=100, marker='^', label='始发/终点站')
    ax.scatter([], [], color='purple', s=100, marker='s', label='停靠站')
    ax.scatter([], [], color='green', s=100, marker='o', label='通过站')
    
    # 设置图形标题和轴标签
    ax.set_title('多列车STS网格模型可视化')
    ax.set_xlabel('空间维度 (站点位置/km)')
    ax.set_ylabel('时间维度 (时间/min)')
    ax.set_zlabel('速度维度 (速度/km/min)')
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 调整视角以获得更好的3D效果
    ax.view_init(elev=30, azim=45)
    
    # 每列车使用单独的路径搜索逻辑，并追踪已占用的时空节点
    occupied_space_time = set()  # 存储已占用的时空节点 (位置, 时间)
    
    # 主循环，处理每一列列车
    for train_idx, stations in enumerate(processed_schedules):
        train_color = train_colors[train_idx % len(train_colors)]
        print(f"\n正在处理第 {train_idx+1} 列列车...")
        
        # 创建STS网格模型
        print("正在创建STS网格模型...")
        
        # 创建节点集合
        all_nodes = set()
        for i in space_range:  # 空间维度
            for t in time_range:  # 时间维度
                for v in speed_range:  # 速度维度
                    all_nodes.add((i, t, v))
        
        print(f"初始网格节点总数: {len(all_nodes)}")
        
        # 移除不合理的节点
        valid_nodes_set = set()
        for node in all_nodes:
            i, t, v = node
            
            # 检查节点是否在站点上
            is_at_station = False
            station_info = None
            for station in stations:
                if i == station[0]:
                    is_at_station = True
                    station_info = station
                    break
            
            # 检查该时空点是否已被其他列车占用
            if (i, t) in occupied_space_time and not is_at_station:
                continue  # 跳过已占用的非站点节点
            
            # 如果节点在站点上，应用站点特定规则
            if is_at_station:
                station_type = station_info[1]
                arrive_time = station_info[2]
                depart_time = station_info[3]
                speed_limit = min(station_info[4], train_max_speed)
                
                # 规则1: 通过站不允许速度为0
                if station_type == 1 and v == 0:
                    continue
                    
                # 规则2.2: 停靠站在停靠期间速度必须为0
                if station_type == 2 and v != 0:
                    continue
                    
                # 规则3: 速度不能超过站点限制
                if v > speed_limit:
                    continue
                # 规则6: 始发站和终点站的速度必须为0
                if station_type == 0 and v != 0:
                    continue
            else:
                # 非站点位置的规则
                # 规则4: 非站点位置不允许速度为0
                if v == 0:
                    continue
                
                # 规则5: 速度不能超过全局最大速度
                if v > train_max_speed:
                    continue
            
            # 通过所有规则检查，添加到有效节点集合
            valid_nodes_set.add(node)
        
        # 将有效节点集合转换为列表  
        valid_nodes = list(valid_nodes_set)
        
        # 为不同类型的节点添加垂直线标记（如果第一次绘制）
        for node in valid_nodes:
            i, t, v = node
            # 检查节点是否在站点上
            is_at_station = False
            station_info = None
            for station in stations:
                if i == station[0]:
                    is_at_station = True
                    station_info = station
                    break
            
            # 特殊标记站点上的节点
            if is_at_station:
                station_type = station_info[1]
                
                # 始发/终点站的节点
                if station_type == 0 and v == 0:
                    # 在站点位置绘制垂直线，只画一次
                    if not hasattr(ax, f'vertical_line_drawn_{i}'):
                        ax.plot([i, i], [0, time_nodes], [0, 0], color='red', linestyle='--', alpha=0.5)
                        setattr(ax, f'vertical_line_drawn_{i}', True)
                
                # 停靠站的节点
                elif station_type == 2 and v == 0:
                    # 在站点位置绘制垂直线，只画一次
                    if not hasattr(ax, f'vertical_line_drawn_{i}'):
                        ax.plot([i, i], [0, time_nodes], [0, 0], color='purple', linestyle='--', alpha=0.5)
                        setattr(ax, f'vertical_line_drawn_{i}', True)
        
        # 根据计划时刻表筛选出离计划时刻表较近的节点
        if select_near_plan:
            print("正在计算计划时刻表的直线方程...")
            
            # 提取站点位置和时间信息
            station_positions = []
            station_times = []
            
            for station in stations:
                position = station[0]
                arrive_time = station[2]
                depart_time = station[3]
                
                # 添加到达时间点
                station_positions.append(position)
                station_times.append(arrive_time)
                
                # 如果到达时间和出发时间不同，也添加出发时间点
                if arrive_time != depart_time:
                    station_positions.append(position)
                    station_times.append(depart_time)
            
            # 计算直线方程参数
            if len(station_positions) > 1:
                # 按照站点对划分线段，每两个相邻站点之间绘制一条直线
                close_nodes = set()
                print("计算站点间的直线方程：")
                
                # 获取唯一的站点位置
                unique_station_positions = []
                for pos in station_positions:
                    if pos not in unique_station_positions:
                        unique_station_positions.append(pos)
                unique_station_positions.sort()
                
                # 为每对相邻站点计算直线方程
                for i in range(len(unique_station_positions) - 1):
                    start_pos = unique_station_positions[i]
                    end_pos = unique_station_positions[i+1]
                    
                    # 找出这两个站点对应的时间点
                    start_times = []
                    end_times = []
                    
                    for j, pos in enumerate(station_positions):
                        if pos == start_pos:
                            start_times.append(station_times[j])
                        elif pos == end_pos:
                            end_times.append(station_times[j])
                    
                    # 使用最早的到达时间和最晚的出发时间
                    start_time = min(start_times)
                    end_time = max(end_times)
                    
                    # 计算直线方程: t = m*s + c
                    if end_pos > start_pos:  # 防止除以零
                        m = (end_time - start_time) / (end_pos - start_pos)
                        c = start_time - m * start_pos
                        
                        print(f"站点 {station_names.get(start_pos, start_pos)} 到 {station_names.get(end_pos, end_pos)} 的直线方程: t = {m:.4f} * s + {c:.4f}")
                        
                        # 绘制这段直线
                        s_range = np.array([start_pos, end_pos])
                        t_range = m * s_range + c
                        ax.plot(s_range, t_range, [0, 0], color=train_color, linestyle='--', linewidth=2, 
                            label=f"列车{train_idx+1}：{station_names.get(start_pos, start_pos)}-{station_names.get(end_pos, end_pos)}" if i == 0 else "_nolegend_")
                        
                        # 筛选离这条直线较近的节点
                        for node in valid_nodes:
                            i_pos, t, v = node
                            # 只考虑在当前站点区间内的节点
                            if start_pos <= i_pos <= end_pos:
                                # 计算节点到直线的距离 (在空间-时间平面上)
                                distance = abs(-m * i_pos + t - c) / np.sqrt(m**2 + 1)
                                
                                if distance <= max_distance:
                                    close_nodes.add(node)
                
                print(f"找到离计划时刻表较近的节点: {len(close_nodes)} 个")
                
                # 更新有效节点集合为离直线较近的节点
                valid_nodes = list(close_nodes)
                
            else:
                print("警告：站点数量不足，无法计算直线方程")
        
        # 存储所有有效弧
        valid_arcs = []
        
        # 按时间步骤组织节点
        nodes_by_time = {}
        for node in valid_nodes:
            i, t, v = node
            if t not in nodes_by_time:
                nodes_by_time[t] = []
            nodes_by_time[t].append(node)
        
        # 遍历每个时间步的节点，连接到下一时间步的有效节点
        for t in sorted(nodes_by_time.keys()):
            if t + 1 not in nodes_by_time:
                continue  # 没有下一时间步的节点
                
            for node_from in nodes_by_time[t]:
                i, _, v = node_from  # 当前节点的位置、时间和速度
                
                for node_to in nodes_by_time[t + 1]:
                    j, _, u = node_to  # 下一节点的位置、时间和速度
    
                    # 检查节点是否在站点上
                    is_at_station = False
                    for station in stations:
                        if i == station[0] or j == station[0]:
                            is_at_station = True
                            break
                            
                    # 检查是否有中间停靠站
                    has_intermediate_stop = False
                    for station in stations:
                        station_pos = station[0]
                        station_type = station[1]
                        # 检查是否是停靠站且位于当前节点和下一节点之间
                        if station_type == 2 and min(i, j) < station_pos < max(i, j):
                            has_intermediate_stop = True
                            break
                    
                    # 如果有中间停靠站，则不添加这条弧
                    if has_intermediate_stop:
                        continue
                    
                    # 检查区段冲突 - 确保当前弧不与已占用的区段冲突
                    segment_conflict = False 
                    # 检查列车运行区段冲突
                    # 遍历当前弧所覆盖的所有空间位置，如果这些位置在当前时间t或下一时间步t+1已被其他列车占用
                    # 则标记为区段冲突，不允许列车通过，这种机制可以防止列车在同一区段同时运行，从而避免越行冲突 
                    for pos in range(min(i, j), max(i, j) + 1):
                        if (pos, t) in occupied_space_time or (pos, t+1) in occupied_space_time:
                            segment_conflict = True
                            break
                    
                    # 如果存在区段冲突，不添加这条弧
                    if segment_conflict and not is_at_station:
                        continue
                    
                    # 计算位移
                    d = j - i
                    
                    # 检查速度和位置的耦合关系（匀加速/匀减速）
                    expected_end_speed = 2 * (d / 1) - v
                    
                    # 检查加速度约束
                    acceleration = (u - v) / 1
                    
                    # 如果满足条件，则添加弧
                    if abs(u - expected_end_speed) < 0.001 and abs(acceleration) <= a_max:
                        valid_arcs.append((i, j, t, t+1, v, u))
                        
                        if draw_line:
                            # 确定弧的颜色
                            if i == j:  # 停留弧
                                arc_color = 'blue'
                            elif u > v:  # 加速弧
                                arc_color = 'red'
                            elif u < v:  # 减速弧
                                arc_color = 'orange'
                            else:  # 匀速弧
                                arc_color = 'green'
                            
                            # 添加弧到图中
                            arrow = Arrow3D([i, j], [t, t+1], [v, u], 
                                        mutation_scale=15, lw=1.5, arrowstyle='-|>', color=arc_color)
                            ax.add_artist(arrow)
        
        print(f"添加的有效弧数量: {len(valid_arcs)}")
    
        # 在时间-空间二维平面上绘制计划方案
        if draw_plan:
            z_min = 0
            
            # 直接根据站点计划时间绘制列车运行线
            train_space_coords = []
            train_time_coords = []
            
            # 添加始发站出发时间点
            train_space_coords.append(stations[0][0])
            train_time_coords.append(stations[0][3])  # 出发时间
            
            # 添加中间停靠站的到达和出发时间点
            for station in stations[1:-1]:
                if station[1] == 2:  # 停靠站
                    # 到达时间点
                    train_space_coords.append(station[0])
                    train_time_coords.append(station[2])
                    # 出发时间点
                    train_space_coords.append(station[0])
                    train_time_coords.append(station[3])
                elif station[1] == 1:  # 通过站
                    train_space_coords.append(station[0])
                    train_time_coords.append(station[2])  # 通过时间
            
            # 添加终点站到达时间点
            train_space_coords.append(stations[-1][0])
            train_time_coords.append(stations[-1][2])  # 到达时间
            
            # 绘制列车运行线
            ax.plot(train_space_coords, train_time_coords, [z_min] * len(train_space_coords), 
                    color=train_color, alpha=0.9, linewidth=3, label=f'列车{train_idx+1}计划运行线')
            
            # 标记站点的计划时间
            for station in stations:
                station_pos = station[0]
                station_type = station[1]
                arrive_time = station[2]
                depart_time = station[3]
                
                # 对于始发站，只标记出发时间
                if station_type == 0 and station_pos == stations[0][0]:
                    ax.scatter(station_pos, depart_time, z_min, color=train_color, s=80, marker='o')
                
                # 对于终点站，只标记到达时间
                elif station_type == 0 and station_pos == stations[-1][0]:
                    ax.scatter(station_pos, arrive_time, z_min, color=train_color, s=80, marker='o')
                
                # 对于停靠站，标记到达和出发时间
                elif station_type == 2:
                    ax.scatter(station_pos, arrive_time, z_min, color=train_color, s=80, marker='o')
                    ax.scatter(station_pos, depart_time, z_min, color=train_color, s=80, marker='o')
                
                # 对于通过站，标记通过时间
                elif station_type == 1:
                    ax.scatter(station_pos, arrive_time, z_min, color=train_color, s=80, marker='o')
        
        # 使用动态规划方法构建最优路径
        print("正在使用动态规划方法构建最优路径...")
        
        # 找出起点和终点节点
        start_node = None
        end_node = None
        
        for node in valid_nodes:
            i, t, v = node
            # 检查是否为起点
            if i == stations[0][0] and t == stations[0][3] and v == 0:
                start_node = node
            # 检查是否为终点
            elif i == stations[-1][0] and t == stations[-1][2] and v == 0:
                end_node = node
        
        if not start_node or not end_node:
            print(f"错误：无法找到列车{train_idx+1}的起点或终点节点")
            continue
        
        print(f"起点节点: {start_node}")
        print(f"终点节点: {end_node}")
        
        # 构建邻接表表示图
        graph = {}
        for arc in valid_arcs:
            i, j, t1, t2, v1, v2 = arc
            from_node = (i, t1, v1)
            to_node = (j, t2, v2)
            
            if from_node not in graph:
                graph[from_node] = []
            
            # 计算弧的代价（考虑能耗最小化和与时刻表的一致性）
            cost = 1.0  # 基础代价
            
            # 计算加速度和能耗
            if t2 > t1:
                acc = max((v2 - v1) / (t2 - t1), 0)
                
                # 特殊情况处理
                if i == j:  # 停留弧
                    energy = 0
                elif v2 < v1:  # 减速弧
                    energy = 0
                else:  # 加速或匀速
                    m = 1.0  # 列车质量
                    energy = m * acc * (v1 + v2) / 2 * (t2 - t1)
                    
                cost = energy
            
            # 考虑与时刻表的一致性
            for station in stations:
                station_pos, station_type, arrive_time, depart_time, _ = station
                
                # 计算与计划时间的偏差
                if i == station_pos:
                    time_diff = abs(t1 - depart_time)
                    cost += time_diff * 10.0
                elif j == station_pos:
                    time_diff = abs(t2 - arrive_time)
                    cost += time_diff * 10.0
            
            graph[from_node].append((to_node, cost))
        
        # 使用Dijkstra算法找最短路径
        if not check_end_node_in_graph(graph, stations):
            print(f"列车{train_idx+1}无法找到路径，跳过")
            continue
        
        # 初始化距离和前驱节点字典
        all_graph_nodes = set(graph.keys())
        for neighbors in graph.values():
            for neighbor, _ in neighbors:
                all_graph_nodes.add(neighbor)
        if start_node: all_graph_nodes.add(start_node)
        if end_node: all_graph_nodes.add(end_node)
    
        dist = {node: math.inf for node in all_graph_nodes}
        prev = {node: None for node in all_graph_nodes}
    
        dist[start_node] = 0
        pq = [(0, start_node)]
        visited_nodes = set()
    
        print(f"开始为列车{train_idx+1}寻找最短路径...")
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if d > dist[u]:
                continue
                
            if u in visited_nodes:
                continue
            visited_nodes.add(u)
    
            if u == end_node:
                print(f"已找到列车{train_idx+1}终点节点，最短距离为 {dist[u]:.2f}")
                break
                
            if u in graph:
                for v, cost in graph[u]:
                    if v in dist:
                        alt = dist[u] + cost
                        if alt < dist[v]:
                            dist[v] = alt
                            prev[v] = u
                            heapq.heappush(pq, (alt, v))
    
        # 检查是否找到了路径
        if dist.get(end_node, math.inf) == math.inf:
            print(f"错误：列车{train_idx+1}无法找到路径")
            continue
    
        # 重建最短路径
        path = []
        current = end_node
        
        while current:
            path.append(current)
            current = prev[current]
        
        path.reverse()  # 反转路径，从起点到终点
        
        if path[0] != start_node or path[-1] != end_node:
            print(f"错误：列车{train_idx+1}无法找到有效路径")
            continue
        
        print(f"列车{train_idx+1}最优路径已找到！路径长度: {len(path)} 节点")
        
        # 将路径添加到结果列表
        train_paths.append(path)
        
        # 标记被占用的时空节点
        for node in path:
            i, t, v = node
            occupied_space_time.add((i, t))
        
        # 打印时刻表
        print(f"\n列车{train_idx+1}最优路径时刻表:")
        print("站点\t计划到达\t计划出发\t实际到达\t实际出发\t最高速度")
        
        # 记录每个站点的实际到达和出发时间
        station_actual_times = {}
        
        for node in path:
            i, t, v = node
            # 检查是否在站点上
            for station in stations:
                if i == station[0]:
                    if i not in station_actual_times:
                        station_actual_times[i] = {"arrive": t, "depart": t, "max_speed": v}
                    else:
                        # 更新出发时间和最高速度
                        station_actual_times[i]["depart"] = t
                        station_actual_times[i]["max_speed"] = max(station_actual_times[i]["max_speed"], v)
        
        # 打印对比时刻表
        for station in stations:
            station_pos, _, plan_arrive, plan_depart, _ = station
            actual_arrive = station_actual_times.get(station_pos, {}).get("arrive", "-")
            actual_depart = station_actual_times.get(station_pos, {}).get("depart", "-")
            max_speed = station_actual_times.get(station_pos, {}).get("max_speed", 0)
            
            print(f"{station_names[station_pos]}\t{plan_arrive}\t{plan_depart}\t{actual_arrive}\t{actual_depart}\t{max_speed}")
        
        # 绘制最优路径轨迹
        print(f"正在绘制列车{train_idx+1}最优路径轨迹...")
        
        # 提取路径中的坐标
        path_space = [node[0] for node in path]
        path_time = [node[1] for node in path]
        path_speed = [node[2] for node in path]
        
        # 绘制路径线
        ax.plot(path_space, path_time, path_speed, color=train_color, linewidth=3, 
                linestyle='-', marker='o', markersize=5, label=f'列车{train_idx+1}最优路径')
        
        # 添加路径起点和终点的特殊标记
        ax.scatter(path_space[0], path_time[0], path_speed[0], 
                  color='green', s=150, marker='*', edgecolors='black')
        ax.scatter(path_space[-1], path_time[-1], path_speed[-1], 
                  color='red', s=150, marker='*', edgecolors='black')
        
        # 投影到空间-时间平面
        ax.plot(path_space, path_time, np.zeros_like(path_speed), 
                color=train_color, linewidth=2, linestyle='--', alpha=0.5)
        
        # 投影到空间-速度平面
        ax.plot(path_space, np.zeros_like(path_time), path_speed, 
                color=train_color, linewidth=2, linestyle='-.', alpha=0.5)
        
        # 投影到时间-速度平面
        ax.plot(np.zeros_like(path_space), path_time, path_speed, 
                color=train_color, linewidth=2, linestyle=':', alpha=0.5)

    # 在这里添加最终的可视化和展示代码
    # 修改坐标轴，使用实际的时间和位置值
    # 设置空间轴的刻度
    space_ticks = np.linspace(0, space_segments, num=6)
    space_tick_labels = [f"{x*delta_d:.1f}" for x in space_ticks]
    ax.set_xticks(space_ticks)
    ax.set_xticklabels(space_tick_labels)
    
    # 设置时间轴的刻度
    time_ticks = np.linspace(0, time_nodes, num=6)
    # 将时间单位转换回实际时间
    base_time = 8 * 60  # 基准时间为8:00（分钟表示）
    time_tick_labels = []
    for t in time_ticks:
        minutes = base_time + t * delta_t  # 转换为分钟
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        time_tick_labels.append(f"{hours:02d}:{mins:02d}")
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_tick_labels)
    
    # 速度轴的刻度
    speed_ticks = np.linspace(0, speed_levels, num=speed_levels+1)
    # 转换速度单位为实际速度 km/h
    speed_tick_labels = [f"{v*3.6*delta_d/delta_t:.0f}" for v in speed_ticks]
    ax.set_zticks(speed_ticks)
    ax.set_zticklabels(speed_tick_labels)
    
    # 更新坐标轴标签
    ax.set_xlabel('空间维度 (km)')
    ax.set_ylabel('时间')
    ax.set_zlabel('速度 (km/h)')
    
    # 添加列车间最小间隔的说明
    ax.text(0, time_nodes-5, 0, f"最小列车间隔: {min_headway*delta_t*60:.0f}秒", 
            color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # 添加总结信息
    if train_paths:
        ax.text(0, time_nodes-10, 0, f"总计生成了{len(train_paths)}个列车运行轨迹", 
                color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # 提取并返回最终结果
    results = []
    for train_idx, path in enumerate(train_paths):
        # 提取时刻表信息
        station_times = {}
        for node in path:
            i, t, v = node
            # 对于每个站点，记录通过时间
            for station in processed_schedules[train_idx]:
                if i == station[0]:
                    if i not in station_times:
                        station_times[i] = {"arrive": t, "depart": t}
                    else:
                        station_times[i]["depart"] = t
        
        # 按照合适的格式返回
        train_result = {
            "train_id": train_idx + 1,
            "path": path,
            "station_times": station_times
        }
        results.append(train_result)
    
    return results

if __name__ == "__main__":
    # 定义多个列车的时刻表
    train_schedules = [
        # 第一列列车（最早发车）
        [
            [0, 0, '8:00', '8:00', 10],     # 北京南站(始发站)，格式: [位置, 类型, 到达时间, 出发时间, 最高速度]
            [70, 2, '8:20', '8:25', 10],    # 廊坊站(停靠站)
            [140, 1, '9:00', '9:00', 10],   # 天津站(通过站)
            [200, 0, '9:30', '9:30', 10],   # 滨海站(终点站)
        ],
        # 第二列列车（晚20分钟发车）
        [
            [0, 0, '8:20', '8:20', 10],     # 北京南站(始发站)
            [70, 2, '8:40', '8:45', 10],    # 廊坊站(停靠站)
            [140, 1, '9:20', '9:20', 10],   # 天津站(通过站)
            [200, 0, '9:50', '9:50', 10],   # 滨海站(终点站)
        ],
        # 第三列列车（晚40分钟发车）
        [
            [0, 0, '8:40', '8:40', 10],     # 北京南站(始发站)
            [70, 2, '9:00', '9:05', 10],    # 廊坊站(停靠站)
            [140, 1, '9:40', '9:40', 10],   # 天津站(通过站)
            [200, 0, '10:10', '10:10', 10], # 滨海站(终点站)
        ],
        # 第四列列车（复兴号，不停廊坊站）
        [
            [0, 0, '8:10', '8:10', 15],     # 北京南站(始发站)，速度更快
            [70, 1, '8:30', '8:30', 15],    # 廊坊站(通过站)
            [140, 1, '9:10', '9:10', 15],   # 天津站(通过站)
            [200, 0, '9:40', '9:40', 15],   # 滨海站(终点站)
        ]
    ]
    
    # 站点名称字典
    station_names = {
        0: "北京南",
        70: "廊坊",
        140: "天津",
        200: "滨海"
    }
    
    # 调用多列车运行轨迹生成函数
    results = create_multi_train_schedule_sts_grid(
        train_schedules,
        station_names, 
        delta_d=5,                # 空间间隔（km）
        delta_t=5,                # 时间间隔（分钟）
        speed_levels=5,           # 速度级别数量
        time_diff_minutes=4*60+1, # 时间窗口（分钟）
        total_distance=300,       # 线路总长度（km）
        draw_plan=True,           # 是否绘制计划时刻表
        draw_line=False,          # 是否绘制弧线（通常设为False以减少图表复杂度）
        max_distance=10,          # 最大距离约束
        min_headway=2             # 最小列车间隔（时间单位）
    )
    
    # 打印最终结果摘要
    if results:
        print("\n============ 多列车运行轨迹生成结果摘要 ============")
        for train_result in results:
            train_id = train_result["train_id"]
            path_length = len(train_result["path"])
            station_count = len(train_result["station_times"])
            
            print(f"列车 {train_id}: 路径长度 {path_length} 节点, 经过 {station_count} 个站点")
        print("===================================================")
    else:
        print("没有成功生成列车运行轨迹。") 