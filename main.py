"""
本代码是用来生成一个运行区间的列车时刻表的STS网格模型，根据给定的计划时刻表，完成最优列车速度位置曲线的确定
"""


import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import math
import heapq
from utils import plot_sts_grid_nodes, draw_plan_on_sts_grid, finalize_plot, plot_optimal_path, export_train_schedule


def check_end_node_in_graph(graph, stations):
    # 检查图中是否包含起始节点和是否有弧通向终点节点
    start_node_in_graph = False
    end_node_reachable = False

    # 定义起始节点和终点节点
    start_node = (stations[0][0], stations[0][2], 0)  # (位置, 到达/出发时间, 速度)
    end_node = (stations[-1][0], stations[-1][2], 0) # (位置, 到达时间, 速度)

    # 1. 检查起始节点是否存在于图中（即是否有从起始节点出发的弧）
    if start_node in graph:
        start_node_in_graph = True
        print(f"起始节点 {start_node} 存在于图中。")
    else:
        print(f"警告：起始节点 {start_node} 不在图中或没有出边，无法开始路径搜索。")

    # 2. 检查是否有弧通向终点节点
    for from_node, neighbors in graph.items():
        for to_node, _ in neighbors:
            if to_node == end_node:
                end_node_reachable = True
                print(f"找到通向终点节点 {end_node} 的弧: {from_node} -> {to_node}")
                break
        if end_node_reachable:
            break

    if not end_node_reachable:
        print(f"警告：图中没有通向终点节点 {end_node} 的弧，可能无法找到有效路径。")
    else:
        print(f"图中包含通向终点节点 {end_node} 的弧。")

    # 综合判断
    if start_node_in_graph and end_node_reachable:
        print("起始节点和终点节点检查通过，可以继续寻找最短路径。")
    else:
        print("错误：由于起始节点缺失或终点节点不可达，无法保证找到有效路径。")
        # 根据需要，这里可以决定是否提前退出或采取其他措施
        # return False # 例如，如果这是一个检查函数，可以返回布尔值
        pass # 当前保持打印警告/错误信息
    

def convert_time_to_units(time_str, delta_t=1):
    """
    将时间字符串(格式为'HH:MM')转换为以delta_t为单位的时间值
    
    参数:
        time_str: 时间字符串，格式为'HH:MM'
        delta_t: 时间单位，默认为1分钟
        
    返回:
        转换后的时间单位值
    """
    if time_str == '8:00':
        return 0  # 基准时间点
    
    hours, minutes = map(int, time_str.split(':'))
    # 计算与基准时间(8:00)的差值，单位为分钟
    time_diff = (hours - 8) * 60 + minutes
    # 转换为delta_t单位
    return int(time_diff / delta_t)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def convert_time_to_minutes(time_str):
    """
    将时间字符串(格式为'HH:MM')转换为分钟数值 
    """
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def check_time_nodes_and_space_segments(time_nodes, space_segments, delta_t, delta_d, time_diff_minutes, total_distance):
    # 矫正机制：确保time_nodes小于space_segments
    if time_nodes >= space_segments:
        # 方案1：调整时间划分，使其更粗略，但不超过10s
        max_delta_t = 10/60  # 最小时间间隔为10s
        new_delta_t = min(time_diff_minutes / (space_segments - 1), max_delta_t)
        
        # 如果调整后的时间间隔仍然不能满足条件，则调整空间划分
        if new_delta_t >= max_delta_t and time_diff_minutes / max_delta_t >= space_segments:
            # 方案2：调整空间划分，使其更细致
            new_delta_d = total_distance / (time_diff_minutes / max_delta_t + 1)
            delta_d = new_delta_d
            delta_t = max_delta_t
        else:
            delta_t = new_delta_t
    
        # 重新计算最终的节点数和段数
        time_nodes = math.ceil(time_diff_minutes / delta_t) # 使用传入的时间差分钟数
        space_segments = math.ceil(total_distance / delta_d)

    print(f"时间节点数: {time_nodes}, 空间段数: {space_segments}")
    print(f"时间间隔: {delta_t*60:.2f}秒, 空间间隔: {delta_d*1000:.2f}米")
    print(f"总时间: {time_nodes * delta_t:.2f} 分钟, 总距离: {space_segments * delta_d:.2f} km")
    return time_nodes, space_segments, delta_t, delta_d


def trans_df_grid(stations_df, station_names, delta_t, delta_d):
    # 支持输入单个时刻表或多个时刻表列表
    if not isinstance(stations_df[0][0], list):  # 如果不是列表的列表，则为单个时刻表
        stations_df = [stations_df]  # 转换为列表的列表格式
    
    # 处理每个时刻表
    for schedule_idx in range(len(stations_df)):
        # 将站点时刻表转换为单位值，位置为栅格索引
        for i in range(len(stations_df[schedule_idx])):
            arrive_time = stations_df[schedule_idx][i][2]
            depart_time = stations_df[schedule_idx][i][3]
            # 转换到达和出发时间
            stations_df[schedule_idx][i][2] = convert_time_to_units(arrive_time, delta_t)
            stations_df[schedule_idx][i][3] = convert_time_to_units(depart_time, delta_t) 
            stations_df[schedule_idx][i][0] = int(round(stations_df[schedule_idx][i][0] / delta_d))

    # 修改station_names中的键，使其与栅格化后的站点位置对应
    new_station_names = {}
    for pos, name in station_names.items():
        # 栅格化站点位置
        new_pos = int(round(pos / delta_d))
        new_station_names[new_pos] = name 
    station_names = new_station_names 
    
    # 如果原始输入是单个时刻表，则返回单个时刻表
    if len(stations_df) == 1:
        stations_df = stations_df[0]
        
    return stations_df, station_names


def find_optimal_path_with_dp(valid_nodes, graph, start_node, end_node, ax=None):
        """
        使用动态规划（Dijkstra算法）寻找最优路径
        
        参数:
            valid_nodes: 有效节点集合
            graph: 图的邻接表表示
            start_node: 起点节点 (位置, 时间, 速度)
            end_node: 终点节点 (位置, 时间, 速度)
            ax: 可选的绘图对象，用于可视化不可达节点
            
        返回:
            path: 最优路径列表
            dist: 距离字典
            prev: 前驱节点字典
        """
        all_graph_nodes = valid_nodes 

        dist = {node: math.inf for node in all_graph_nodes}   # 记录每个节点的最短距离
        prev = {node: None for node in all_graph_nodes}       # 记录每个节点的前驱节点
        
        dist[start_node] = 0
        
        # 使用优先队列（最小堆）存储待访问节点 (距离, 节点)
        pq = [(0, start_node)]
        
        visited_nodes = set() # 用于优化，避免重复处理已确定最短路径的节点

        print("开始使用Dijkstra算法寻找最短路径...")
        
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
                print(f"已找到终点节点 {end_node}，最短距离（代价）为 {dist[u]:.2f}")
                break
                
            # 遍历当前节点的邻居
            if u in graph: # 确保当前节点有出边
                for v, cost in graph[u]:
                    # 检查邻居节点是否有效
                    if v in dist:
                        # 计算通过当前节点 u 到达邻居 v 的新距离
                        alt = dist[u] + cost
                        # 如果找到了更短的路径
                        if alt < dist[v]:
                            dist[v] = alt
                            prev[v] = u
                            # 将邻居节点加入优先队列
                            heapq.heappush(pq, (alt, v)) 

        # 检查是否找到了到达终点的路径
        if dist.get(end_node, math.inf) == math.inf:
            print(f"错误：无法从起点 {start_node} 找到到达终点 {end_node} 的路径。")
            # 如果提供了绘图对象，标记可达节点
            if ax is not None:
                # 标记 dist 中最短路径不为无穷大的节点 (即可达节点) 
                print("终点不可达，正在标记从起点可达的节点...")
                reachable_nodes_labeled = False # 确保图例标签只添加一次
                for node, distance in dist.items():
                    # 检查距离是否为有限值 (不等于无穷大)，表示节点可达
                    if distance != math.inf:
                        # 检查节点是否是有效的元组 (位置, 时间, 速度)
                        if isinstance(node, tuple) and len(node) == 3:
                            i_node, t_node, v_node = node
                            # 在3D图上用特定标记可视化可达节点
                            label_to_add = ""
                            if not reachable_nodes_labeled:
                                label_to_add = '可达节点' # "Reachable Node"
                                reachable_nodes_labeled = True
                            # 使用不同颜色/标记与原始有效节点区分，例如橙色五角星
                            ax.scatter(i_node, t_node, v_node, color='red', s=60, marker='p', alpha=0.8, label=label_to_add)

                # 如果没有可达节点被标记（除了起点本身），也打印一条信息
                if not reachable_nodes_labeled and start_node in dist and dist[start_node] == 0:
                     print("除了起点外，没有其他可达节点。")
                plt.title("STS网格模型（未找到路径）")
                plt.show()
            return None, dist, prev

        # 重建最短路径
        path = []
        current = end_node
        
        while current:
            path.append(current)
            current = prev[current]
        
        path.reverse()  # 反转路径，从起点到终点
        
        return path, dist, prev


def filter_valid_nodes(all_nodes, station_positions, stations_df, train_max_speed):
    """
    筛选STS网格中的有效节点，根据列车时刻表内通过站、停靠站、始发站和终点站的速度限制
    后续还要考虑根据时刻表进一步筛选出离计划比较近的节点，这里主要是筛选车站内的节点
    
    参数:
        all_nodes: 所有网格节点的集合
        station_positions: 站点位置列表
        stations_df: 站点信息列表，每个元素包含[位置,类型,到达时间,出发时间,速度限制]
        train_max_speed: 列车最大速度
        
    返回:
        valid_nodes: 筛选后的有效节点列表
    """
    valid_nodes = []
    for node in all_nodes:
        i, t, v = node
        
        # 检查节点是否在站点上 
        if i in station_positions:
            is_at_station = True
            station_info = stations_df[station_positions.index(i)]
        else:
            is_at_station = False
            station_info = None
        
        # 如果节点在站点上，应用站点特定规则
        if is_at_station:
            station_type = station_info[1]  # 站点类型 0:始发和终点站 1:通过站 2:停靠站
            arrive_time = station_info[2]  # 到达时间
            depart_time = station_info[3]  # 出发时间
            speed_limit = min(station_info[4], train_max_speed)  # 速度限制
            
            # 规则1: 通过站不允许速度为0
            if station_type == 1 and v == 0:
                continue
            # region    
            # 规则2: 停靠站只在到达和离开时间范围内允许速度为0
            # if station_type == 2 and v == 0 and (t < arrive_time or t > depart_time):
            #     continue

            # 规则2.1: 停靠站在时间窗范围外的时间点删掉
            # if station_type == 2 and (t < arrive_time - time_gap or t > depart_time + time_gap):
            #     continue 
            # endregion
            # 规则2.2: 停靠站在停靠期间速度必须为0
            if station_type == 2 and v != 0:
                continue
                
            # 规则3: 速度不能超过站点限制和自身最大速度
            if v > speed_limit or v > train_max_speed:
                continue

            # 规则6: 始发站和终点站的速度必须为0
            if station_type == 0 and v != 0:
                continue
            # region
            # 规则7: 所有车站列车的到达时间需要在预定时间的范围内 
            # 对于停靠站，检查节点时间是否在预定到达时间前10分钟到发车时间后10分钟的范围内
            # if station_type == 2:
            #     # 到达时间向前延展10分钟，发车时间向后延展10分钟
            #     earliest_allowed_time = arrive_time - time_gap
            #     latest_allowed_time = depart_time + time_gap
            #     # 检查时间是否在允许范围内
            #     if not (earliest_allowed_time <= t <= latest_allowed_time):
            #         continue
            # 规则8: 对于起点和终点站，列车的时间同样需要在计划时间的10分钟范围内
            # if station_type == 0:
            #     # 起点站使用出发时间，终点站使用到达时间
            #     if i == stations[0][0]:  # 起点站
            #         reference_time = depart_time
            #     else:  # 终点站
            #         reference_time = arrive_time
                    
            #     # 允许的时间范围：计划时间前后10分钟
            #     earliest_allowed_time = reference_time - time_gap
            #     latest_allowed_time = reference_time + time_gap
                
            #     # 检查时间是否在允许范围内
            #     if not (earliest_allowed_time <= t <= latest_allowed_time):
            #         continue           
            # endregion

        else:
            # 非站点位置的规则
            # region
            # 规则4: 非站点位置不允许速度为0
            if v == 0:
                continue

            # endregion
            # 规则5: 速度不能超过全局最大速度
            if v > train_max_speed:
                continue
        
        # 通过所有规则检查，直接添加到有效节点列表
        valid_nodes.append(node)
    
    return valid_nodes


def filter_nodes_near_plan(valid_nodes, stations_df, station_names, max_distance):
    """
    筛选离计划时刻表较近的节点
    
    参数:
        valid_nodes: 有效节点集合
        stations_df: 站点数据
        station_names: 站点名称字典
        max_distance: 节点到直线的最大距离
        
    返回:
        筛选后的节点集合
    """
    print("正在计算计划时刻表的直线方程...")
    # 提取站点位置和时间信息
    station_positions = []
    station_times = []
    
    for station in stations_df:
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
    
    # 计算直线方程参数 (使用最小二乘法)
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
                # s_range = np.array([start_pos, end_pos])
                # t_range = m * s_range + c
                # ax.plot(s_range, t_range, [0, 0], 'b--', linewidth=2, 
                #     label=f"{station_names.get(start_pos, start_pos)}-{station_names.get(end_pos, end_pos)}" if i == 0 else "_nolegend_")
                
                # 筛选离这条直线较近的节点
                for node in valid_nodes:
                    i_pos, t, v = node
                    # 只考虑在当前站点区间内的节点
                    if start_pos <= i_pos <= end_pos:
                        # 计算节点到直线的距离 (在空间-时间平面上)
                        # 点到直线距离公式: |ax + by + c| / sqrt(a^2 + b^2)
                        # 这里直线方程是 t = m*s + c，转换为标准形式 -m*s + t - c = 0
                        distance = abs(-m * i_pos + t - c) / np.sqrt(m**2 + 1)
                        
                        if distance <= max_distance:
                            close_nodes.add(node)
        
        print(f"找到离计划时刻表较近的节点: {len(close_nodes)} 个")
        
        # # 可视化这些节点                                                # todo： 这块绘制离直线较近的节点
        # for node in close_nodes:
        #     i, t, v = node
        #     ax.scatter(i, t, v, color='cyan', s=30, alpha=0.7)
        
        # plt.show()
        # 更新有效节点集合为离直线较近的节点
        return close_nodes
    else:
        print("警告：站点数量不足，无法计算直线方程")
        return valid_nodes


def create_valid_arcs(valid_nodes, stations_df, a_max, ax=None, draw_line=False):
    """
    根据有效节点创建STS网格模型中的有效弧
    
    参数:
        valid_nodes: 有效节点集合
        stations_df: 站点数据
        a_max: 最大加速度
        draw_line: 是否绘制弧线
        ax: 绘图对象
        
    返回:
        graph: 图结构
        len_valid_arcs: 有效弧的数量
    """
    graph = {}
    len_valid_arcs = 0
    
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

                # 检查是否有中间停靠站
                has_intermediate_stop = False
                for station in stations_df:
                    station_pos = station[0]
                    station_type = station[1]
                    # 检查是否是停靠站（类型2）且位于当前节点和下一节点之间
                    if station_type == 2 and i < station_pos < j:
                        has_intermediate_stop = True
                        break
                
                # 如果有中间停靠站，则不添加这条弧
                if has_intermediate_stop:
                    continue
                
                # 计算位移
                d = j - i
                
                # 检查速度和位置的耦合关系（匀加速/匀减速）  末速度应该等于2倍的平均速度减去初始速度 
                expected_end_speed = 2 * (d / 1) - v
                
                # 检查加速度约束
                acceleration = (u - v) / 1
                
                # 如果满足两个条件，则添加弧
                if abs(u - expected_end_speed) < 0.001 and abs(acceleration) <= a_max:
                    # 直接计算弧的代价并添加到图中
                    from_node = (i, t, v)
                    to_node = (j, t+1, u)
                    
                    # 初始化图结构
                    if from_node not in graph:
                        graph[from_node] = []
                    
                    # 计算能耗成本
                    cost = 1.0  # 基础代价
                    
                    # 计算加速度和能耗
                    acc = max((u - v), 0)
                    
                    # 特殊情况处理
                    if i == j:  # 火车等待或停止的弧
                        energy = 0
                    elif u < v:  # 减速的行驶弧
                        energy = 0
                    else:  # 加速或匀速
                        m = 1.0  # 列车质量假设为1
                        energy = m * acc * (v + u) / 2
                    
                    cost = energy
                    
                    # 考虑与时刻表的一致性
                    for station in stations_df:
                        station_pos, station_type, arrive_time, depart_time, _ = station
                        
                        if i == station_pos:
                            # 出发站点，与出发时间比较
                            time_diff = abs(t - depart_time)
                            cost += time_diff * 10.0
                        elif j == station_pos:
                            # 到达站点，与到达时间比较
                            time_diff = abs(t+1 - arrive_time)
                            cost += time_diff * 10.0
                    
                    # 添加到图中
                    graph[from_node].append((to_node, cost)) 
                    len_valid_arcs += 1
                    
                    if draw_line and ax is not None:
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
        
    return graph, len_valid_arcs



def create_train_schedule_sts_grid(stations_df, station_names, delta_d=5,
                                   delta_t=5, speed_levels=5, time_diff_minutes=5*60, # 默认为5小时的时间窗
                                   total_distance=50, draw_plan=True, draw_line=False,
                                   max_distance=30, select_near_plan=True, a_max=5,
                                   train_max_speed=5):   # 50km的线路
    """
    创建包含列车时刻表约束的STS网格模型
    
    参数:
        stations_df: 列车时刻表列表，每个元素包含[位置, 类型, 到达时间, 出发时间，速度限制]
        station_names: 车站名称字典，键为位置，值为站名
        delta_d: 空间单位长度，默认为5km
        delta_t: 时间单位长度，默认为5分钟
        speed_levels: 速度级别数量，默认为5 也就是最大速度为300km/h
        time_diff_minutes: 时间窗口长度，默认为5小时1分钟
        total_distance: 线路总长度，默认为50km
        draw_plan: 是否绘制计划时刻表，默认为True
        draw_line: 是否绘制有效弧，默认为False
        max_distance: 节点到计划时刻表直线的最大距离，默认为30
        select_near_plan: 是否使用计划表完成有效点的筛选，默认为True
        a_max: 最大加速度约束，默认为5
        train_max_speed: 列车最大速度，默认为300km/h

    返回:
        创建的STS网格模型，以及规划的列车速度位置曲线
    """

    """创建包含列车时刻表约束的STS网格模型"""  
    import math 
    
    # 初步计算
    time_nodes =math.ceil(time_diff_minutes / delta_t)
    space_segments = math.ceil(total_distance / delta_d)

    time_nodes, space_segments, delta_t, delta_d = check_time_nodes_and_space_segments(time_nodes, space_segments, delta_t, delta_d, time_diff_minutes, total_distance)
    
    # 转换站点时刻表为单位值，位置为栅格索引
    stations_df, station_names = trans_df_grid(stations_df, station_names, delta_t, delta_d)

    # 将站点位置映射到分段后的位置
    station_positions = [int(round(stations_df[i][0])) for i in range(len(stations_df))]    
    
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标范围
    space_range = np.arange(0, space_segments + 1)
    time_range = np.arange(0, time_nodes+1)
    speed_range = np.arange(0, speed_levels+1)

    """============================创建STS网格模型============================="""  
    print("正在创建STS网格模型...")
    # 创建节点集合
    all_nodes = set()
    for i in space_range:  # 空间维度
        for t in time_range:  # 时间维度
            for v in speed_range:  # 速度维度
                all_nodes.add((i, t, v))
    print(f"初始网格节点总数: {len(all_nodes)}")
    
    # 筛选有效节点
    valid_nodes = filter_valid_nodes(all_nodes, station_positions, stations_df, train_max_speed)
    if select_near_plan:
        print("根据计划时刻表筛选出离计划时刻表较近的节点...")  
        valid_nodes = filter_nodes_near_plan(valid_nodes, stations_df, station_names, max_distance) # 调用函数筛选节点

    """======================绘制有效STS网格节点============================="""  
    plot_sts_grid_nodes(ax, station_names, time_nodes, valid_nodes=valid_nodes)

    """============================添加有效弧============================="""  
    graph, len_valid_arcs = create_valid_arcs(valid_nodes, stations_df, a_max, ax, draw_line=draw_line)   
    print(f"添加的有效弧数量: {len_valid_arcs}") 

    """======================绘制时间-空间二维平面上的计划方案======================="""      
    draw_plan_on_sts_grid(ax, stations_df, space_segments, time_nodes, draw_plan=draw_plan)

    """=========================动态规划方法构建最优路径============================"""  
    # 使用动态规划方法来从这个STS网络里构建出最优路径出来，得到时刻表
    start_node = (stations_df[0][0], stations_df[0][2], 0)
    end_node = (stations_df[-1][0], stations_df[-1][2], 0)  # 因为时空速度都是网格里的坐标，如果还不在网格里，只有一种可能，就是网格范围不对
    print(f"起点节点: {start_node}")
    print(f"终点节点: {end_node}")
    # 使用Dijkstra算法找最短路径
    # check_end_node_in_graph(graph, stations_df)   # debug
    # 调用函数寻找最优路径
    path, dist, prev = find_optimal_path_with_dp(valid_nodes, graph, start_node, end_node, ax)
        
    # 如果没有找到路径，直接返回
    if path is None:
        return
    
    if path[0] != start_node or path[-1] != end_node:
        print("错误：无法找到从起点到终点的有效路径")
    else:
        print("最优路径已找到！")
        print(f"路径长度: {len(path)} 节点") 
        print(f"总能耗代价: {dist[end_node]:.2f}")
                
        # 导出时刻表
        station_actual_times = export_train_schedule(path, station_positions, station_names, stations_df)
        
        # 更新图例
        ax.legend()

        # 绘制最优路径轨迹
        print("\n正在绘制最优路径轨迹...")
        plot_optimal_path(ax, path)
        print("最优路径轨迹绘制完成！")
                        
        # 调用函数完善图形显示
        finalize_plot(ax, space_segments, time_nodes, speed_levels, delta_d, delta_t)

if __name__ == "__main__":
    train_schedule = [
        [0, 0, '8:00', '8:00', 4],  # 北京南站(始发站) 
        [70, 2, '8:20', '8:25', 4],  # 廊坊站(停靠站) 
        [140, 1, '9:00', '9:00', 4],  # 天津站(通过站)
        [200, 0, '9:30', '9:30', 4],  # 滨海站(终点站)
    ]
    station_names = {
        0: "北京南",
        70: "廊坊",
        140: "天津",
        200: "滨海"
    }
    create_train_schedule_sts_grid(train_schedule, 
                                   station_names, 
                                   delta_d=5,      # 5km
                                   delta_t=5,      # 5分钟
                                   speed_levels=5, 
                                   time_diff_minutes=2*60+1,   # 调度范围是2个小时
                                   total_distance=300,    # 300km的线路
                                   draw_plan=True, 
                                   draw_line=False,
                                   max_distance=10)    # 10个格子