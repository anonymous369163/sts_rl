"""
本代码是用来生成一个运行区间的列车时刻表的STS网格模型，根据给定的计划时刻表，完成最优列车速度位置曲线的确定
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


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


def create_train_schedule_sts_grid(train_schedule, station_names_input, delta_d=0.5,
                                   delta_t=20/60, speed_levels=5, time_diff_minutes=5*60+1, # 默认为5小时的时间窗
                                   total_distance=50, draw_plan=False, draw_line=False,
                                   max_distance=30, select_near_plan=True, a_max=10):   # 50km的线路
    """
    创建包含列车时刻表约束的STS网格模型
    
    参数:
        train_schedule: 列车时刻表列表，每个元素包含[位置, 类型, 到达时间, 出发时间]
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
    
    返回:
        创建的STS网格模型
    """

    """创建包含列车时刻表约束的STS网格模型""" 
    import copy
    stations = copy.deepcopy(train_schedule)
    station_names = copy.deepcopy(station_names_input) 
    
    # 计算时间节点数和空间段数 
    station_distances = [stations[i][0] for i in range(len(stations))]  
    
    # 初步计算
    time_nodes = time_diff_minutes / delta_t
    space_segments = total_distance / delta_d
    
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
    time_nodes = int(round(time_diff_minutes / delta_t)) # 使用传入的时间差分钟数
    space_segments = int(round(total_distance / delta_d))

    print(f"时间节点数: {time_nodes}, 空间段数: {space_segments}")
    print(f"时间间隔: {delta_t*60:.2f}秒, 空间间隔: {delta_d*1000:.2f}米")
    print(f"总时间: {time_nodes * delta_t:.2f} 分钟, 总距离: {space_segments * delta_d:.2f} km")
    
    # 转换站点时刻表为单位值，位置为栅格索引
    for i in range(len(stations)):
        arrive_time = stations[i][2]
        depart_time = stations[i][3]
        # 转换到达和出发时间
        stations[i][2] = convert_time_to_units(arrive_time, delta_t)
        stations[i][3] = convert_time_to_units(depart_time, delta_t) 
        stations[i][0] = int(round(stations[i][0] / delta_d))
    # 修改station_names中的键，使其与栅格化后的站点位置对应
    new_station_names = {}
    for pos, name in station_names.items():
        # 栅格化站点位置
        new_pos = int(round(pos / delta_d))
        new_station_names[new_pos] = name 
    station_names = new_station_names 
        
    # 将站点位置映射到分段后的位置
    station_positions = [int(round(dist / delta_d)) for dist in station_distances]
    
    # 列车信息
    train_max_speed = 20  # 最高速度级别(3·Δd/Δt)
    
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
        
        # 如果节点在站点上，应用站点特定规则
        if is_at_station:
            station_type = station_info[1]
            arrive_time = station_info[2]
            depart_time = station_info[3]
            speed_limit = min(station_info[4], train_max_speed)
            
            # 规则1: 通过站不允许速度为0
            if station_type == 1 and v == 0:
                continue
                
            # 规则2: 停靠站只在到达和离开时间范围内允许速度为0
            # if station_type == 2 and v == 0 and (t < arrive_time or t > depart_time):
            #     continue

            # 规则2.1: 停靠站在时间窗范围外的时间点删掉
            # if station_type == 2 and (t < arrive_time - time_gap or t > depart_time + time_gap):
            #     continue 
                
            # 规则2.2: 停靠站在停靠期间速度必须为0
            if station_type == 2 and v != 0:
                continue
                
            # 规则3: 速度不能超过站点限制
            if v > speed_limit:
                continue
            # 规则6: 始发站和终点站的速度必须为0
            if station_type == 0 and v != 0:
                continue
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
    """============================绘制有效STS网格节点============================="""  
    # 将有效节点绘制成三维空间的散点图
    print("正在绘制有效STS网格节点...") 
    # 提取有效节点的坐标    # todo： 这块绘制有效的节点
    # space_coords = [node[0] for node in valid_nodes]
    # time_coords = [node[1] for node in valid_nodes]
    # speed_coords = [node[2] for node in valid_nodes] 
    # ax.scatter(space_coords, time_coords, speed_coords,                     
    #            color='blue', s=15, alpha=0.6, marker='o',
    #            label='有效STS网格顶点')
    
    # 为不同类型的节点使用不同颜色
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
                # ax.scatter(i, t, v, color='red', s=80, marker='*')
                # 在站点位置绘制垂直线，只画一次
                if not hasattr(ax, f'vertical_line_drawn_{i}'):
                    ax.plot([i, i], [0, time_nodes], [0, 0], color='red', linestyle='--', alpha=0.5)
                    setattr(ax, f'vertical_line_drawn_{i}', True)
            
            # 停靠站的节点
            elif station_type == 2 and v == 0:
                # ax.scatter(i, t, v, color='purple', s=80, marker='s')
                # 在站点位置绘制垂直线，只画一次
                if not hasattr(ax, f'vertical_line_drawn_{i}'):
                    ax.plot([i, i], [0, time_nodes], [0, 0], color='purple', linestyle='--', alpha=0.5)
                    setattr(ax, f'vertical_line_drawn_{i}', True)
    # 设置图形标题和轴标签
    ax.set_title('列车时刻表STS网格模型可视化')
    ax.set_xlabel('空间维度 (站点位置/km)')
    ax.set_ylabel('时间维度 (时间/min)')
    ax.set_zlabel('速度维度 (速度/km/min)')

    # 在图中标记车站信息
    print("正在标记车站信息...")
    
    # 为每个车站添加标记和标签
    for station in stations:
        position, station_type = station[0], station[1]
        
        # 获取车站名称
        station_name = ""
        if len(station) > 2:  # 如果stations中已包含车站名
            station_name = station[2]
        elif position in station_names:  # 否则从我们定义的字典中获取
            station_name = station_names[position]
        else:
            station_name = f"站点{position}"  # 默认名称
        
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
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 添加网格线以提高可读性，按照实际时空分割的最小单位进行分割
    ax.grid(True)  # 先关闭默认网格
    
    # # 在空间维度上添加网格线（每个delta_d单位）
    # for i in range(0, int(space_segments) + 1, 1):
    #     ax.plot([i, i], [0, time_nodes], [0, 0], 'gray', alpha=0.3, linestyle=':')
    
    # # 在时间维度上添加网格线（每个delta_t单位）
    # for t in range(0, int(time_nodes) + 1, 1):
    #     ax.plot([0, space_segments], [t, t], [0, 0], 'gray', alpha=0.3, linestyle=':')
    
    # # 在速度维度上添加网格线（每个速度级别）
    # for v in range(0, speed_levels + 1, 1):
    #     ax.plot([0, 0], [0, time_nodes], [v, v], 'gray', alpha=0.3, linestyle=':')
    
    # print(f"已添加网格线：空间单位={delta_d}km，时间单位={delta_t}min，速度级别={speed_levels}")
    
    # 调整视角以获得更好的3D效果
    ax.view_init(elev=30, azim=45)
         
    """============================添加有效弧============================="""  
    print("开始添加有效弧...") 

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
                    s_range = np.array([start_pos, end_pos])
                    t_range = m * s_range + c
                    ax.plot(s_range, t_range, [0, 0], 'b--', linewidth=2, 
                        label=f"{station_names.get(start_pos, start_pos)}-{station_names.get(end_pos, end_pos)}" if i == 0 else "_nolegend_")
                    
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
            
            # 可视化这些节点                                                # todo： 这块绘制离直线较近的节点
            # for node in close_nodes:
            #     i, t, v = node
            #     ax.scatter(i, t, v, color='cyan', s=30, alpha=0.7)
            
            # plt.show()
            # 更新有效节点集合为离直线较近的节点
            valid_nodes = close_nodes
            
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

                # 检查是否有中间停靠站
                has_intermediate_stop = False
                for station in stations:
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
    # plt.show()
    print(f"添加的有效弧数量: {len(valid_arcs)}") 

    """======================绘制时间-空间二维平面上的计划方案=======================""" 
    if draw_plan:
        # 在当前三维图中添加二维投影 
        x_min, x_max = 0, space_segments
        y_min, y_max = 0, time_nodes
        z_min = 0
        
        # 创建一个平面网格
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(y_min, y_max, 2))
        zz = np.zeros_like(xx) + z_min
        
        # 绘制半透明平面
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        # 在平面上绘制站点位置
        for station in stations:
            station_pos = station[0]
            # 绘制站点水平线
            ax.plot([station_pos, station_pos], [0, time_nodes], [z_min, z_min], 'k--', alpha=0.5)
        
        # 直接根据站点计划时间绘制列车运行线
        # 创建站点位置和时间的列表
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
        
        # 绘制列车运行线（改为浅蓝色虚线）
        ax.plot(train_space_coords, train_time_coords, [z_min] * len(train_space_coords), 
                'c--', alpha=0.9, linewidth=3, label='列车运行线')
        
        # 标记站点的计划时间
        for station in stations:
            station_pos = station[0]
            station_type = station[1]
            arrive_time = station[2]
            depart_time = station[3]
            
            # 对于始发站，只标记出发时间
            if station_type == 0 and station_pos == stations[0][0]:
                ax.scatter(station_pos, depart_time, z_min, color='red', s=80, marker='o') 
            
            # 对于终点站，只标记到达时间
            elif station_type == 0 and station_pos == stations[-1][0]:
                ax.scatter(station_pos, arrive_time, z_min, color='red', s=80, marker='o') 
            
            # 对于停靠站，标记到达和出发时间
            elif station_type == 2:
                station_idx = [s[0] for s in stations].index(station_pos) 
                
                ax.scatter(station_pos, arrive_time, z_min, color='green', s=80, marker='o') 
                
                ax.scatter(station_pos, depart_time, z_min, color='blue', s=80, marker='o') 
            
            # 对于通过站，标记通过时间
            elif station_type == 1:
                station_idx = [s[0] for s in stations].index(station_pos) 
                
                ax.scatter(station_pos, arrive_time, z_min, color='purple', s=80, marker='o') 
        
        # 添加图例说明
        ax.text(0, time_nodes-1, z_min, "时间-空间二维投影", color='black', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    """============================动态规划方法构建最优路径============================="""  
    # 使用动态规划方法来从这个STS网络里构建出最优路径出来，得到时刻表
    print("正在使用动态规划方法构建最优路径...")
    
    # 找出起点和终点节点
    start_node = None
    end_node = None
    
    for node in valid_nodes:
        i, t, v = node
        # 检查是否为起点（第一个站点，时间为0，速度为0）
        if i == stations[0][0] and t == stations[0][2] and v == 0:
            start_node = node
        # 检查是否为终点（最后一个站点，时间为最后一个时间点，速度为0）
        elif i == stations[-1][0] and t == stations[-1][2] and v == 0:
            end_node = node
    
    if not start_node or not end_node:
        print("错误：无法找到起点或终点节点")
        return
    
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
        
        # 计算弧的代价（考虑能耗最小化）
        # 计算能耗
        cost = 1.0  # 基础代价
        
        # 计算加速度 acc(i, j, t, s, u, v)
        if t2 > t1:  # 确保时间差不为零
            # 根据公式计算加速度：acc(i, j, t, s, u, v) = max{(v-u)/(s-t) + w((u+v)/2) + (h(i)+h(j))/2, 0}
            # 这里简化处理，不考虑坡度和风阻，即 w=0, h(i)=h(j)=0
            acc = max((v2 - v1) / (t2 - t1), 0)
            
            # 计算能耗 e(i, j, t, s, u, v) = m × acc × (u + v) / 2 × (s − t)
            # 假设列车质量 m = 1（可以根据需要调整）
            m = 1.0
            
            # 特殊情况处理
            if i == j:  # 火车等待或停止的弧
                energy = 0
            elif v2 < v1:  # 减速的行驶弧
                energy = 0
            else:  # 加速或匀速
                energy = m * acc * (v1 + v2) / 2 * (t2 - t1)
                
            cost = energy
        
        # 额外考虑与时刻表的一致性 
        for station in stations:
            station_pos, station_type, arrive_time, depart_time, _ = station
            
            # 如果弧的起点或终点在站点上，计算与计划时间的偏差
            if i == station_pos:
                # 如果是出发站点，与出发时间比较
                time_diff = abs(t1 - depart_time)
                # 根据偏差大小增加惩罚，偏差越大惩罚越大
                cost += time_diff * 10.0  # 使用线性惩罚，可以根据需要调整系数
            elif j == station_pos:
                # 如果是到达站点，与到达时间比较
                time_diff = abs(t2 - arrive_time)
                # 根据偏差大小增加惩罚，偏差越大惩罚越大
                cost += time_diff * 10.0  # 使用线性惩罚，可以根据需要调整系数
        
        graph[from_node].append((to_node, cost))
    # 使用Dijkstra算法找最短路径
    check_end_node_in_graph(graph, stations)
    import heapq
    import math  

    # 初始化距离和前驱节点字典
    # 确保图中所有节点（包括只有入度或只有出度的节点）都被包含
    all_graph_nodes = set(graph.keys())
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            all_graph_nodes.add(neighbor)
    # 确保起点和终点在节点集合中
    if start_node: all_graph_nodes.add(start_node)
    if end_node: all_graph_nodes.add(end_node)

    dist = {node: math.inf for node in all_graph_nodes}
    prev = {node: None for node in all_graph_nodes}

    # 检查起点是否存在于图中
    if start_node not in dist:
        print(f"错误：起始节点 {start_node} 不在图的节点集合中。")
        # 可以选择返回或抛出异常
        return
    
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
            
        # 如果节点已经访问过（其最短路径已确定），跳过 注意：在某些Dijkstra实现中，此检查可能不是必需的， 因为 d > dist[u] 的检查通常足够。但加上也无妨。
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
                # else:
                    # 这个情况理论上不应发生，因为 all_graph_nodes 已包含所有节点
                    # print(f"警告：邻居节点 {v} 未在距离字典中初始化。")

    # 检查是否找到了到达终点的路径
    if dist.get(end_node, math.inf) == math.inf:
        print(f"错误：无法从起点 {start_node} 找到到达终点 {end_node} 的路径。")
        # 根据需要处理错误，例如显示图形并返回
        # 标记 dist 中最短路径不为无穷大的节点 (即可达节点)
        # 这个代码块在终点不可达时执行
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
                # else:
                    # 可以选择性地添加警告，如果 dist 中的键格式不正确
                    # print(f"警告：dist 中的键 {node} 格式不正确，无法在图上标记。")

        # 如果没有可达节点被标记（除了起点本身），也打印一条信息
        if not reachable_nodes_labeled and start_node in dist and dist[start_node] == 0:
             # 如果只有起点可达，单独标记起点（如果需要）
             # ax.scatter(start_node[0], start_node[1], start_node[2], color='orange', s=45, marker='p', alpha=0.8, label='可达节点')
             print("除了起点外，没有其他可达节点。")
        plt.title("STS网格模型（未找到路径）")
        plt.show()
        return

    """============================重建最短路径============================="""
    path = []
    current = end_node
    
    while current:
        path.append(current)
        current = prev[current]
    
    path.reverse()  # 反转路径，从起点到终点
    
    if path[0] != start_node or path[-1] != end_node:
        print("错误：无法找到从起点到终点的有效路径")
    else:
        print("最优路径已找到！")
        print(f"路径长度: {len(path)} 节点")
        print(f"总能耗代价: {dist[end_node]:.2f}")
                
        # 打印时刻表
        print("\n最优路径时刻表:")
        print("站点\t计划到达\t计划出发\t实际到达\t实际出发\t最高速度")
        
        # 记录每个站点的实际到达和出发时间
        station_actual_times = {}
        
        for node in path:
            i, t, v = node
            # 检查是否在站点上
            for station_pos in station_positions:
                if i == station_pos:
                    if station_pos not in station_actual_times:
                        station_actual_times[station_pos] = {"arrive": t, "depart": t, "max_speed": v}
                    else:
                        # 更新出发时间和最高速度
                        station_actual_times[station_pos]["depart"] = t
                        station_actual_times[station_pos]["max_speed"] = max(station_actual_times[station_pos]["max_speed"], v)
        
        # 打印对比时刻表
        for station_idx, station in enumerate(stations):
            station_pos, _, plan_arrive, plan_depart, _ = station
            actual_arrive = station_actual_times.get(station_pos, {}).get("arrive", "-")
            actual_depart = station_actual_times.get(station_pos, {}).get("depart", "-")
            max_speed = station_actual_times.get(station_pos, {}).get("max_speed", 0)
            
            print(f"{station_names[station_pos]}\t{plan_arrive}\t{plan_depart}\t{actual_arrive}\t{actual_depart}\t{max_speed}")
        
        # 计算与时刻表的一致性
        consistency = 0
        total_stations = len(stations)
        for station_pos, times in station_actual_times.items():
            for station in stations:
                if station_pos == station[0]:
                    if times["arrive"] == station[2] and times["depart"] == station[3]:
                        consistency += 1
                    break
        
        consistency_percentage = (consistency / total_stations) * 100
        print(f"\n时刻表一致性: {consistency_percentage:.2f}% ({consistency}/{total_stations}个站点完全符合)")
        
        # 更新图例
        ax.legend()

        # 绘制最优路径轨迹
        print("\n正在绘制最优路径轨迹...")
        
        # 提取路径中的坐标
        path_space = [node[0] for node in path]
        path_time = [node[1] for node in path]
        path_speed = [node[2] for node in path]
        
        # 绘制路径线
        ax.plot(path_space, path_time, path_speed, color='red', linewidth=3, 
                linestyle='-', marker='o', markersize=5, label='最优路径')
        
        # 添加路径起点和终点的特殊标记
        ax.scatter(path_space[0], path_time[0], path_speed[0], 
                  color='green', s=150, marker='*', edgecolors='black', label='起点')
        ax.scatter(path_space[-1], path_time[-1], path_speed[-1], 
                  color='red', s=150, marker='*', edgecolors='black', label='终点')
        
        # 在三维空间中添加路径投影到各个平面
        # 投影到空间-时间平面
        ax.plot(path_space, path_time, np.zeros_like(path_speed), 
                color='blue', linewidth=2, linestyle='--', alpha=0.5, label='空间-时间投影')
        
        # 投影到空间-速度平面
        ax.plot(path_space, np.zeros_like(path_time), path_speed, 
                color='green', linewidth=2, linestyle='--', alpha=0.5, label='空间-速度投影')
        
        # 投影到时间-速度平面
        ax.plot(np.zeros_like(path_space), path_time, path_speed, 
                color='purple', linewidth=2, linestyle='--', alpha=0.5, label='时间-速度投影')
        
        print("最优路径轨迹绘制完成！")
        
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
        
        plt.tight_layout() 
        plt.show() 

if __name__ == "__main__":
    train_schedule = [
        [0, 0, '8:00', '8:00', 10],  # 北京南站(始发站) 
        [70, 2, '8:20', '8:25', 10],  # 廊坊站(停靠站) 
        [140, 1, '9:00', '9:00', 10],  # 天津站(通过站)
        [200, 0, '9:30', '9:30', 10],  # 滨海站(终点站)
    ]
    station_names = {
        0: "北京南",
        70: "廊坊",
        140: "天津",
        200: "滨海"
    }
    create_train_schedule_sts_grid(train_schedule, 
                                   station_names, 
                                   delta_d=5,    # 0.5km
                                   delta_t=5,   # 20秒
                                   speed_levels=5, 
                                   time_diff_minutes=2*60+1,   # 调度范围是一个小时
                                   total_distance=300,    # 100km的线路
                                   draw_plan=False, 
                                   draw_line=False,
                                   max_distance=10)    # 20