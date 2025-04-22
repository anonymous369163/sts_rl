"""
本代码是用来根据给定的时刻表，生成所有列车的完整运行轨迹
"""


import numpy as np
import matplotlib.pyplot as plt 
from sts_model_single_train import check_end_node_in_graph, convert_time_to_units, Arrow3D



def create_train_schedule_sts_grid():
    """创建包含列车时刻表约束的STS网格模型"""
    # 基本参数
    time_nodes = 31   
    space_segments = 100  # 将总距离分成20段
    speed_levels = 5  # 速度级别
    delta_t = 1  # 时间增量
    
    # 实际站点距离信息(单位: km)
    # 北京南站-廊坊站-天津南站-沧州站-德州站-济南西站
    station_names = ['北京南', '廊坊']
    
    # 各站点的实际里程(从起点开始)
    station_distances = [0, 10]  # 单位: km
    
    # 计算总距离
    total_distance = station_distances[-1]
    
    # 空间维度等间隔分割 
    delta_d = total_distance / space_segments  # 每段距离(km)
    
    # 将站点位置映射到分段后的位置
    station_positions = [int(round(dist / delta_d)) for dist in station_distances]
    
    # 列车信息
    train_max_speed = 10  # 最高速度级别(3·Δd/Δt)
    
    # 站点信息 [站点ID, 类型(0=始发/终点,1=通过站,2=停靠站), 到达时间, 出发时间, 速度限制]
    stations = [
        [station_positions[0], 0, '8:00', '8:00', 10],  # 北京南站(始发站)，出发时间0
        [station_positions[1], 0, '8:30', '8:30', 10],  # 廊坊站(通过站)，时间点1 
    ]

    # 转换站点时刻表
    for i in range(len(stations)):
        arrive_time = stations[i][2]
        depart_time = stations[i][3]
        
        # 转换到达和出发时间
        stations[i][2] = convert_time_to_units(arrive_time, delta_t)
        stations[i][3] = convert_time_to_units(depart_time, delta_t)
    
    fig = plt.figure(figsize=(15, 12))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标范围
    space_range = np.arange(0, space_segments + 1)
    time_range = np.arange(0, time_nodes)
    speed_range = np.arange(0, speed_levels)

    time_gap = 2

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
            if station_type == 2 and v == 0 and (t < arrive_time or t > depart_time):
                continue

            # 规则2.1: 停靠站在时间窗范围外的时间点删掉
            if station_type == 2 and (t < arrive_time - time_gap or t > depart_time + time_gap):
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
            # 规则7: 所有车站列车的到达时间需要在预定时间的范围内 
            # 对于停靠站，检查节点时间是否在预定到达时间前10分钟到发车时间后10分钟的范围内
            if station_type == 2:
                # 到达时间向前延展10分钟，发车时间向后延展10分钟
                earliest_allowed_time = arrive_time - time_gap
                latest_allowed_time = depart_time + time_gap
                # 检查时间是否在允许范围内
                if not (earliest_allowed_time <= t <= latest_allowed_time):
                    continue
            # 规则8: 对于起点和终点站，列车的时间同样需要在计划时间的10分钟范围内
            if station_type == 0:
                # 起点站使用出发时间，终点站使用到达时间
                if i == stations[0][0]:  # 起点站
                    reference_time = depart_time
                else:  # 终点站
                    reference_time = arrive_time
                    
                # 允许的时间范围：计划时间前后10分钟
                earliest_allowed_time = reference_time - time_gap
                latest_allowed_time = reference_time + time_gap
                
                # 检查时间是否在允许范围内
                if not (earliest_allowed_time <= t <= latest_allowed_time):
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

    """============================绘制有效STS网格节点============================="""  
    # 将有效节点绘制成三维空间的散点图
    print("正在绘制有效STS网格节点...")
    
    # 提取有效节点的坐标
    space_coords = [node[0] for node in valid_nodes]
    time_coords = [node[1] for node in valid_nodes]
    speed_coords = [node[2] for node in valid_nodes]
    
    # 绘制散点图 - 减小点的大小从30到15
    # ax.scatter(space_coords, time_coords, speed_coords,                       # todo： 这块绘制有效的节点
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
                ax.scatter(i, t, v, color='red', s=80, marker='*')
            
            # 停靠站的节点
            elif station_type == 2 and v == 0:
                ax.scatter(i, t, v, color='purple', s=80, marker='s')
    # 设置图形标题和轴标签
    ax.set_title('列车时刻表STS网格模型可视化')
    ax.set_xlabel('空间维度 (站点位置/km)')
    ax.set_ylabel('时间维度 (时间/min)')
    ax.set_zlabel('速度维度 (速度/km/min)')

    # 在图中标记车站信息
    print("正在标记车站信息...")
    
    # 假设stations列表中包含车站名称信息
    # 如果stations中没有具体车站名，这里需要定义车站名称
    station_names = {
        station_positions[0]: "北京南",
        station_positions[1]: "廊坊"
    }
    
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
    
    # 添加网格线以提高可读性
    ax.grid(True)
    
    # 调整视角以获得更好的3D效果
    ax.view_init(elev=30, azim=45)
         
    # 接下来就是添加有效的弧
    # 添加有效弧 - 根据物理约束连接相邻时间步的节点
    # plt.show()

    """============================添加有效弧============================="""  
    print("开始添加有效弧...")
    # 计算计划时刻表的直线方程
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
        A = np.vstack([station_positions, np.ones(len(station_positions))]).T
        m, c = np.linalg.lstsq(A, station_times, rcond=None)[0]
        
        print(f"计划时刻表直线方程: t = {m:.4f} * s + {c:.4f}")
        
        # 绘制计划时刻表直线
        s_range = np.array([min(station_positions), max(station_positions)])
        t_range = m * s_range + c
        ax.plot(s_range, t_range, [0, 0], 'b--', linewidth=2, label='计划时刻表')
        
        # 定义节点到直线的最大距离阈值
        max_distance = 5  # 可以根据需要调整
        
        # 筛选离直线较近的节点
        close_nodes = set()
        for node in valid_nodes:
            i, t, v = node
            # 计算节点到直线的距离 (在空间-时间平面上)
            # 点到直线距离公式: |ax + by + c| / sqrt(a^2 + b^2)
            # 这里直线方程是 t = m*s + c，转换为标准形式 -m*s + t - c = 0
            distance = abs(-m * i + t - c) / np.sqrt(m**2 + 1)
            
            if distance <= max_distance:
                close_nodes.add(node)
        
        print(f"找到离计划时刻表较近的节点: {len(close_nodes)} 个")
        
        # 可视化这些节点                                                # todo： 这块绘制离直线较近的节点
        # for node in close_nodes:
        #     i, t, v = node
        #     ax.scatter(i, t, v, color='cyan', s=30, alpha=0.7)
        
        # 更新有效节点集合为离直线较近的节点
        valid_nodes = close_nodes
    else:
        print("警告：站点数量不足，无法计算直线方程")

    draw_line = False
    # 定义最大加速度约束
    a_max = 10  # 最大加速度阈值
    
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
                
                # 计算位移
                d = j - i
                
                # 检查速度和位置的耦合关系（匀加速/匀减速）
                # 末速度应该等于2倍的平均速度减去初始速度 
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
    
    print(f"添加的有效弧数量: {len(valid_arcs)}") 

    """======================绘制时间-空间二维平面上的计划方案=======================""" 

    # 在当前三维图中添加二维投影
    # 创建一个平面来显示时间-空间投影
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
        # 能耗模型：加速消耗更多能量，匀速消耗适中，减速消耗较少
        cost = 1.0  # 基础代价
        if v2 > v1:  # 加速
            cost += 0.5 * (v2 - v1)**2  # 加速能耗与加速度平方成正比
        elif v2 < v1:  # 减速
            cost += 0.1 * (v1 - v2)  # 减速能耗较小
        else:  # 匀速
            cost += 0.2 * v1  # 匀速能耗与速度成正比
        
        # 额外考虑与时刻表的一致性
        # 检查是否经过站点，如果经过站点，检查时间是否与时刻表一致
        for station in stations:
            station_pos, station_type, arrive_time, depart_time, _ = station
            
            # 如果弧的起点或终点在站点上
            if (i == station_pos and t1 != arrive_time and t1 != depart_time) or \
               (j == station_pos and t2 != arrive_time and t2 != depart_time):
                # 不符合时刻表的站点时间，增加惩罚
                cost += 100.0
                break
        
        graph[from_node].append((to_node, cost))
    
    # 使用Dijkstra算法找最短路径
    check_end_node_in_graph(graph, stations)
    import heapq
    import math # 使用 math.inf 替代 float('inf')

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
            
        # 如果节点已经访问过（其最短路径已确定），跳过
        # 注意：在某些Dijkstra实现中，此检查可能不是必需的，
        # 因为 d > dist[u] 的检查通常足够。但加上也无妨。
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


    # 重建最短路径
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
        
        # 提取路径坐标用于绘图
        path_space = [node[0] for node in path]
        path_time = [node[1] for node in path]
        path_speed = [node[2] for node in path]
        
        # 在图上绘制最优路径
        ax.plot(path_space, path_time, path_speed, 'y-', linewidth=3, label='最优路径')
        
        # 在路径上标记关键点
        for idx, node in enumerate(path):
            i, t, v = node
            # 检查是否在站点上
            for station in stations:
                if i == station[0]: 
                    # 检查是否是到达或出发时间点
                    if t == station[2] or t == station[3]:
                        ax.scatter(i, t, v, color='yellow', s=120, marker='*', edgecolors='black') 
                    break
        
        # 打印时刻表
        print("\n最优路径时刻表:")
        print("站点\t计划到达\t计划出发\t实际到达\t实际出发\t最高速度")
        
        # 记录每个站点的实际到达和出发时间
        station_actual_times = {}
        
        for node in path:
            i, t, v = node
            # 检查是否在站点上
            for station_idx, station_pos in enumerate(station_positions):
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
            
            print(f"{station_names[station_positions[station_idx]]}\t{plan_arrive}\t{plan_depart}\t{actual_arrive}\t{actual_depart}\t{max_speed}")
        
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
        plt.tight_layout() 
        plt.show() 

if __name__ == "__main__":
    create_train_schedule_sts_grid() 