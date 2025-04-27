"""
工具函数
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_sts_grid_nodes(ax, station_names, time_nodes, valid_nodes):
    """
    绘制STS网格节点和站点标记
    
    参数:
        ax: 3D图形对象
        stations_df: 站点数据
        station_names: 站点名称
        time_nodes: 时间节点数量
        space_segments: 空间分段数量
        speed_levels: 速度级别数量
    """
    # 为每个车站添加标记和标签
    for position in station_names.keys():
        station_name = station_names[position]
        line_color = 'red'
        marker_color = 'red'
        marker_style = '^'
        marker_size = 150 

        # 在空间轴上标记车站位置（在z=0平面上） # 在站点位置绘制垂直线
        ax.plot([position, position], [0, time_nodes], [0, 0], color=line_color, linestyle='--', alpha=0.5)
        ax.scatter(position, 0, 0, color=marker_color, s=marker_size, marker=marker_style)
        
        # 添加具体车站名称标签
        ax.text(position, -1, -0.5, station_name, color=marker_color, 
                fontsize=12, ha='center', va='top', weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=marker_color, boxstyle='round,pad=0.3'))
    
    ax.set_title('列车时刻表STS网格模型可视化')
    ax.set_xlabel('空间维度 (站点位置/km)')
    ax.set_ylabel('时间维度 (时间/min)')
    ax.set_zlabel('速度维度 (速度/km/min)') 
    ax.scatter([], [], color='red', s=100, marker='^', label='始发/终点站')
    ax.scatter([], [], color='purple', s=100, marker='s', label='停靠站')
    ax.scatter([], [], color='green', s=100, marker='o', label='通过站')
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 添加网格线以提高可读性
    ax.grid(True)

    # 绘制有效节点
    if valid_nodes:
        # 提取有效节点的坐标
        valid_x = [node[0] for node in valid_nodes]
        valid_y = [node[1] for node in valid_nodes]
        valid_z = [node[2] for node in valid_nodes]
        
        # 绘制有效节点
        ax.scatter(valid_x, valid_y, valid_z, color='blue', s=10, alpha=0.3, label='有效节点')
        
        print(f"绘制了 {len(valid_nodes)} 个有效节点")
    else:
        print("没有有效节点可供绘制")


def draw_plan_on_sts_grid(ax, stations_df, space_segments, time_nodes, draw_plan=True):
    """在STS网格上绘制列车运行计划"""
    if not draw_plan:
        return
        
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
    for station in stations_df:
        station_pos = station[0]
        # 绘制站点水平线
        ax.plot([station_pos, station_pos], [0, time_nodes], [z_min, z_min], 'k--', alpha=0.5)
    
    # 直接根据站点计划时间绘制列车运行线
    # 创建站点位置和时间的列表
    train_space_coords = []
    train_time_coords = []
    
    # 添加始发站出发时间点
    train_space_coords.append(stations_df[0][0])
    train_time_coords.append(stations_df[0][3])  # 出发时间
    
    # 添加中间停靠站的到达和出发时间点
    for station in stations_df[1:-1]:
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
    train_space_coords.append(stations_df[-1][0])
    train_time_coords.append(stations_df[-1][2])  # 到达时间
    
    # 绘制列车运行线（改为浅蓝色虚线）
    ax.plot(train_space_coords, train_time_coords, [z_min] * len(train_space_coords), 
            'c--', alpha=0.9, linewidth=3, label='列车运行线')
    
    # 标记站点的计划时间
    for station in stations_df:
        station_pos = station[0]
        station_type = station[1]
        arrive_time = station[2]
        depart_time = station[3]
        
        # 对于始发站，只标记出发时间
        if station_type == 0 and station_pos == stations_df[0][0]:
            ax.scatter(station_pos, depart_time, z_min, color='red', s=80, marker='o') 
        
        # 对于终点站，只标记到达时间
        elif station_type == 0 and station_pos == stations_df[-1][0]:
            ax.scatter(station_pos, arrive_time, z_min, color='red', s=80, marker='o') 
        
        # 对于停靠站，标记到达和出发时间
        elif station_type == 2:
            station_idx = [s[0] for s in stations_df].index(station_pos) 
            
            ax.scatter(station_pos, arrive_time, z_min, color='green', s=80, marker='o') 
            
            ax.scatter(station_pos, depart_time, z_min, color='blue', s=80, marker='o') 
        
        # 对于通过站，标记通过时间
        elif station_type == 1:
            station_idx = [s[0] for s in stations_df].index(station_pos) 
            
            ax.scatter(station_pos, arrive_time, z_min, color='purple', s=80, marker='o') 
    
    # 添加图例说明
    ax.text(0, time_nodes-1, z_min, "时间-空间二维投影", color='black', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7))
    


def finalize_plot(ax, space_segments, time_nodes, speed_levels, delta_d, delta_t, base_hour=8):
    """
    完善图形显示，设置坐标轴刻度和标签
    
    参数:
        ax: 3D图形对象
        space_segments: 空间分段数量
        time_nodes: 时间节点数量
        speed_levels: 速度级别数量
        delta_d: 空间分辨率(km)
        delta_t: 时间分辨率(分钟)
        base_hour: 基准小时(默认为8点)
    """
    # 设置空间轴的刻度
    space_ticks = np.linspace(0, space_segments, num=6)
    space_tick_labels = [f"{x*delta_d:.1f}" for x in space_ticks]
    ax.set_xticks(space_ticks)
    ax.set_xticklabels(space_tick_labels)
    
    # 设置时间轴的刻度
    time_ticks = np.linspace(0, time_nodes, num=6)
    # 将时间单位转换回实际时间
    base_time = base_hour * 60  # 基准时间(分钟表示)
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
    
    # plt.tight_layout() 
    plt.show()              


def plot_optimal_path(ax, path):
    """
    在三维空间中绘制最优路径及其在各平面上的投影
    
    参数:
        ax: 3D图形对象
        path: 最优路径节点列表，每个节点为(空间,时间,速度)的元组
    """
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
    

def export_train_schedule(path, station_positions, station_names, stations_df):
    """
    根据最优路径导出列车运行时刻表
    
    参数:
        path: 最优路径节点列表
        station_positions: 站点位置列表
        station_names: 站点名称字典
        stations_df: 站点数据
        
    返回:
        station_actual_times: 包含各站点实际到达和出发时间的字典
    """
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
    for station in stations_df:
        station_pos, _, plan_arrive, plan_depart, _ = station
        actual_arrive = station_actual_times.get(station_pos, {}).get("arrive", "-")
        actual_depart = station_actual_times.get(station_pos, {}).get("depart", "-")
        max_speed = station_actual_times.get(station_pos, {}).get("max_speed", 0)
        
        print(f"{station_names[station_pos]}\t{plan_arrive}\t{plan_depart}\t{actual_arrive}\t{actual_depart}\t{max_speed}")
    
    return station_actual_times