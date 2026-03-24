import sys
import os
import numpy as np
import pickle 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection


# load data
res_file = "exp/Ours_submodular/Compare/res_train_on_riverside_Ours.pickle"
res_FT_seq1 = pickle.load(open(res_file, 'rb'))
records = []
x_min, x_max, y_min, y_max = sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min
n_correct, n_incorrect, n_count = 0, 0, 0
for key in res_FT_seq1:
    record = res_FT_seq1[key]
    if n_count % 2 == 0:
        records.append(record)
        if record['pos'][0] < x_min:
            x_min = record['pos'][0]
        if record['pos'][0] > x_max:
            x_max = record['pos'][0]
        if record['pos'][1] < y_min:
            y_min = record['pos'][1]
        if record['pos'][1] > y_max:
            y_max = record['pos'][1]
    n_count += 1
print("Record size: ", len(records))

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0-10, x_max-x_min+10)
ax.set_ylim(0-10, y_max-y_min+10)
ax.set_title('Dynamic Point Set: Even=Green, Odd=Red')
ax.grid(True)

# 初始化空点集和颜色列表
points = np.zeros((0, 2))  # 存储点坐标
colors = []                # 存储颜色值

# 创建散点图对象
scatter = ax.scatter([], [], s=40, alpha=0.7)

def init():
    """初始化函数"""
    scatter.set_offsets(np.empty((0, 2)))
    return scatter,

def update(frame):
    """动画更新函数（每帧调用）"""
    global ani, points, colors, n_correct, n_incorrect
    
    # 生成新点（随机坐标）
    record = records[frame]
    new_point = np.array([record['pos'][0] - x_min, record['pos'][1] - y_min])
    new_point = new_point.reshape(1, -1)
    pt_state = record['top1_state']
    
    # 添加新点到集合
    points = np.vstack([points, new_point])
    
    # 根据top 1检索结果赋色
    if pt_state == 1:
        colors.append('green')
        n_correct += 1
    else:
        colors.append('red')
        n_incorrect += 1
    
    # 更新散点图数据
    scatter.set_offsets(points)
    scatter.set_facecolor(colors)
    
    # title
    ax.set_title('Correct: {}, Incorrect: {}, R@1: {:.2f}%'.format(n_correct, n_incorrect, n_correct*100.0/(n_correct+n_incorrect)))
    
    # 修改动画的间隔
    if frame == 100:
        ani.event_source.interval = 20
    if frame == 500:
        ani.event_source.interval = 10
    if frame == 1000:
        ani.event_source.interval = 5
    
    return scatter,

# 创建动画（每100毫秒添加一个新点，共1000帧)
ani = FuncAnimation(fig, update, frames=1000,
                    init_func=init, blit=True, interval=100)

plt.tight_layout()

filename = os.path.splitext(os.path.basename(res_file))[0]
ani.save('/home/ericxhzou/Code/LifelongPR/exp/Ours_submodular/Compare/{}.mp4'.format(filename), writer='ffmpeg', dpi=100)