import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


class IncrementalTracker:
    def __init__(self):
        self.most_recent = {}
        self.greatest_past = {}
        self.start_indexes = {} # Recall@1 for latest 
        self.seen_envs = []
        self.inc_metrics = {}

    def update(self, update_dict, env_idx):
        self.inc_metrics[env_idx] = update_dict
        for k,v in update_dict.items():
            if k not in self.seen_envs:
                self.seen_envs.append(k)
                self.most_recent[k] = v  
                self.greatest_past[k] = np.nan 
                self.start_indexes[k] = env_idx 
            else:
                self.greatest_past[k] = self.most_recent[k] if np.isnan(self.greatest_past[k]) else max(self.greatest_past[k], v)
                self.most_recent[k] = v 

    def get_results(self):
        # Get recall and forgetting
        results = {}
        for k in self.start_indexes:
            results[k] = {}
            results[k]['Recall@1'] = self.most_recent[k]
            if k in self.greatest_past:
                results[k]['Forgetting'] = self.greatest_past[k] - self.most_recent[k]
            else:
                results[k]['Forgetting'] = np.nan
        
        # Merge
        results_merged = {} 
        for v in self.start_indexes.values():
            merge_keys = [k for k in self.start_indexes if self.start_indexes[k] == v] # Get keys which should be merged 
            new_key = '/'.join(merge_keys) # Get new key 
            merged_recall = np.mean([results[m]['Recall@1'] for m in merge_keys])
            merged_forgetting = np.mean([results[m]['Forgetting'] for m in merge_keys])

            results_merged[new_key] = {'Recall@1': merged_recall, 'Forgetting': merged_forgetting}
        
        # Average Incremental Recall@1
        AIR_1 = []
        for k_i in self.inc_metrics:
            AR_1 = []
            for k_j in self.inc_metrics[k_i]:
                AR_1.append(self.inc_metrics[k_i][k_j])
            AIR_1.append(np.mean(AR_1))
        AIR_1 = np.mean(AIR_1)

        # Print 
        results_final = pd.DataFrame(columns = ['Recall@1', 'Forgetting'])  # i.e. AR@1 / F
        for k in results_merged:
            results_final.loc[k] = [results_merged[k]['Recall@1'], results_merged[k]['Forgetting']]
        results_final.loc['Average (Recall@1, Forgetting)'] = results_final.mean(0)
        results_final.loc['Average Incremeantal Recall@1'] = AIR_1
        return results_final
    
    def vis_grid_plot(self, filepath, font_size=16):
        end_env_idx = np.max(list(self.inc_metrics.keys()))
        assert len(self.inc_metrics[end_env_idx].keys()) == end_env_idx+1
        envs = list(self.inc_metrics[end_env_idx].keys())
        
        # 创建数据
        x = np.linspace(1, end_env_idx+1, end_env_idx+1)
        y = np.linspace(1, end_env_idx+1, end_env_idx+1)
        X, Y = np.meshgrid(x, y)
        Z = np.full((end_env_idx+1, end_env_idx+1), np.nan)
        for i in range(end_env_idx+1):
            for j in range(len(self.inc_metrics[i])):
                Z[i,j] = self.inc_metrics[i][envs[j]]
        Z = Z.T

        # 绘制栅格图
        plt.clf()
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('Greens')
        cmap.set_bad('whitesmoke', alpha=1.0)  # 设置nan值为白色
        cax = ax.pcolormesh(X, Y, Z, vmin=60, vmax=100, cmap=cmap)
        colorbar = fig.colorbar(cax)  # 添加颜色条
        colorbar.ax.tick_params(labelsize=font_size)
        
        ax.invert_yaxis()
        
        plt.xticks(ticks=np.arange(end_env_idx+1)+1, fontsize=font_size)  # 默认字体大小为10
        plt.yticks(np.arange(len(envs))+1, envs, fontsize=font_size)

        # 显示格子内的数字
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                text = '-' if np.isnan(Z[i, j]) else '{:.2f}'.format(Z[i, j])
                ax.text(X[i, j], Y[i, j], text, ha='center', va='center', color='black', fontsize=font_size+1)

        # 调整坐标轴以显示文本标签
        plt.tight_layout()
        
        ax.set_aspect('equal')

        # 显示图形
        plt.savefig(filepath)


if __name__ == '__main__':
    step_0 = {'Oxford': 93.8}
    step_1 = {'Oxford': 84.09552543691579, 'DCC': 81.06894504403417}
    step_2 = {'Oxford': 83.91506599787158, 'DCC': 77.28090467849917, 'Riverside': 82.6737801326445}
    step_3 = {'Oxford': 82.93509829383399, 'DCC': 68.0391049140116, 'Riverside': 76.65990835042547, 'In-house': 95.75022601898989}

    metrics = IncrementalTracker()

    metrics.update(step_0, 0)
    metrics.update(step_1, 1)
    metrics.update(step_2, 2)
    metrics.update(step_3, 3)

    r = metrics.get_results()
    print(r)
    
    metrics.vis_grid_plot(filepath='/home/ericxhzou/Code/InCloud/exp/metrics.png')
