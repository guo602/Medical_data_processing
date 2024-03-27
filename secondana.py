import os
import sys
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import argparse

#config
data_root_dir = './data2_3'
save_root_dir = './results'
X_AXIS_LENGTH = 2000
SMOOTH_WIN_SIZE = 200
# SMOOTH_WIN_SIZE = 10
#对比的两个亚组，名称与data_root_dir下的亚组文件夹名称一致
two_subgroups_for_comparison = ['subgroup_1', 'subgroup_2']
absoute_value_diff_threshold = {
    'AngleL': 20,
    'AngleR': 20,
    'CBD': 20,
    'LL': 20,
    'PI': 20,
    'PT': 20,
    'SS': 20,
    'SVA': 20,
    'T1-SPI': 2,
    'T9-SPI': 1,
    'TK': 20,
    'TKL': 20,
    'TPA': 20,
    'ZTSZZ': 20,
}
pearson_corr_diff_threshold = 0.8
pearson_corr_diff_thresholds = {
    'AngleL': 0.7,
    'AngleR': 0.7,
    'CBD': 0.7,
    'LL': 0.7,
    'PI': 0.7,
    'PT': 0.7,
    'SS': 0.7,
    'SVA': 0.7,
    'T1-SPI': 0.7,
    'T9-SPI': 0.7,
    'TK': 0.7,
    'TKL': 0.7,
    'TPA': 0.7,
    'ZTSZZ': 0.7,
    
}


PERIOD_WDITH_FLUCATE_FACTOR = 0.1
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

def getPeriodWidth(peaks):
    period_widths = []
    for i,peak in enumerate(peaks):
        if i < len(peaks) - 1:
            period_widths.append(peaks[i + 1] - peaks[i])
        else:
            break
    median_width = np.median(period_widths)
    return  median_width

    



def findPeaks(data, H=30, D=100):
    # 使用 AngleL 寻找峰值
    data_peak = data['AngleL'].values
    peaks, _ = find_peaks(data_peak, height=H, distance=D)
    _, _, lefts, rights = peak_widths(data_peak, peaks)
    peaks = peaks.astype(int)
    lefts = lefts.astype(int)
    rights = rights.astype(int)
    
    # return rights + 18

    period_width = getPeriodWidth(peaks)
   

    return rights + int(0.1*period_width)


def standardize(data, peak):
    result = {}
    period_width = getPeriodWidth(peak)
    colnames = list(data.columns[1:])
    for factorsname in colnames:
        dat = data[factorsname].values
        max_length = max([len(dat[p:q]) for p, q in zip(peak, peak[1:])])
        normalized_profiles = []
        for i, p in enumerate(peak):
            
            # pp = [p - peak[i-1] for i,p in enumerate(peak) if i]
            if peak[i] >= len(dat):
                continue
            elif i >= len(peak) - 1 or peak[i + 1] > len(dat):
                period = range(p, len(dat))
            else:
                period = range(p, peak[i + 1])

            # if len(period) > (1 + PERIOD_WDITH_FLUCATE_FACTOR) * period_width or len(period) < (1 - PERIOD_WDITH_FLUCATE_FACTOR) * period_width:
            #     continue
            # print(i,p,period)

            # 归一化x轴到0到1的范围
            # print(np.max(period) , np.min(period))
            # print("period",period)
            x_normalized = (period - np.min(period)) / (np.max(period) - np.min(period))
            # 归一化y轴到0到1的范围
            # print(period)
            # print(dat)
            # print(len(dat))
            # print(len(period))
            y_normalized = dat[period]
           
            # 使用插值方法补齐归一化峰型的长度
            f = interp1d(x_normalized, y_normalized, kind='linear')
            # x_new = np.linspace(0, 1, max_length)
            x_new = np.linspace(0, 1, X_AXIS_LENGTH)

            y_new = f(x_new)
            
            # 将归一化峰型数据添加到数组中
            normalized_profiles.append(y_new)
        # 计算所有归一化峰型的平均值和标准差
        # print('len normalized_profiles',len(normalized_profiles))
        mean_profile = np.mean(normalized_profiles, axis=0)
        # print('len mean_profile',len(mean_profile))

        result[factorsname] = mean_profile
        
        # 绘制平均值和标准差的图形
        # plt.plot(x_new, mean_profile, color='red', label='Mean')
        # plt.savefig('1.pdf')
        # sys.exit()
   
    return result

def combine_subgroup_data(subgroup_data):
    if subgroup_data == None:
        print('Error:subgroup_data is None')
        sys.exit()

    # print(subgroup_data)
    
    result = {}
    for factorsname in subgroup_data[0].keys():
        single_metric = []
        for i in range(len(subgroup_data)):
            print(i,subgroup_data[i].keys())
            print(i,factorsname)
            single_metric.append(subgroup_data[i][factorsname])
        single_metric = np.array(single_metric)
        single_metric = np.mean(single_metric,axis = 0)
        result[factorsname] = single_metric
    return result

def plot_subgroups(data,only_one_group):
    x = np.linspace(0, 1, X_AXIS_LENGTH)  
        
    for factorsname in data[list(data.keys())[0]].keys():
        for subgroup_data_i in data.keys():
            # 在同一个 plot() 函数中绘制多条曲线
            y = data[subgroup_data_i][factorsname]
            
            ## 平滑
            # 定义滑动窗口大小
            window_size = SMOOTH_WIN_SIZE
            smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
            first_y = y[0]
            first_window_first_y = smoothed_y[0]
            
            missing_y = np.linspace(first_y, first_window_first_y, num=window_size)
            missing_y = missing_y[:window_size - 1]
            y = np.concatenate((missing_y, smoothed_y))

            window_size_2 = 10
            smoothed_y_2 = np.convolve(y, np.ones(window_size_2)/window_size_2, mode='valid')
            missing_y_2 = y[:window_size_2 - 1]
            y = np.concatenate((missing_y_2, smoothed_y_2))


            
            plt.plot(x, y, label=subgroup_data_i)
            if not only_one_group:
                plt.legend()
            
            plt.title(factorsname)
        # 设置 x 轴刻度线的位置
        plt.xticks([i/10 for i in range(11)])  # 每隔 0.1 设置一个刻度线

        # 获取当前图形对象，并设置 x 轴刻度线的样式为虚线
        ax = plt.gca()  # 获取当前图形对象
        ax.tick_params(axis='x', length=0, width=0)  # 隐藏刻度线的长度和宽度
        ax.grid(axis='x', linestyle='dashed')  # 绘制虚线刻度线
        plt.savefig(f'{save_root_dir}/{factorsname}.pdf')
        plt.close()


def process_and_plot_data(only_one_group):
    if not os.path.exists(save_root_dir):
        print("创建结果保存文件夹：",save_root_dir)
        os.mkdir(save_root_dir)

    subgroup_data_dict = {}
    subgroup_lists = [i for i in os.listdir(data_root_dir) if 'subgroup' in i]
    print('读取亚组数据',subgroup_lists)
    
    for subgroup in subgroup_lists:
        print(f"亚组样本：{subgroup}")
        subgroup_result = []
        patient_lists = [i for i in os.listdir(f'{data_root_dir}/{subgroup}') if  i.endswith('txt')]
        for patient in patient_lists:
            print(f"\t正在处理样本：{patient}")

            dat = pd.read_csv(f'{data_root_dir}/{subgroup}/{patient}', sep='\t', index_col=False)
            print('\t样本长度：',len(dat))
            
            dat['AngleL'] = -dat['AngleL']
            dat['AngleR'] = -dat['AngleR']

            peak = findPeaks(dat, 30, 100)
            standardized_dat = standardize(dat, peak)

           
            subgroup_result.append(standardized_dat)
            
        subgroup_data = combine_subgroup_data(subgroup_result)
        subgroup_data_dict[subgroup] = subgroup_data

    plot_subgroups(subgroup_data_dict,only_one_group)
    return subgroup_data_dict

def exist_diff_between(seq_a,seq_b,ab_th,pearson_th):
    mean_a = np.mean(seq_a)
    mean_b = np.mean(seq_b)
    if abs(mean_a - mean_b) > ab_th:
        return '有差异'
    
    # 使用 pearsonr 函数计算 Pearson 相关系数和 p 值
    correlation, _ = pearsonr(seq_a, seq_b)
    #若相关系数较低则返回 '有差异'
    if abs(correlation) < pearson_th:
        return '有差异'
    else:
        return '无差异'


def AnaDiff(subgroup_data_dict,intervals):
    subgroup_a = two_subgroups_for_comparison[0]
    subgroup_b = two_subgroups_for_comparison[1]
    row_index = list(subgroup_data_dict[list(subgroup_data_dict.keys())[0]].keys())
    col_index = intervals
    df = pd.DataFrame(index=row_index, columns=col_index)
    for row in df.index: 
        #row is metric name like  AngleL
        for idx,col in enumerate(df.columns):
            start = int(col*X_AXIS_LENGTH)
            if idx == len(df.columns) - 1:
                end = X_AXIS_LENGTH
            else:
                end = int(df.columns[idx+1]*X_AXIS_LENGTH)
            
            df.loc[row,col] = exist_diff_between(subgroup_data_dict[subgroup_a][row][start:end],\
                               subgroup_data_dict[subgroup_b][row][start:end],\
                                absoute_value_diff_threshold[row],\
                                pearson_corr_diff_threshold)
    df.to_csv(f'{save_root_dir}/{subgroup_a}_{subgroup_b}_{len(intervals)}intervals_comparision.csv', encoding="utf-8-sig",index=True)




def main(only_one_group):
    subgroup_data_dict = process_and_plot_data(only_one_group)
    if not only_one_group:
        AnaDiff(subgroup_data_dict,[0,0.1,0.5,0.6])
        AnaDiff(subgroup_data_dict,[0,0.1,0.3,0.5,0.6,0.7,0.9])
    
    print('Done')
       
      
       
if __name__ == '__main__':
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数
    parser.add_argument("only_one_group", nargs = "?",default = 0, help="是否只有一个组")

    # 解析命令行参数
    args = parser.parse_args()
    only_one_group = args.only_one_group
    main(only_one_group)
    

