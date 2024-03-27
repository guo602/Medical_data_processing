import os

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import argparse

PERIOD_WDITH_FLUCATE_FACTOR = 0.1
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
PLOT_PER_ROW = 4 #画单个周期的多子图每行显示多少个子图
# SINGLW_PERIOD_PLOT_SIZE = (18,6) #画单个周期的多子图全图大小

def getPeriodWidth(peaks):
    period_widths = []
    for i,peak in enumerate(peaks):
        if i < len(peaks) - 1:
            period_widths.append(peaks[i + 1] - peaks[i])
        else:
            break
    median_width = np.median(period_widths)
    return  median_width

    


def viewData(name, dat, factor='AngleL'):
    dat = dat[factor].values
    diff1 = numpy.diff(dat)
    
    _, axs = plt.subplots(2, 1, figsize=(20, 6))
    ax, ax2 = axs
    ax.plot(dat, color='red', label=factor)
    ax2.plot(diff1)
    
    if len(dat) > 10000:
        ax.set_xlim(10000, 15000)
        ax2.set_xlim(10000, 15000)
    ax2.set_ylim(-2, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.grid()
    ax.legend()
    #plt.show()
    plt.savefig(f"Data/{name}_观察_{factor}.pdf")
    plt.close()

def viewProcessedData(name, data, peak,factor='AngleL'):
    origin_dat = data[factor].values
    processed_dat = []

    _, axs = plt.subplots(3, 1, figsize=(36, 6))
    ax1, ax2 ,ax3= axs
    #plot origin data
    ax1.plot(origin_dat, color='red', label=factor)
    ax1.set_xlim(0, 5000)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid()
    ax1.legend()


    # plot proceessed data
    period_width = getPeriodWidth(peak)
    

    ax2.plot(origin_dat, color='#FFC0C0', linestyle='--',label='')
    start = 0
    for i, p in enumerate(peak):
        # if i == len(peak) - 1:
        #     period = range(p, len(origin_dat))
        # else:
        #     period = range(p, peak[i + 1])
        if peak[i] > len(origin_dat):
                continue
        elif i >= len(peak) - 1 or peak[i + 1] > len(origin_dat):
            period = range(p, len(origin_dat))
        else:
            period = range(p, peak[i + 1])
        

        if len(period) > (1 + PERIOD_WDITH_FLUCATE_FACTOR) * period_width or len(period) < (1 - PERIOD_WDITH_FLUCATE_FACTOR) * period_width:
            start += len(period)
            continue
        else:
            ax2.plot([x for x in period]  ,origin_dat[period], color='red',label='')
            start += len(period)
    ax2.set_xlim(0, 5000)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid()

    for i, p in enumerate(peak):

        if peak[i] > len(origin_dat):
                continue
        elif i >= len(peak) - 1 or peak[i + 1] > len(origin_dat):
            period = range(p, len(origin_dat))
        else:
            period = range(p, peak[i + 1])

        # if i == len(peak) - 1:
        #     period = range(p, len(origin_dat))
        # else:
        #     period = range(p, peak[i + 1])

        if len(period) > (1 + PERIOD_WDITH_FLUCATE_FACTOR) * period_width or len(period) < (1 - PERIOD_WDITH_FLUCATE_FACTOR) * period_width:
            continue
        else:
            processed_dat.extend(origin_dat[period])


    ax3.plot(processed_dat, color='red', label=factor)
    # ax2.set_xlim(10000, 15000)
    ax3.set_xlim(0, 5000)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
   
    ax3.grid()
    ax3.legend()

    # ax3.legend()
    ax1.set_title('原始数据')
    ax2.set_title('取中的数据（虚线代表被删除的）')
    ax3.set_title('取中的数据直接横向拼在一起')

    plt.savefig(f'Data/{name}_观察处理后的_{factor}.pdf') 
    plt.close()

    


def findPeaks(data, H=30, D=100):
    # 使用 AngleL 寻找峰值
    data_peak = data['AngleL'].values
    peaks, _ = find_peaks(data_peak, height=H, distance=D)
    _, _, lefts, rights = peak_widths(data_peak, peaks)
    peaks = peaks.astype(int)
    lefts = lefts.astype(int)
    rights = rights.astype(int)

    
    return rights + 18

    period_width = getPeriodWidth(peaks)
    print(0.1*period_width)

    return rights + int(0.1*period_width)


def plotBySample(name, data, peak):
    period_width = getPeriodWidth(peak)
    colnames = list(data.columns[1:])
    # 保存两张图到一个PDF文件
    with PdfPages(f'Data/{name}_所有指标.pdf') as pdf:
        for factorsname in colnames:
            dat = data[factorsname].values
            max_length = max([len(dat[p:q]) for p, q in zip(peak, peak[1:])])
            normalized_profiles = []
            original_profiles = []
            # print("peak:",peak)

            #单个个周期画图
            subplot_data_list = []
           
            for i, p in enumerate(peak):
                
                if peak[i] > len(dat):
                        continue
                elif i >= len(peak) - 1 or peak[i + 1] > len(dat):
                    period = range(p, len(dat))
                else:
                    period = range(p, peak[i + 1])
        
                # if i == len(peak) - 1:
                #     period = range(p, len(dat))
                # else:
                #     period = range(p, peak[i + 1])

                if len(period) > (1 + PERIOD_WDITH_FLUCATE_FACTOR) * period_width or len(period) < (1 - PERIOD_WDITH_FLUCATE_FACTOR) * period_width:
                    continue

                # print(f"{factorsname} {period}")

                # 归一化x轴到0到1的范围
                x_normalized = (period - np.min(period)) / (np.max(period) - np.min(period))
                # 归一化y轴到0到1的范围
                y_normalized = (dat[period] - np.min(dat[period])) / (np.max(dat[period]) - np.min(dat[period])+1e-6)
                #反归一化数据
                y_original= dat[period]
                # print(x_normalized)
                # print(y_normalized)
                # 使用插值方法补齐归一化峰型的长度
                f = interp1d(x_normalized, y_normalized, kind='linear')
             

                x_new = np.linspace(0, 1, max_length)
                y_new = f(x_new)
                result = {'x':x_new,
                          'y':y_new,
                          'title':f'Period {i+1}'
                }
                subplot_data_list.append(result)

            if len(subplot_data_list) == 0:
                print(f"x_new 未定义，可能是由于{name}该组数据周期不稳定导致")
                return

             # 计算子图的行数和列数
            num_plots = len(subplot_data_list)
            num_rows = int(num_plots/PLOT_PER_ROW) + int(num_plots%PLOT_PER_ROW > 0)  # 每行最多显示三个子图
            num_cols = min(PLOT_PER_ROW, num_plots)  # 每列最多显示三个子图
            # 创建画布和子图
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_rows,4*num_cols))
            # 获取默认颜色循环顺序
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            # 遍历数据并绘制子图
            for i, ax in enumerate(axes.flat):
                if i < num_plots:
                    ax.plot(subplot_data_list[i]['x'],subplot_data_list[i]['y']
                            ,label = subplot_data_list[i]['title']
                            ,color = default_colors[i%len(default_colors)])
                    # ax.set_title(subplot_data_list[i]['title'])
                    ax.legend()
                else:
                    # 如果子图数量超过实际数据数量，则隐藏多余的子图
                    ax.axis('off')

            # 自动调整子图之间的间距
            plt.tight_layout()
            # plt.xlabel('time ')
            # plt.ylabel('value')
            fig.suptitle(factorsname,color = 'blue')
            # 调整子图布局
            fig.subplots_adjust(top=0.95) 
            # plt.title(factorsname)
            pdf.savefig()
            plt.close()
              
            for i, p in enumerate(peak):
                
                if peak[i] > len(dat):
                        continue
                elif i >= len(peak) - 1 or peak[i + 1] > len(dat):
                    period = range(p, len(dat))
                else:
                    period = range(p, peak[i + 1])
        
                # if i == len(peak) - 1:
                #     period = range(p, len(dat))
                # else:
                #     period = range(p, peak[i + 1])

                if len(period) > (1 + PERIOD_WDITH_FLUCATE_FACTOR) * period_width or len(period) < (1 - PERIOD_WDITH_FLUCATE_FACTOR) * period_width:
                    continue

                # print(f"{factorsname} {period}")

                
                

                # 归一化x轴到0到1的范围
                x_normalized = (period - np.min(period)) / (np.max(period) - np.min(period))
                # 归一化y轴到0到1的范围
                y_normalized = (dat[period] - np.min(dat[period])) / (np.max(dat[period]) - np.min(dat[period])+1e-6)
                #反归一化数据
                y_original= dat[period]
                # print(x_normalized)
                # print(y_normalized)
                # 使用插值方法补齐归一化峰型的长度
                f = interp1d(x_normalized, y_normalized, kind='linear')
                f2 = interp1d(x_normalized, y_original, kind='linear')

                x_new = np.linspace(0, 1, max_length)
                y_new = f(x_new)
                y_ori_new = f2(x_new)
                plt.plot(x_new, y_new, label=f'Period {i+1}')
                # 将归一化峰型数据添加到数组中
                normalized_profiles.append(y_new)
                # 将反归一化峰型数据添加到数组中
                original_profiles.append(y_ori_new)

            plt.xlabel('time ')
            plt.ylabel('value')
            plt.title(factorsname)

            plt.legend()
            pdf.savefig()
            plt.close()


            # 计算所有归一化峰型的平均值和标准差
            mean_profile = np.mean(normalized_profiles, axis=0)
            std_profile = np.std(normalized_profiles, axis=0)

            # 计算所有反归一化峰型的平均值和标准差
            ori_mean_profile = np.mean(original_profiles, axis=0)
            ori_std_profile = np.std(original_profiles, axis=0)

            # 绘制平均值和标准差的图形
            try:
                max_index = np.argmax(mean_profile)
                min_index = np.argmin(mean_profile)
                y = mean_profile
                x = x_new
                plt.annotate(f'Max:{max(mean_profile):.2f}', xy=(x[max_index], y[max_index]), xytext=(x[max_index]+0.05, y[max_index]+0.05),arrowprops=dict(arrowstyle='->'))
                plt.annotate(f'Min:{min(mean_profile):.2f}', xy=(x[min_index], y[min_index]), xytext=(x[min_index]+0.05, y[min_index]-0.05),arrowprops=dict(arrowstyle='->'))
                plt.plot(x_new, mean_profile, color='red', label='Mean')
                plt.fill_between(x_new, mean_profile - std_profile, mean_profile + std_profile, color='gray', alpha=0.3, label='Standard Deviation')
                plt.xlabel('time')
                plt.ylabel('value')
                plt.title(factorsname)
                plt.legend()
                pdf.savefig()
                plt.close()
            except UnboundLocalError as e:
                print(e)
                print(f"x_new 未定义，可能是由于{name}该组数据周期不稳定导致")
                return
            
            # 绘制反归一化的图形
            try:
                max_index = np.argmax(ori_mean_profile)
                min_index = np.argmin(ori_mean_profile)
                y = ori_mean_profile
                x = x_new
                plt.annotate(f'Max:{max(ori_mean_profile):.2f}', xy=(x[max_index], y[max_index]), xytext=(x[max_index]+0.05, y[max_index]+0.05),arrowprops=dict(arrowstyle='->'))
                plt.annotate(f'Min:{min(ori_mean_profile):.2f}', xy=(x[min_index], y[min_index]), xytext=(x[min_index]+0.05, y[min_index]-0.05),arrowprops=dict(arrowstyle='->'))
                plt.plot(x_new, ori_mean_profile, color='red', label='Mean')
                plt.fill_between(x_new, ori_mean_profile - ori_std_profile, ori_mean_profile + ori_std_profile, color='gray', alpha=0.3, label='Standard Deviation')
                plt.xlabel('time')
                plt.ylabel('value')
                plt.title(factorsname)
                plt.legend()
                pdf.savefig()
                plt.close()
            except UnboundLocalError as e:
                print(e)
                print(f"x_new 未定义，可能是由于{name}该组数据周期不稳定导致")
                return
            


def main(frame_start_id,frame_end_id):
    file_list = [i for i in os.listdir('Data') if i.endswith('.txt')]
    for sample in file_list:
        print(f"正在处理样本：{sample}")
        # dat = pandas.read_csv(f'Data/{sample}', sep='\t', index_col=False)
        dat = pandas.read_csv(f'data/{sample}', sep='\t', index_col=False)


        frame_start = 0
        frame_end   = 9223372036854775806

        if frame_start_id!= -1:
            frame_start = frame_start_id
        if frame_end_id != -1:
            frame_end = frame_end_id
        # try:
        if 'frameInd' in dat.columns:
            dat = dat[(dat['frameInd'] >= frame_start) & (dat['frameInd'] <= frame_end)]
        else:
            dat = dat[(dat['No.'] >= frame_start) & (dat['No.'] <= frame_end)]


        dat['AngleL'] = -dat['AngleL']
        dat['AngleR'] = -dat['AngleR']
        viewData(sample[:3], dat)
        peak = findPeaks(dat, 30, 100)
        plotBySample(sample[:3], dat, peak)
        viewProcessedData(sample[:3], dat, peak)
        # except Exception as e:
        #     print(f"处理样本{sample}时出错,可能是起始桢和结束桢间隔过短导致：{e}")

if __name__ == '__main__':
    # 创建解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数
    parser.add_argument("frame_start_id", nargs = "?",default = -1, help="开始桢id")
    parser.add_argument("frame_end_id", nargs = "?",default = -1, help="结束桢id")

    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数并传入参数
    main(int(args.frame_start_id), int(args.frame_end_id))
    
    


