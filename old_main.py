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


def viewData(name, dat, factor='AngleL'):
    dat = dat[factor].values
    diff1 = numpy.diff(dat)
    _, axs = plt.subplots(2, 1, figsize=(20, 6))
    ax, ax2 = axs
    ax.plot(dat, color='red', label=factor)
    ax2.plot(diff1)
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


def findPeaks(data, H=30, D=100):
    # 使用 AngleL 寻找峰值
    data_peak = data['AngleL'].values
    peaks, _ = find_peaks(data_peak, height=H, distance=D)
    _, _, lefts, rights = peak_widths(data_peak, peaks)
    peaks = peaks.astype(int)
    lefts = lefts.astype(int)
    rights = rights.astype(int)
    return rights + 18


def plotBySample(name, data, peak):
    colnames = list(data.columns[1:])
    # 保存两张图到一个PDF文件
    with PdfPages(f'Data/{name}_所有指标.pdf') as pdf:
        for factorsname in colnames:
            dat = data[factorsname].values
            max_length = max([len(dat[p:q]) for p, q in zip(peak, peak[1:])])
            normalized_profiles = []
            for i, p in enumerate(peak):

                if i == len(peak) - 1:
                    period = range(p, len(dat))
                else:
                    period = range(p, peak[i + 1])
                # 归一化x轴到0到1的范围
                x_normalized = (period - np.min(period)) / (np.max(period) - np.min(period))
                # 归一化y轴到0到1的范围
                y_normalized = (dat[period] - np.min(dat[period])) / (np.max(dat[period]) - np.min(dat[period]))
                # 使用插值方法补齐归一化峰型的长度
                f = interp1d(x_normalized, y_normalized, kind='linear')
                x_new = np.linspace(0, 1, max_length)
                y_new = f(x_new)
                plt.plot(x_new, y_new, label=f'Period {i+1}')
                # 将归一化峰型数据添加到数组中
                normalized_profiles.append(y_new)

            plt.xlabel('time ')
            plt.ylabel('value')
            plt.title(factorsname)

            plt.legend()
            pdf.savefig()
            plt.close()

            # 计算所有归一化峰型的平均值和标准差
            mean_profile = np.mean(normalized_profiles, axis=0)
            std_profile = np.std(normalized_profiles, axis=0)

            # 绘制平均值和标准差的图形
            plt.plot(x_new, mean_profile, color='red', label='Mean')
            plt.fill_between(x_new, mean_profile - std_profile, mean_profile + std_profile, color='gray', alpha=0.3, label='Standard Deviation')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.title(factorsname)
            plt.legend()
            pdf.savefig()
            plt.close()


if __name__ == '__main__':
    file_list = [i for i in os.listdir('Data') if i.endswith('.txt')]
    for sample in file_list:
        print(f"正在处理样本：{sample}")
        dat = pandas.read_csv(f'Data/{sample}', sep='\t', index_col=False)
        dat['AngleL'] = -dat['AngleL']
        dat['AngleR'] = -dat['AngleR']
        viewData(sample[:3], dat)
        peak = findPeaks(dat, 30, 100)
        plotBySample(sample[:3], dat, peak)
