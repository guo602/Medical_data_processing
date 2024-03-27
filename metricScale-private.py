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
from openpyxl import Workbook
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


#config
data_root_dir = './data_scale_private'
save_root_dir = data_root_dir + '/scaled_results_private'
scale_config = {
    "CBD":{
        'min':-90,
        'max':50,
    },
    "SVA":{
        'min':-10,
        'max':70,
    },
    "TK":{
        'min':10,
        'max':30,
    },
    "LL":{
        'min':30,
        'max':60,
    },
    "PT":{
        'min':1,
        'max':20,
    },
    "PI":{
        'min':40,
        'max':60,
    },
    "T1-SPI":{
        'min':1,
        'max':15,
    },
    "T9-SPI":{
        'min':1,
        'max':15,
    },
    "TPA":{
        'min':1,
        'max':15,
    },
    "TKL":{
        'min':-10,
        'max':15,
    },
}
def scale_list(lst, new_min, new_max):
    old_min = min(lst)
    old_max = max(lst)
    if old_min == old_max:
        print("这组数据的最大值和最小值一样无法scale")
        sys.exit()
    scaled_lst = []
    
    for value in lst:
        scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        scaled_lst.append(scaled_value)
    
    return scaled_lst

def scale_data(period_data):

    for metric in scale_config.keys():
        scaler = MinMaxScaler(feature_range=(scale_config[metric]['min'], scale_config[metric]['max']))
        period_data[metric] = scaler.fit_transform(period_data[[metric]])
    return period_data


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

def generate_split_excel_for_each_patient(data,patient_name,split_intervals):

    peak = findPeaks(data, 30, 100)

    excel_data = [['frameInd','AngleL','AngleR','CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPI','TPA', 'ZTSZZ','TKL']]
    period_id = 1
    for i, p in enumerate(peak):
        if peak[i] > len(data):
                continue
        elif i >= len(peak) - 1 or peak[i + 1] > len(data):
            period = range(p, len(data))
        else:
            period = range(p, peak[i + 1])
        excel_data.append([f"period_{period_id}"])
        frame_id = 1
        
        frame_length = len(period)
        period_data = data.iloc[period]
        
        period_data = scale_data(period_data)
        for idx,interval_i in enumerate(split_intervals):
            start = split_intervals[idx]
            if idx==len(split_intervals)-1:
                end = 1
            else:
                end = split_intervals[idx + 1]
            excel_data.append([f"{start}-{end}"])

            start_int = int(start * frame_length)
            end_int = int(end * frame_length)
            for i in range(start_int,end_int,1):
                row = []
                # row.append(frame_id)
                for _,metric in enumerate([ 'frameInd','AngleL','AngleR','CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPI','TPA', 'ZTSZZ','TKL']):
                    if metric == 'frameInd' and metric not in period_data.columns:
                        metric = 'No.'
                    elif metric == 'AngleL' :
                        row.append(-1 * period_data[metric].iloc[i])
                        continue
                    row.append(period_data[metric].iloc[i])
                excel_data.append(row)
                
                frame_id += 1
        period_id += 1
    # print(excel_data)

    # 创建一个新的工作簿
    workbook = Workbook()

    # 选择默认的活动工作表
    sheet = workbook.active

    for row in excel_data:
        sheet.append(row)

    # 保存工作簿为Excel文件
    workbook.save(save_root_dir + f'/{patient_name}_split.xlsx')

def generate_combine_excel_for_each_patient(data,patient_name,split_intervals):

    peak = findPeaks(data, 30, 100)

    excel_data = [['frame','AngleL','AngleR', 'CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPi','TPA','ZTSZZ', 'TKL']]
    period_id = 1
    for i, p in enumerate(peak):
        if peak[i] > len(data):
                continue
        elif i >= len(peak) - 1 or peak[i + 1] > len(data):
            period = range(p, len(data))
        else:
            period = range(p, peak[i + 1])
        # excel_data.append([f"period_{period_id}"])
        frame_id = 1
        
        frame_length = len(period)
        period_data = data.iloc[period]
        
        period_data = scale_data(period_data)
        for idx,interval_i in enumerate(split_intervals):
            start = split_intervals[idx]
            if idx==len(split_intervals)-1:
                end = 1
            else:
                end = split_intervals[idx + 1]
            # excel_data.append([f"{start}-{end}"])

            start_int = int(start * frame_length)
            end_int = int(end * frame_length)
            for i in range(start_int,end_int,1):
                row = []
                # row.append(frame_id)
                for _,metric in enumerate([ 'frameInd','AngleL','AngleR','CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPI','TPA', 'ZTSZZ','TKL']):
                    if metric == 'frameInd' and metric not in period_data.columns:
                        metric = 'No.'
                    elif metric == 'AngleL' :
                        row.append(-1 * period_data[metric].iloc[i])
                        continue
                    row.append(period_data[metric].iloc[i])
                excel_data.append(row)
                
                frame_id += 1
        period_id += 1
    # print(excel_data)

    # 创建一个新的工作簿
    workbook = Workbook()

    # 选择默认的活动工作表
    sheet = workbook.active

    for row in excel_data:
        sheet.append(row)

    # 保存工作簿为Excel文件
    workbook.save(save_root_dir + f'/{patient_name}_combine.xlsx')
    



def changeAngle(data):
    
    data['AngleL'] = -data['AngleL']
    return data

def main():
    if not os.path.exists(save_root_dir):
        os.mkdir(save_root_dir)
    print("Data scaled results will be saved in: {}".format(save_root_dir))

    patient_lists = [i for i in os.listdir(f'{data_root_dir}/') if  i.endswith('txt')]
    for patient in patient_lists:
        print(f"\t正在处理样本：{patient}")

        dat = pd.read_csv(f'{data_root_dir}/{patient}', sep='\t', index_col=False)
        print('\t样本长度：',len(dat))

        patient_name = patient.replace(".txt",'')

        
        neg_dat = changeAngle(dat)

        generate_split_excel_for_each_patient(neg_dat,patient_name,[0,0.1,0.3,0.5,0.6])
        generate_combine_excel_for_each_patient(neg_dat,patient_name,[0,0.1,0.3,0.5,0.6])


      



if __name__ == '__main__':
    main()