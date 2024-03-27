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


#config
data_root_dir = './data_split'
save_root_dir = data_root_dir + '/results'

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
    # print(f"data_peak{data_peak}")
    peaks, _ = find_peaks(data_peak, height=H, distance=D)
    # print(f"peaks{peaks}")
    _, _, lefts, rights = peak_widths(data_peak, peaks)
    peaks = peaks.astype(int)
    lefts = lefts.astype(int)
    rights = rights.astype(int)
    
    # return rights + 18

    period_width = getPeriodWidth(peaks)
   

    return rights + int(0.1*period_width)

def generate_split_excel_for_each_patient(data,patient_name,split_intervals):
    peak = findPeaks(data, 30, 100)

    excel_data = [['frame', 'original-frame','CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPI','TPA', 'ZTSZZ', 'TKL']]
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
        # print(frame_length)
        # print(len(period_data))
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
                row.append(frame_id)
                for _,metric in enumerate([ 'frameInd','CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPI','TPA', 'ZTSZZ', 'TKL']):
                    if metric == 'frameInd' and metric not in period_data.columns:
                        metric = 'No.'
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

    excel_data = [[ 'frameInd','CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPI','TPA', 'ZTSZZ', 'TKL']]
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
        # print(frame_length)
        # print(len(period_data))
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
                for _,metric in enumerate([ 'frameInd','CBD','SVA', 'TK','LL', 'PT','PI','SS','T1-SPI', 'T9-SPI','TPA', 'ZTSZZ', 'TKL']):
                    if metric == 'frameInd' and metric not in period_data.columns:
                        metric = 'No.'
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
    print("Data split results will be saved in: {}".format(save_root_dir))

    patient_lists = [i for i in os.listdir(f'{data_root_dir}/') if  i.endswith('txt')]
    for patient in patient_lists:
        print(f"\t正在处理样本：{patient}")

        dat = pd.read_csv(f'{data_root_dir}/{patient}', sep='\t', index_col=False)
        print('\t样本长度：',len(dat))

        patient_name = patient.replace(".txt",'')

        dat = changeAngle(dat)
        generate_split_excel_for_each_patient(dat,patient_name,[0,0.1,0.3,0.5,0.6])
        generate_combine_excel_for_each_patient(dat,patient_name,[0,0.1,0.3,0.5,0.6])

      



if __name__ == '__main__':
    main()