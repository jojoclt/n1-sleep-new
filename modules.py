from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import logging

# This function has been modified to make use of numpy and optimized to be faster.
def n1_multi_range(signal_df, thres = 25, CONSECUTIVE_STEPS = 50, total_size = 1500, DEBUG=False):
    index = []
    start = -1
    interval = 0
    N = signal_df.shape[0]
    # 1 STEP = 0.01s (100Hz)
    # i -> sliding window in 1500 steps
    data = abs(signal_df).to_numpy()
    for i in range(0, N, 3000):
        count = 0
        inner_count = 0
        # Going to try consecutive 50 steps (0.5s) and check if the signal is below a threshold
        # If it is in 50 steps, then the count will be increased by 50 and so on...
        values = data[i:min(N, i + 3000)] < thres
        for ele in values:
            if ele:
                inner_count += 1
            else:
                if inner_count >= CONSECUTIVE_STEPS-1:
                    count += inner_count
                inner_count = 0

        if (DEBUG):
            print(count+inner_count,)

        if count + inner_count >= total_size:
            if interval == 0:
                start = i
            interval += 1
        
        else:
            if interval == 0:
                pass
            elif interval == 1:
                index.append((start,start+3000))
            else:
                index.append((start,i))
            interval = 0
    if (DEBUG):
        print("--------")
    return index

def get_n1_eeg(signal_df, eeg_index, eeg2_index, eog_index, psg_id):
    start = pd.to_datetime(signal_df.index[0])
        
    def annotate_sleep_stage_predict(signal_df, eeg_index):
        mask = np.zeros(len(signal_df), dtype=int)
        for interval in eeg_index:
            start_index, end_index = interval
            start_time = start + pd.to_timedelta(start_index / 100, unit='seconds')
            end_time = start + pd.to_timedelta(end_index / 100, unit='seconds')
            interval_mask = (signal_df.index >= start_time) & (signal_df.index < end_time)
            mask |= interval_mask

        return mask

    if len(eeg_index) == 0:
        print("EEG_INDEX_NOT_FOUND at", psg_id)
        pass
    else:
        signal_df['N1_predict_EEG'] = annotate_sleep_stage_predict(signal_df, eeg_index)

    if eeg2_index is None:
        pass
    elif len(eeg2_index) == 0:
        print("EEG2_INDEX_NOT_FOUND at", psg_id)
        pass
    else:
        signal_df['N1_predict_EEG2'] = annotate_sleep_stage_predict(signal_df, eeg2_index)
    
    if len(eog_index) == 0:
        print("EOG_NOT_FOUND at", psg_id)
        pass
    else:
        signal_df['N1_predict_EOG'] = annotate_sleep_stage_predict(signal_df, eog_index)

    try:
        not_empty_EEG = (signal_df.get('N1_predict_EEG') != 0).any()
    except Exception as e:
        # print("EEG_ERROR",e)
        not_empty_EEG = False
    try:
        not_empty_EEG2 = (signal_df.get('N1_predict_EEG2') != 0).any()
    except Exception as e:
        # print("EEG2_ERROR",e)
        not_empty_EEG2 = False
    try:
        not_empty_EOG = (signal_df.get('N1_predict_EOG') != 0).any()
    except Exception as e:
        # print("EOG_ERROR",e)
        not_empty_EOG = False

    signal_df['N1_predict'] = np.ones(len(signal_df), dtype=int)

    eeg_con, eeg2_con, eog_con = None, None, None

    # 1. normal methods
    if (not_empty_EEG and not_empty_EEG2):
        eeg_con = signal_df['N1_predict_EEG'][signal_df['N1_predict_EEG']==int(1)]
        eeg2_con = signal_df['N1_predict_EEG2'][signal_df['N1_predict_EEG2']==int(1)]
        signal_df['N1_predict'] &= eeg_con & eeg2_con

    elif (not_empty_EEG):
        eeg_con = signal_df['N1_predict_EEG'][signal_df['N1_predict_EEG']==int(1)]
        signal_df['N1_predict'] &= eeg_con
        
    elif (not_empty_EEG2):
        eeg2_con = signal_df['N1_predict_EEG2'][signal_df['N1_predict_EEG2']==int(1)]
        signal_df['N1_predict'] &= eeg2_con

    if (not_empty_EOG):
        eog_con = signal_df['N1_predict_EOG'][signal_df['N1_predict_EOG']==int(1)]
        signal_df['N1_predict'] &= eog_con
    
    try:
        length_pred = len(signal_df['N1_predict'][signal_df['N1_predict']==int(1)])
        # 2. if normal methods fail, use eeg2 & eog
        if length_pred == 0:
            signal_df['N1_predict'] = eog_con & eeg2_con
            length_pred = len(signal_df['N1_predict'][signal_df['N1_predict']==int(1)])
            if length_pred == 0:
                signal_df['N1_predict'] = signal_df['N1_predict_EOG']
    except:
        pass

    print()
    return signal_df

def lol(signal_df, write_folder, psg_id, save_fig=False):
    num_columns = len(signal_df.columns)

    fig, axes = plt.subplots(num_columns, 1, figsize=(20, 12), sharex=True)

    # 迴圈遍歷每個欄位
    for i, column in enumerate(signal_df.columns):
        # 取得目前的軸
        ax = axes[i]

        # 繪製折線圖
        ax.plot(signal_df.index, signal_df[column])
        
        # 繪製虛線
        start_time = signal_df.index[0]
        end_time = signal_df.index[-1]
        interval = pd.Timedelta(seconds=30)
        current_time = start_time + interval
        while current_time < end_time:
            ax.axvline(x=current_time, linestyle='--', color='gray')
            current_time += interval

        # 設定軸的標籤
        ax.set_ylabel(column)
        loc = mdates.MinuteLocator(interval=1)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # 設定圖表標題和共用 x 軸標籤
    fig.suptitle('Signal Visualization')
    axes[-1].set_xlabel('Time')
    # 調整子圖之間的間距
    plt.tight_layout()
    plt.ylim(0,10)
    # 顯示圖表
    if save_fig:
        plt.savefig(write_folder + psg_id.split('.')[0],bbox_inches='tight')
    else:
        plt.show()
    plt.close()

# Condition
# 1. predict start = n1 start
# 2. in N1
# 3. +- 1 min
def accuracy_method_1(signal_df):
    cond_pred = signal_df['N1_predict'][signal_df['N1_predict']==int(1)]
    cond_true = signal_df['N1'][signal_df['N1']==int(1)]
    return cond_pred.index[0] == cond_true.index[0]

def accuracy_method_2(signal_df):
    cond_pred = signal_df['N1_predict'][signal_df['N1_predict']==int(1)]
    cond_true = signal_df['N1'][signal_df['N1']==int(1)]
    arr = cond_pred & cond_true
    return (len(arr[arr]) > 0) and (cond_pred.index[0] >= cond_true.index[0])

def accuracy_method_3(signal_df):
    cond_pred = signal_df['N1_predict'][signal_df['N1_predict']==int(1)]
    cond_true = signal_df['N1'][signal_df['N1']==int(1)]
    # extend cond_true by +- 1 min
    s = cond_true.index[0] - pd.Timedelta(minutes=1)
    e = cond_true.index[-1] + pd.Timedelta(minutes=1)
    cond_true[s:e] = True
    arr = cond_pred & cond_true
    return (len(arr[arr]) > 0) and (cond_pred.index[0] >= cond_true.index[0])
