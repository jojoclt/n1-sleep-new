import pyedflib # ref: https://pyedflib.readthedocs.io/en/latest/
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
from modules import *
from tqdm import tqdm
import logging


def main_function(eeg_thres, eeg2_thres, eog_thres):
    # the format of plot will be THRES/CONSEC/TOTAL_SIZZE
    # EEG_THRES = 25
    EEG_CONSEC = 300
    EEG_TOTAL_SIZE = 1500

    # EEG2_THRES = 20
    EEG2_CONSEC = 500
    EEG2_TOTAL_SIZE = 1500

    # EOG_THRES = 120
    EOG_CONSEC = 2500
    EOG_TOTAL_SIZE = 2500
    extend = 30

    EEG_THRES = eeg_thres
    EEG2_THRES = eeg2_thres
    EOG_THRES = eog_thres
    
    def reprval(text):
        thres = eval(f"{text}_THRES")
        consec = eval(f"{text}_CONSEC")
        total_size = eval(f"{text}_TOTAL_SIZE")
        return f"{text}_{thres}-{consec}-{total_size}"

    read_folder = f'./clips_front_extend_{extend}/'
    # write_folder = f'plot_{extend}_{reprval("EEG")}_{reprval("EEG2")}_{reprval("EOG")}/'

    # logging.basicConfig(filename=write_folder+'print.log', encoding='utf-8', level=logging.DEBUG)
    # print(write_folder)

    # os.makedirs(write_folder, exist_ok=True)
    pattern1 = '*-PSG.edf'
    pattern2 = '*-Hypnogram.edf'

    psg_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern1)])
    hypnogram_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern2)])
    
    # COUNT = len(psg_list)
    COUNT = 20
    
    psg_iter = iter(psg_list)
    hypnogram_iter = iter(hypnogram_list)

    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
    channel = channels[1]
    count_1, count_2, count_3, count_4, count_5, count_6 = 0, 0, 0, 0, 0, 0

    df = pd.DataFrame()

    for i, (psg_id, hypnogram_id) in enumerate(zip(psg_iter, hypnogram_iter)):
        if i == COUNT:
            break
        # print(f"{i}/{COUNT}", end="\r")
        signal_path = os.path.join(read_folder, psg_id)
        label_path = os.path.join(read_folder, hypnogram_id)
        edf_signal = pyedflib.EdfReader(signal_path)
        edf_label = pyedflib.EdfReader(label_path)
        annotations = edf_label.readAnnotations()
        start = edf_signal.getStartdatetime()
        signals, frequencies = edf_signal.getSignalLabels(), edf_signal.getSampleFrequencies()
        
        data = []
        for ch_idx, sig_name, freq in zip( range(len(signals)), signals, frequencies,):
            sig = edf_signal.readSignal(chn=ch_idx)
            idx = pd.date_range(  start=start, periods=len(sig), freq=pd.Timedelta(1 / freq, unit="s") )
            data += [pd.Series(sig, index=idx, name=sig_name)]
        # create DataFrames
        annotations_df = pd.DataFrame(annotations)
        annotations_df = annotations_df.T
        annotations_df.rename(columns={0: "Onset", 1: "Duration", 2:"Annotations"}, inplace=True)
        signal_df =pd.concat(data[0:3], axis=1)
            
        def annotate_sleep_stage(signal_df, annotations_df, start):
            start_time = start + pd.to_timedelta(int(annotations_df.at[1, 'Onset']), unit='seconds')
            end_time = start_time + pd.to_timedelta(int(annotations_df.at[1, 'Duration']), unit='seconds')

            mask = (signal_df.index >= start_time) & (signal_df.index < end_time)
            return mask.astype(int)
        
        signal_df['N1'] = annotate_sleep_stage(signal_df, annotations_df, start)

        eeg_index = n1_multi_range(signal_df[channels[0]], thres=EEG_THRES, CONSECUTIVE_STEPS=EEG_CONSEC, total_size=EEG_TOTAL_SIZE)
        eeg2_index = n1_multi_range(signal_df[channels[1]], thres=EEG2_THRES, CONSECUTIVE_STEPS=EEG2_CONSEC, total_size=EEG2_TOTAL_SIZE)
        eog_index = n1_multi_range(signal_df[channels[2]], thres=EOG_THRES, CONSECUTIVE_STEPS=EOG_CONSEC, total_size=EOG_TOTAL_SIZE)

        signal_df = get_n1_eeg(signal_df, eeg_index, eeg2_index, eog_index, psg_id)
        # acc1 = accuracy_method_1(signal_df)
        # acc2 = accuracy_method_2(signal_df)
        # acc3 = accuracy_method_3(signal_df)
        f1_score = accuracy_method_4(signal_df)
        cohen = accuracy_method_5(signal_df)
        accuracy = accuracy_method_6(signal_df)
        # if f1_score <= 0.25:
        #     lol(signal_df, write_folder, psg_id, save_fig=True)
        #     print(psg_id, f1_score, f"idx: {i}")
        # print(psg_id, acc1, acc2, acc3, f1_score)

        # count_1 += acc1
        # count_2 += acc2
        # count_3 += acc3
        count_4 += f1_score
        count_5 += cohen
        count_6 += accuracy

        # print(count_1, count_2, count_3, f"/{i+1}")
        # df = pd.concat([df, pd.DataFrame([psg_id, acc1, acc2, acc3, f1_score])], axis=0)
        # temp = pd.DataFrame([psg_id, f1_score, cohen]).T
        # df = pd.concat([df, temp], axis=0)

        # break
    # write to file
    # for full file
    # df = pd.DataFrame([eeg_thres, eeg2_thres, eog_thres, count_4/len(psg_list), count_5/len(psg_list)]).T
    df = pd.DataFrame([eeg_thres, eeg2_thres, eog_thres, count_4/COUNT, count_5/COUNT]).T
    return count_4/COUNT, count_5/COUNT, count_6/COUNT

    # df.to_csv("experiment_20.csv", index=False, mode='a', header=False)

if __name__ == '__main__':
    res = main_function(35, 22, 156)
    print(res)