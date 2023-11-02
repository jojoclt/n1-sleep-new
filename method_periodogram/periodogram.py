import sys
sys.path.append('../')

import pyedflib # ref: https://pyedflib.readthedocs.io/en/latest/
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
from modules import *
from modules_periodogram import *
from tqdm import tqdm
import logging
# import warnings
# warnings.filterwarnings("ignore")

freq_low = 8
freq_high = 12
extend = 10

read_folder = f'../clips_front_extend_{extend}/'
write_folder = f'./plot_periodogram_{freq_low}_{freq_high}_ACC/'

# logging.basicConfig(filename=write_folder+'print.log', encoding='utf-8', level=logging.DEBUG)
print(write_folder)

os.makedirs(write_folder, exist_ok=True)
pattern1 = '*-PSG.edf'
pattern2 = '*-Hypnogram.edf'

psg_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern1)])
hypnogram_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern2)])
psg_iter = iter(psg_list)
hypnogram_iter = iter(hypnogram_list)

channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
channel = channels[1]
count_1, count_2, count_3 = 0, 0, 0

for i, (psg_id, hypnogram_id) in enumerate(zip(psg_iter, hypnogram_iter)):
    print(f"{i}/{len(psg_list)}", end="\r")
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

    periodgram = get_periodogram_n1(signal_df.iloc[:, 0], psg_id, sf=100, freq_low=freq_low, freq_high=freq_high)
    signal_df['Per'] = np.repeat(periodgram, 3000)[:len(signal_df)]

    signal_df['N1_predict'] = (signal_df['Per'] < 0.5 * signal_df['Per'].shift(1)) * 1
    signal_df['N1_predict'] = signal_df['N1_predict'].fillna(0)

    signal_df['N1_predict'] = signal_df['N1_predict'].astype(int)
    signal_df['N1_predict'] = signal_df['N1_predict'].rolling(3000, min_periods=1).max()
    signal_df['N1_predict'] = signal_df['N1_predict'].fillna(0)
    signal_df['N1_predict'] = signal_df['N1_predict'].astype(int)
    
    print(psg_id, accuracy_method_1(signal_df), accuracy_method_2(signal_df), accuracy_method_3(signal_df))

    count_1 += accuracy_method_1(signal_df)
    count_2 += accuracy_method_2(signal_df)
    count_3 += accuracy_method_3(signal_df)
    # count_4 += accuracy_method_4(signal_df)
    print(count_1, count_2, count_3, f"/{i+1}")

    lol(signal_df, write_folder, psg_id, save_fig=True)
    # break
# write to file
value = pd.DataFrame([count_1, count_2, count_3])
value.to_csv("periodogram.csv")
