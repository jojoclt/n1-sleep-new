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

# the format of plot will be THRES/CONSEC/TOTAL_SIZZE
EEG_THRES = 25
EEG_CONSEC = 300
EEG_TOTAL_SIZE = 1500

# EEG2_THRES = 25
# EEG2_CONSEC = 300
# EEG2_TOTAL_SIZE = 1500

EOG_THRES = 120
EOG_CONSEC = 2500
EOG_TOTAL_SIZE = 2500
extend = 30 
file = '4491'

def reprval(text):
    thres = eval(f"{text}_THRES")
    consec = eval(f"{text}_CONSEC")
    total_size = eval(f"{text}_TOTAL_SIZE")
    return f"{text}_{thres}-{consec}-{total_size}"

read_folder = f'./clips_front_extend_{extend}/'
write_folder = f'./plot_single/'

# logging.basicConfig(filename=write_folder+'print.log', encoding='utf-8', level=logging.DEBUG)
print(write_folder)

os.makedirs(write_folder, exist_ok=True)
pattern1 = f'*{file}*-PSG.edf'
pattern2 = f'*{file}*-Hypnogram.edf'

psg_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern1)])
hypnogram_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern2)])
psg_iter = iter(psg_list)
hypnogram_iter = iter(hypnogram_list)

channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
channel = channels[1]

for i, (psg_id, hypnogram_id) in enumerate(zip(psg_iter, hypnogram_iter)):
    # print(f"{i}/{len(psg_list)}", end="\r")
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

    eeg_index = n1_multi_range(signal_df[channels[0]], thres=EEG_THRES, CONSECUTIVE_STEPS=EEG_CONSEC, total_size=EEG_TOTAL_SIZE, DEBUG = True)
    eog_index = n1_multi_range(signal_df[channels[2]], thres=EOG_THRES, CONSECUTIVE_STEPS=EOG_CONSEC, total_size=EOG_TOTAL_SIZE, DEBUG = True)

    signal_df = get_n1_eeg(signal_df, eeg_index, eog_index, psg_id)
    lol(signal_df, write_folder, psg_id, save_fig=True)
    # break

    