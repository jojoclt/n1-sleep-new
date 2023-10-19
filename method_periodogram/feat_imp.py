import pyedflib # ref: https://pyedflib.readthedocs.io/en/latest/
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import fnmatch
from modules_same import *
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from modules_periodogram import *
import seaborn as sns
import joblib

EEG_THRES = 25
EOG_THRES = 120

EEG_CONSEC = 300
EOG_CONSEC = 2500

read_folder = '../clips_front_extend_10/'
write_folder = f'./sleep_cas_extend_{EEG_THRES}_EEGCONSEC_{EEG_CONSEC}_eog_{EOG_THRES}_EEGCONSEC_{EOG_CONSEC}/'

os.makedirs(write_folder, exist_ok=True)
pattern1 = '*-PSG.edf'
pattern2 = '*-Hypnogram.edf'

model_save = "SKLEARN.joblib"

psg_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern1)])
hypnogram_list = sorted([f for f in os.listdir(read_folder) if fnmatch.fnmatch(f, pattern2)])
psg_iter = iter(psg_list)
hypnogram_iter = iter(hypnogram_list)

channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
channel = channels[1]

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
    signal_df =pd.concat(data, axis=1)
    periodgram = get_periodogram_n1(signal_df.iloc[:, 0], psg_id, sf=100, freq_low = 0, freq_high = 1)
    signal_df['Per'] = np.repeat(periodgram, 3000)[:len(signal_df)]

    def annotate_sleep_stage(signal_df, annotations_df, start):
        start_time = start + pd.to_timedelta(int(annotations_df.at[1, 'Onset']), unit='seconds')
        end_time = start_time + pd.to_timedelta(int(annotations_df.at[1, 'Duration']), unit='seconds')

        mask = (signal_df.index >= start_time) & (signal_df.index < end_time)
        return mask.astype(int)
    signal_df['N1'] = annotate_sleep_stage(signal_df,annotations_df, start)
    # features = np.array(channels)
    # sns.heatmap(signal_df)
    # plt.savefig(os.path.join(write_folder, psg_id[:-8] + '.png'))
    # break
    cleaned_data = signal_df.fillna(0)
    filename = "SKLEARN.joblib"

    loaded_model = joblib.load(filename)
    features = np.array(cleaned_data.columns[:-1])
    signal_df['T'] = pd.Series(loaded_model.predict(cleaned_data[features]))
    lol(signal_df, "TEST", psg_id, save_fig=True)

    break
    clf = RandomForestClassifier(n_estimators=500, random_state=0, max_features = 5, n_jobs=-1, bootstrap=True, oob_score=True)

    # print(cleaned_data[features].iloc[-60000:-30000,:])
    # print(cleaned_data.columns)
    # print(cleaned_data[features])
    clf.fit(cleaned_data[features], cleaned_data['N1'])
    joblib.dump(clf, filename)
    # from the calculated importances, order them from most to least important
    # and make a barplot so we can visualize what is/isn't important
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    plt.barh(padding, importances[sorted_idx], align='center')
    plt.yticks(padding, features[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title(f"Variable Importance {psg_id}")
    # plt.show()
    os.makedirs(os.path.join("FEAT_IMP_FILLNA_THETA"), exist_ok=True)
    plt.savefig(os.path.join("FEAT_IMP_FILLNA_THETA",psg_id[:-8] + '.png'))
    plt.clf()
    # lol(signal_df, write_folder, psg_id, save_fig=True)
    break

    