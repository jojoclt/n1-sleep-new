from main import main_function

for eog_thres in range(150, 201, 2):
    for eeg_thres in range(25,41):
        for eeg2_thres in range(20,36):
            print(f"eeg_thres: {eeg_thres}, eeg2_thres: {eeg2_thres}, eog_thres: {eog_thres}")
            main_function(eeg_thres, eeg2_thres, eog_thres)