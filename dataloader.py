import os
import numpy as np
from sklearn.model_selection import train_test_split
class LoadDataset():
    def __init__(self):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
    def load_stft_data(self, data_path, index):
        data = np.load(data_path[index], mmap_mode='r')
        print(f"Loaded data shape for {data_path[index]}: {data.shape}")
        return data
    def load_labels(self, label_path, dev_range):
        label = np.load(label_path)
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1
        sample_index_list = []
        for dev_idx in dev_range:
            num_pkt = np.count_nonzero(label == dev_idx)
            pkt_range = np.arange(0, num_pkt, dtype=int)
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
            print(f'Dev {dev_idx + 1} has {num_pkt} packets.')
        label = label[sample_index_list]
        return label

def read_train_data(data_folder='/data/czx/paper8data/stft/LOS',
                    label_path='/data/czx/paper8data/stft/label_LOS.npy',
                    dev_range=np.arange(0, 18, dtype=int)):
    data_stft_all = []
    y_all = []
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]
    data_files.sort()
    LoadDatasetObj = LoadDataset()
    for dev_idx in dev_range:
        print(f"Loading data for device {dev_idx + 1}...")
        data_ch0 = LoadDatasetObj.load_stft_data(data_files, dev_idx)
        data_ch0 = np.expand_dims(data_ch0, axis=1)
        data_stft_all.append(data_ch0)
        y_ch0 = LoadDatasetObj.load_labels(label_path, np.array([dev_idx]))
        y_all.append(y_ch0)
    data_stft_all = np.concatenate(data_stft_all, axis=0)
    y_all = np.concatenate(y_all)
    print(f"Total data shape: {data_stft_all.shape}")
    X_train, X_temp, Y_train, Y_temp = train_test_split(data_stft_all, y_all, test_size=0.2, random_state=32, stratify=y_all)
    return X_train, X_temp, Y_train, Y_temp

def read_test_data(data_folder='/data/czx/paper8data/stft/NLOS',
                   label_path='/data/czx/paper8data/stft/label_NLOS.npy',
                   dev_range=np.arange(0, 18, dtype=int)):
    data_stft_all = []
    y_all = []
    data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]
    data_files.sort()
    LoadDatasetObj = LoadDataset()
    for dev_idx in dev_range:
        print(f"Loading data for device {dev_idx + 1}...")
        data_ch0 = LoadDatasetObj.load_stft_data(data_files, dev_idx)
        data_ch0 = np.expand_dims(data_ch0, axis=1)
        y_ch0 = LoadDatasetObj.load_labels(label_path, np.array([dev_idx]))
        data_stft_all.append(data_ch0)
        y_all.append(y_ch0)
    data_stft_all = np.concatenate(data_stft_all, axis=0)
    y_all = np.concatenate(y_all)
    X_test, X_finetune, Y_test, Y_finetune = train_test_split(data_stft_all, y_all,test_size=0.5, random_state=32, stratify=y_all)
    return X_test, Y_test



