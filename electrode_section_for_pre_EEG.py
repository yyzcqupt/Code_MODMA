import os
import mne
import os.path as op
import numpy as np

# 路径
section = 2     # 切片时长设置为2
input_folder = '/home/cquptyyz/EEGdata/MODMA_fif_pre'
output_folder = '/home/cquptyyz/EEGdata/MODMA_elecdtrode'
output_folder = os.path.join(output_folder + f"_{section}s")  # '/home/cquptyyz/EEGdata/MODMA_MODMA_elecdtrode_2s'


# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入目录中的所有 .fif 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.fif'):
        file_path = os.path.join(input_folder, file_name)

        # 创建特定文件夹以存储结果
        subject_folder = os.path.join(output_folder, file_name.replace('.fif', ''))
        os.makedirs(subject_folder, exist_ok=True)

        # 读取原始 EEG 数据
        raw = mne.io.read_raw_fif(file_path, preload=True)

        # 设置 EEG 平均参考
        raw.set_eeg_reference(projection=True)


        # 获取数据长度和采样率
        n_samples = raw.n_times
        sfreq = raw.info['sfreq']
        segment_length = int(section * sfreq)  # 10秒的样本数

        # 分段处理
        for i in range(0, n_samples, segment_length):
            if i + segment_length > n_samples:
                break  # 跳过不满10秒的片段

            # 提取片段
            raw_segment = raw.copy().crop(tmin=i / sfreq, tmax=(i + segment_length) / sfreq)

            # 获取 EEG 信号数据
            eeg_data = raw_segment.get_data()


            # 保存结果
            segment_index = i // segment_length + 1
            file_base_name = file_name.replace('.fif', '') + f'electrode_{segment_index}_{section}s'

            save_path = os.path.join(subject_folder, f'{file_base_name}.npy')
            np.save(save_path, eeg_data)
