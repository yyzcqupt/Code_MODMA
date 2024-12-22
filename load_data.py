import os
import numpy as np

def load_data_for_model(data_type, input_folder):
    # 初始化数据和标签列表
    data, labels = [], []

    if data_type in ['MODMA_ROI', 'MODMA_ROI_PSD']:
        # 选择ROI划分规则
        ROI = 'HCPMMP1'
        ESI = 'dSPM'
        aggragation = 'pca_flip'
        suffix = f"{ESI}_{aggragation}.npy"
        # 设置输入文件夹路径
        input_folder = os.path.join(input_folder + f"_{ROI}" + ("_PSD" if data_type == 'MODMA_ROI_PSD' else ""))

    elif data_type in ['MODMA_EEG', 'MODMA_EEG_PSD']:
        # 设置文件后缀
        suffix = ".npy"

    elif data_type == 'TD_BRAIN_EEG_TYT_2s':
        # 加载数据并调整形状
        Data = np.load(input_folder)
        datas = np.reshape(Data['features'], (-1, Data['features'].shape[2], Data['features'].shape[3]))
        labels = np.reshape(Data['labels'], (-1))
        return datas, labels

    else:
        # 输入格式错误的报错提示
        raise ValueError(
            "Unsupported data_type. Please choose from 'MODMA_ROI', 'MODMA_ROI_PSD', 'MODMA_EEG', 'MODMA_EEG_PSD', 'TD_BRAIN_EEG_TYT_2s'.")

    all_num = 0
    # 遍历子文件夹
    for subject_folder in os.listdir(input_folder):
        subject_path = os.path.join(input_folder, subject_folder)
        if os.path.isdir(subject_path):
            # 确定标签
            label = 1 if subject_folder.startswith('0201') else 0
            npy_num = 0
            # 遍历子文件夹中的 .npy 文件
            for file_name in os.listdir(subject_path):
                if file_name.endswith(suffix):
                    file_path = os.path.join(subject_path, file_name)
                    # 读取 .npy 文件
                    data.append(np.load(file_path))
                    labels.append(label)
                    npy_num += 1
            all_num += npy_num
            print(f'file: {subject_path}, npy_num: {npy_num}')

    print(f'all_npy_num: {all_num}')
    # 转换为 numpy 数组
    datas = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return datas, labels

def load_designated_subjects_data_from_MODMA(input_folder, subjects_list):
    suffix = ".npy"
    data, labels = [], []
    all_num = 0
    for subject_folder in subjects_list:
        subject_path = os.path.join(input_folder, subject_folder)
        if os.path.isdir(subject_path):
            # 确定标签
            label = 1 if subject_folder.startswith('0201') else 0
            npy_num = 0
            # 遍历子文件夹中的 .npy 文件
            for file_name in os.listdir(subject_path):
                if file_name.endswith(suffix):
                    file_path = os.path.join(subject_path, file_name)
                    # 读取 .npy 文件
                    data.append(np.load(file_path))
                    labels.append(label)
                    npy_num += 1
            all_num += npy_num
            print(f'file: {subject_path}, npy_num: {npy_num}')

    print(f'all_npy_num: {all_num}')
    # 转换为 numpy 数组
    datas = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return datas, labels







def load_subject_independent_data_for_model(input_folder, selected_folders):
    # 指定要挑选的子文件夹名称
    # selected_folders = {'02010004', '02010006'}
    suffix = '.npy'
    # 初始化数据和标签
    selected_data = []
    selected_labels = []
    remaining_data = []
    remaining_labels = []

    selected_num = 0
    remaining_num = 0

    # 遍历 input_folder 中的子文件夹
    for subject_folder in os.listdir(input_folder):
        subject_path = os.path.join(input_folder, subject_folder)
        if os.path.isdir(subject_path):
            # 确定标签
            label = 1 if subject_folder.startswith('0201') else 0
            npy_num = 0
            # 根据是否在 selected_folders 中进行分类
            target_data = selected_data if subject_folder in selected_folders else remaining_data
            target_labels = selected_labels if subject_folder in selected_folders else remaining_labels
            num_counter = selected_num if subject_folder in selected_folders else remaining_num

            # 遍历子文件夹中的 .npy 文件
            for file_name in os.listdir(subject_path):
                if file_name.endswith(suffix):
                    file_path = os.path.join(subject_path, file_name)
                    # 读取 .npy 文件
                    target_data.append(np.load(file_path))
                    target_labels.append(label)
                    npy_num += 1

            # 更新计数
            if subject_folder in selected_folders:
                selected_num += npy_num
            else:
                remaining_num += npy_num

            print(f'file: {subject_path}, npy_num: {npy_num}')

    print(f'selected_npy_num: {selected_num}')
    print(f'remaining_npy_num: {remaining_num}')

    # 转换为 numpy 数组
    selected_datas = np.array(selected_data, dtype=np.float32)
    selected_labels = np.array(selected_labels, dtype=np.int64)
    remaining_datas = np.array(remaining_data, dtype=np.float32)
    remaining_labels = np.array(remaining_labels, dtype=np.int64)

    return selected_datas, selected_labels, remaining_datas, remaining_labels


