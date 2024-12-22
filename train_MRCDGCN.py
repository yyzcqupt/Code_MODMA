import statistics
import os
import re
import sys
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from blocks_yyz import Adopt_Gnn_PSD_DE_SE, extract_frequency_bands
import matplotlib.pyplot as plt
# from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from load_data import load_data_for_model, load_subject_independent_data_for_model
from MRCDGCN import MRCDGCN

# 创建 DataLoader 的函数
def create_dataloader(data, labels, shuffle, batch_size=64):
    tensor_data = torch.from_numpy(data).to(torch.float32)
    tensor_labels = torch.from_numpy(labels).to(torch.int64)
    dataset = TensorDataset(tensor_data, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, dataloader


os.chdir('/home/cquptyyz/Code_MODMA/Self_adaptve_gnn_multiple_domain')
print(os.getcwd())

### ========================= Sets the seed for random numbers ===============================================###
torch.manual_seed(0)
np.random.seed(0)

### ========================= Use the GPU to train ===============================================###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### ========================= Setup for saving model ===============================================###
current_dir = os.getcwd()
save_model_file = current_dir + '/model_kfold_MRCDGCN'
# 确保输出目录存在
os.makedirs(save_model_file, exist_ok=True)

### ========================= Load data ===============================================###
data_type = 'MODMA_EEG'
input_folder = '/home/cquptyyz/EEGdata/MODMA_elecdtrode_10s'
selected_folders = {'02010002', '02010004', '02020008', '02020010'}
test_datas, test_labels, train_datas, train_labels = load_subject_independent_data_for_model(input_folder, selected_folders)
fs = 250
nperseg = 50
noverlap = 25
test_datas = np.array(test_datas)
train_datas = np.array(train_datas)

### ========================= Initialization parameters ===============================================###
# dim = train_datas.shape[3]  # 特征维度
# eeg_channels = train_datas.shape[2]
# subgraph_size = 20  # k = 20
# alpha_PSD = 3
# alpha_DE = 3
# alpha_SE = 3
# beta_PSD = 0.05
# beta_DE = 0.05
# beta_SE = 0.05
# gcn_depth = 2
# layers = 3
S = train_datas.shape[2]
l = train_datas.shape[1]
class_num = 2
hidden = 64
num_classes = 2  # 几分类
train_epochs = 10  # 15 10 5 20 2 10
batch_size = 64  # 32
kfolds = 10  # 10 5
test_size = 0.1

### ========================= Split data & Cross validate ==============================================
# 打乱并分割数据
kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
folds = list(kf.split(train_datas))

### ========================= train model ===============================================###
eval_kfold_acc = []
eval_kfold_recall = []
eval_kfold_specificity = []
eval_kfold_precision = []
eval_kfold_f1 = []

for kfold, (train_idx, eval_idx) in enumerate(folds):

    ### ============ Initialization Performance ============
    best_acc = 0
    best_recall = 0
    best_specificity = 0
    best_f1 = 0
    best_epoch = -1
    best_kfold = -1

    ### ============ Prepare data for every fold ============
    x_train = train_datas[train_idx]
    y_train = train_labels[train_idx]
    x_eval = train_datas[eval_idx]
    y_eval = train_labels[eval_idx]

    ### ============ Initialization model ============
    # model
    model = MRCDGCN(S, l, class_num, hidden, device).to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4) # lr=1e-3
    # 损失函数
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device) # reduction='sum'表示计算的loss是求和的

    # 设置每一折的最佳模型保存位置
    folder_bestmodel_file = os.path.join(save_model_file + f"/{kfold}fold_model.pth")

    # 保存训练以及验证集的损失
    train_losses = []
    eval_losses = []

# 开始每个epoch训练
    for iter in range(train_epochs):

        train_data, train_loader = create_dataloader(x_train, y_train, shuffle=True, batch_size=batch_size)
        eval_data, eval_loader = create_dataloader(x_eval, y_eval, shuffle=False, batch_size=batch_size)

        all_train_loss = 0.0  # 用于累加每个batch的损失，计算平均损失
        all_train_accuracy = 0.0  # 用于累加每个batch的准确率，计算平均准确率

        # --------------------------train------------------------------
        model.train()
        for inputs, target in train_loader:
            inputs, target = inputs.to(device).requires_grad_(), target.to(device).long()
            output = model(inputs)
            loss = criterion(output, target)
            optimizer.zero_grad()   # 梯度归零
            loss.backward()     # 计算损失函数梯度
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加这一行
            optimizer.step()    # 根据计算的梯度更新参数

            acc = (output.argmax(dim=1) == target).float().sum()
            all_train_accuracy += acc
            all_train_loss += loss.detach().item()

        print(f"Train: Epoch: {iter} ---- kfold: {kfold+1} ---- acc: {all_train_accuracy / len(train_data):.4f} ---- loss: {all_train_loss / len(train_data):.4f}\n")
        train_losses.append(all_train_loss / len(train_data))

        # --------------------------validation------------------------------
        model.eval()
        all_eval_loss = 0.0  # 用于累加每个batch的损失，计算平均损失
        all_eval_accuracy = 0.0  # 用于累加每个batch的准确率，计算平均准确率
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, target in eval_loader:
                inputs, target = inputs.to(device), target.to(device).long()
                output = model(inputs)
                loss = criterion(output, target)

                acc = (output.argmax(dim=1) == target).float().sum()
                all_eval_accuracy += acc
                all_eval_loss += loss.detach().item()

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(output.argmax(dim=1).cpu().numpy())
        print(f"Eval: Epoch: {iter} ---- kfold: {kfold + 1} ---- acc: {all_eval_accuracy / len(eval_data):.4f} ---- loss: {all_eval_loss / len(eval_data):.4f}\n")
        eval_losses.append(all_eval_loss / len(eval_data))

        # 计算各类指标
        eval_accuracy = all_eval_accuracy / len(eval_data)
        cm = confusion_matrix(all_targets, all_predictions)
        tn, fp, fn, tp = cm.ravel()
        # 计算敏感性、特异性和 F1 分数
        eval_recall = recall_score(all_targets, all_predictions)
        eval_specificity = tn / (tn + fp)
        eval_f1 = f1_score(all_targets, all_predictions)
        eval_precision = precision_score(all_targets, all_predictions)

        if eval_accuracy > best_acc:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, folder_bestmodel_file)
            best_acc = eval_accuracy
            best_recall = eval_recall
            best_specificity = eval_specificity
            best_f1 = eval_f1
            best_precision = eval_precision
            best_kfold = kfold + 1
            best_epoch = iter

        print(f"Eval: Epoch: {iter} ---- kfold: {kfold + 1} ---- accuracy: {eval_accuracy:.4f}\n")
        print(f"Eval: Epoch: {iter} ---- kfold: {kfold + 1} ---- recall(sensitivity): {eval_recall:.4f}\n")
        print(f"Eval: Epoch: {iter} ---- kfold: {kfold + 1} ---- specificity: {eval_specificity:.4f}\n")
        print(f"Eval: Epoch: {iter} ---- kfold: {kfold + 1} ---- precision: {eval_precision:.4f}\n")
        print(f"Eval: Epoch: {iter} ---- kfold: {kfold + 1} ---- f1: {eval_f1:.4f}\n")

        print(f"kfold: {best_kfold} : best_acc: {best_acc:.4f} ---- Epoch: {best_epoch}\n")
        print(f"参数总量：{sum(p.numel() for p in model.parameters())}\n")


    # 记录每一折模型验证集的表现
    eval_kfold_acc.append(best_acc.item())
    eval_kfold_recall.append(best_recall)
    eval_kfold_specificity.append(best_specificity)
    eval_kfold_precision.append(best_precision)
    eval_kfold_f1.append(best_f1)

# 打印验证集平均表现
eval_kfold_acc_avg = sum(eval_kfold_acc) / len(eval_kfold_acc)
eval_kfold_acc_std = statistics.stdev(eval_kfold_acc)
eval_kfold_recall_avg = sum(eval_kfold_recall) / len(eval_kfold_recall)
eval_kfold_recall_std = statistics.stdev(eval_kfold_recall)
eval_kfold_specificity_avg = sum(eval_kfold_specificity) / len(eval_kfold_specificity)
eval_kfold_specificity_std = statistics.stdev(eval_kfold_specificity)
eval_kfold_precision_avg = sum(eval_kfold_precision) / len(eval_kfold_precision)
eval_kfold_precision_std = statistics.stdev(eval_kfold_precision)
eval_kfold_f1_avg = sum(eval_kfold_f1) / len(eval_kfold_f1)
eval_kfold_f1_std = statistics.stdev(eval_kfold_f1)

print(f"{kfolds}折交叉训练的模型验证集上的平均表现如下：\n")
print(f"Accuracy: Average = {eval_kfold_acc_avg * 100:.2f}%, Standard Deviation = {eval_kfold_acc_std * 100:.2f}%")
print(f"Recall: Average = {eval_kfold_recall_avg * 100:.2f}%, Standard Deviation = {eval_kfold_recall_std * 100:.2f}%")
print(f"Specificity: Average = {eval_kfold_specificity_avg * 100:.2f}%, Standard Deviation = {eval_kfold_specificity_std * 100:.2f}%")
print(f"Precision: Average = {eval_kfold_precision_avg * 100:.2f}%, Standard Deviation = {eval_kfold_precision_std * 100:.2f}%")
print(f"F1 Score: Average = {eval_kfold_f1_avg * 100:.2f}%, Standard Deviation = {eval_kfold_f1_std * 100:.2f}%\n")

### ========================= 可视化损失函数变化 ===============================================##

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(train_epochs), train_losses, label='Train Loss')
plt.plot(range(train_epochs), eval_losses, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.show()



### ========================= test model ===============================================###
test_kfold_acc = []
test_kfold_recall = []
test_kfold_specificity = []
test_kfold_precision = []
test_kfold_f1 = []

X_test_tensor = torch.from_numpy(test_datas).float().to(device)
Y_test_tensor = torch.from_numpy(test_labels).long().to(device)

for kfold in range(kfolds):
    model = MRCDGCN(S, l, class_num, hidden, device).to(device)
    model_file = os.path.join(save_model_file + f"/{kfold}fold_model.pth") # 加载训练好的模型
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor).argmax(dim=1).cpu().numpy()
        test_targets = Y_test_tensor.cpu().numpy()

    test_accuracy =accuracy_score(test_targets, test_preds)

    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()
    # 计算敏感性、特异性和 F1 分数
    test_recall = recall_score(test_targets, test_preds)
    test_specificity = tn / (tn + fp)
    test_f1 = f1_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds)

    # 记录每一折模型验证集的表现
    test_kfold_acc.append(test_accuracy)
    test_kfold_recall.append(test_recall)
    test_kfold_specificity.append(test_specificity)
    test_kfold_precision.append(test_precision)
    test_kfold_f1.append(test_f1)

    print(f"Test: ---- kfold: {kfold + 1} ---- accuracy: {test_accuracy:.4f}\n")
    print(f"Test: ---- kfold: {kfold + 1} ---- recall(sensitivity): {test_recall:.4f}\n")
    print(f"Test: ---- kfold: {kfold + 1} ---- specificity: {test_specificity:.4f}\n")
    print(f"Test: ---- kfold: {kfold + 1} ---- precision: {test_precision:.4f}\n")
    print(f"Test: ---- kfold: {kfold + 1} ---- f1: {test_f1:.4f}\n")

    ConfusionMatrixDisplay.from_predictions(test_targets,
                                            test_preds,
                                            display_labels=["MDD", "HD"],    # 第一行和第一列对应于标签 0（即 "MDD"） 第二行和第二列对应于标签 1（即 "HD"）
                                            cmap=plt.cm.Reds,
                                            colorbar=True)
    # 设置标题
    model_safe_name = re.sub(r'[<>:"/\\|?*]', '', model.__class__.__name__)
    plt.title(f'Confusion Matrix for {model_safe_name} of kfold: {kfold + 1}')
    plt.show()

# 打印验证集平均表现
test_kfold_acc_avg = sum(test_kfold_acc) / len(test_kfold_acc)
test_kfold_acc_std = statistics.stdev(test_kfold_acc)
test_kfold_recall_avg = sum(test_kfold_recall) / len(test_kfold_recall)
test_kfold_recall_std = statistics.stdev(test_kfold_recall)
test_kfold_specificity_avg = sum(test_kfold_specificity) / len(test_kfold_specificity)
test_kfold_specificity_std = statistics.stdev(test_kfold_specificity)
test_kfold_precision_avg = sum(test_kfold_precision) / len(test_kfold_precision)
test_kfold_precision_std = statistics.stdev(test_kfold_precision)
test_kfold_f1_avg = sum(test_kfold_f1) / len(test_kfold_f1)
test_kfold_f1_std = statistics.stdev(test_kfold_f1)

print(f"{kfolds}折交叉训练的模型测试集上的平均表现如下：")
print(f"Accuracy: Average = {test_kfold_acc_avg * 100:.2f}%, Standard Deviation = {test_kfold_acc_std * 100:.2f}%")
print(f"Recall: Average = {test_kfold_recall_avg * 100:.2f}%, Standard Deviation = {test_kfold_recall_std * 100:.2f}%")
print(f"Specificity: Average = {test_kfold_specificity_avg * 100:.2f}%, Standard Deviation = {test_kfold_specificity_std * 100:.2f}%")
print(f"Precision: Average = {test_kfold_precision_avg * 100:.2f}%, Standard Deviation = {test_kfold_precision_std * 100:.2f}%")
print(f"F1 Score: Average = {test_kfold_f1_avg * 100:.2f}%, Standard Deviation = {test_kfold_f1_std * 100:.2f}%\n")

### ========================= 保存测试集结果 ===============================================###
# 准备每一折的结果数据
fold_results = []
for i in range(kfolds):
    fold_results.append({
        "Fold": i + 1,
        "Accuracy": test_kfold_acc[i],
        "Recall": test_kfold_recall[i],
        "Specificity": test_kfold_specificity[i],
        "Precision": test_kfold_precision[i],
        "F1 Score": test_kfold_f1[i]
    })

# 准备平均结果数据
average_results = {
    "Accuracy": (test_kfold_acc_avg, test_kfold_acc_std),
    "Recall": (test_kfold_recall_avg, test_kfold_recall_std),
    "Specificity": (test_kfold_specificity_avg, test_kfold_specificity_std),
    "Precision": (test_kfold_precision_avg, test_kfold_precision_std),
    "F1 Score": (test_kfold_f1_avg, test_kfold_f1_std)
}

# 生成Markdown报告内容
report_lines = []
report_lines.append("# 测试集评价指标报告\n")
report_lines.append(f"共进行了{kfolds}折交叉验证。\n")

# 每一折的结果
report_lines.append("## 每一折的结果\n")
for result in fold_results:
    report_lines.append(f"### 第{result['Fold']}折\n")
    report_lines.append(f"- 准确率: {result['Accuracy'] * 100:.2f}%\n")
    report_lines.append(f"- 召回率: {result['Recall'] * 100:.2f}%\n")
    report_lines.append(f"- 特异性: {result['Specificity'] * 100:.2f}%\n")
    report_lines.append(f"- 精确率: {result['Precision'] * 100:.2f}%\n")
    report_lines.append(f"- F1分数: {result['F1 Score'] * 100:.2f}%\n")

# 平均结果
report_lines.append("## 平均结果\n")
for metric, (avg, std) in average_results.items():
    report_lines.append(f"### {metric}\n")
    report_lines.append(f"- {avg * 100:.2f}% ± {std * 100:.2f}%\n")

# 将报告内容保存到文件
report_content = "\n".join(report_lines)
current_date = datetime.now().strftime("%Y-%m-%d")
# report_file = "test_report.md"
report_file = f"test_report_{data_type}_{current_date}_MRCDGCN.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"报告已保存到 {report_file}")
