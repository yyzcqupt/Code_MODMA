import statistics
import os
import re
import sys
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from model_yyz_2 import DGCNN
import matplotlib.pyplot as plt
# from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from load_data import load_data_for_model

def cross_validate_sequential_split(x_data, y_labels, kfold):
    train_indices = {}
    eval_indices = {}
    skf = StratifiedKFold(n_splits=kfold, shuffle=False)  # random_state=42
    i = 0
    for train_idx, eval_idx in skf.split(x_data, y_labels):
        train_indices.update({i: train_idx})
        eval_indices.update({i: eval_idx})
        i += 1
    return train_indices, eval_indices

def split_xdata(eeg_data, train_idx, eval_idx):
    x_train = np.copy(eeg_data[train_idx, :, :])
    x_eval = np.copy(eeg_data[eval_idx, :, :])
    x_train = torch.from_numpy(x_train).to(torch.float32)
    x_eval = torch.from_numpy(x_eval).to(torch.float32)
    return x_train, x_eval

def split_ydata(y_true, train_idx, eval_idx):
    y_train = np.copy(y_true[train_idx])
    y_eval = np.copy(y_true[eval_idx])
    y_train = torch.from_numpy(y_train).to(torch.int64)
    y_eval = torch.from_numpy(y_eval).to(torch.int64)
    return y_train, y_eval


print(os.getcwd())
os.chdir('/home/cquptyyz/Code_MODMA/DGCNN/DGCNN_2')

### ========================= Sets the seed for random numbers ===============================================###
torch.manual_seed(0)
np.random.seed(0)

### ========================= Use the GPU to train ===============================================###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### ========================= Setup for saving model ===============================================###
current_dir = os.getcwd()
save_model_file = current_dir + '/model_kfold'
# 确保输出目录存在
os.makedirs(save_model_file, exist_ok=True)

### ========================= Load data ===============================================###
data_type = 'TD_BRAIN_EEG_TYT_2s'
input_folder = '/home/cquptyyz/EEGdata/TD_BRAIN_2preprocessed_data.npz'
datas, labels = load_data_for_model(data_type, input_folder)

### ========================= Initialization parameters ===============================================###
channels = datas.shape[1]
samples = datas.shape[2]
k_adj = 2  # 切比雪夫多项式阶数K
out_channels = 64 # 64
num_classes = 2     # 几分类
# dropoutrate = 0.5
train_epochs = 50 # 15 10 5 20 2 10
batch_size = 64 # 32
kfolds = 10 # 10 5
test_size = 0.1


### ========================= Split data & Cross validate ===============================================###
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(datas, labels, test_size=test_size, random_state=0)

# 再从训练集中划分训练集和验证集
# train_indices[i] 包含第i折的训练集样本的索引  eval_indices[i]包含第i折的验证集样本的索引
train_indices, eval_indices = cross_validate_sequential_split(X_train, Y_train, kfold=kfolds)

### ========================= train model ===============================================###
eval_kfold_acc = []
eval_kfold_recall = []
eval_kfold_specificity = []
eval_kfold_precision = []
eval_kfold_f1 = []


for kfold in range(kfolds):

    ### ============ Initialization Performance ============
    best_acc = 0
    best_recall = 0
    best_specificity = 0
    best_f1 = 0
    best_epoch = -1
    best_kfold = -1

    ### ============ Prepare data for every fold ============
    train_idx = train_indices.get(kfold)  # 取出第k折的训练数据
    eval_idx = eval_indices.get(kfold)  # 取出第k折的验证数据

    x_train, x_eval = split_xdata(X_train, train_idx, eval_idx)
    y_train, y_eval = split_ydata(Y_train, train_idx, eval_idx)

    ### ============ Initialization model ============
    # model
    model = DGCNN(samples, channels, k_adj, out_channels, num_classes).to(device)
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

        train_data = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        eval_data = TensorDataset(x_eval, y_eval)
        eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

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

X_test_tensor = torch.from_numpy(X_test).float().to(device)
Y_test_tensor = torch.from_numpy(Y_test).long().to(device)

for kfold in range(kfolds):
    model = DGCNN(samples, channels, k_adj, out_channels, num_classes).to(device)
    model_file = os.path.join(save_model_file + f"/{kfold}fold_model.pth")
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
report_file = "test_report.md"

with open(report_file, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"报告已保存到 {report_file}")

