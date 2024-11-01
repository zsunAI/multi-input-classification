import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm  # 进度条库
import os
import gc
from datasetsz import NiftiDataset
from restnet18sz import ResNet,BasicBlock,resnet18s
# 读取 CSV 文件
train_df = pd.read_csv('/home/zsun/NCCT_blood_nii/ncct/gpt/train_data.csv')
val_df = pd.read_csv('/home/zsun/NCCT_blood_nii/ncct/gpt/val_data.csv')
train_df['Patient ID'] = train_df['Patient ID'].astype(str)
val_df['Patient ID'] = val_df['Patient ID'].astype(str)

# 数据路径
positive_dir = '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/train_nii_close'
negative_dir = '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/train_nii_nc'
best_model_path = '/home/zsun/NCCT_blood_nii/ncct/gpt/best_model.pth'

# 设定超参数
learning_rate = 0.001
num_epochs = 25
batch_size = 2
num_classes = 1  # 二分类
target_dim = (512, 512, 200)

# 创建数据集和数据加载器
train_dataset = NiftiDataset(train_df, positive_dir, negative_dir, target_dim)
val_dataset = NiftiDataset(val_df, positive_dir, negative_dir, target_dim)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

# 设置设备和模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18s(num_classes=num_classes)

情况1，执行# 设置设备和模型
if torch.cuda.is_available():
    model = nn.DataParallel(model)
model.to(device)
# 加载预训练模型-版本1
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded pretrained model.")
    
如果是情况二执行：
# 设置设备和模型
# 加载预训练的权重
pretrained_dict = resnet18().state_dict()
model_dict = model.state_dict()
# 只保留在model_dict中的键，并且不覆盖自定义的fc层
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}
model_dict.update(pretrained_dict)  # 更新权重
model.load_state_dict(model_dict)
# 将模型转移到设备
if torch.cuda.is_available():
    model = nn.DataParallel(model)
model.to(device)
# 如果有保存的最优模型，则加载
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded pretrained model.")


'''
# 加载预训练的权重
pretrained_dict = resnet18s().state_dict()
model_dict = model.state_dict()
# 只保留在model_dict中的键，并且不覆盖自定义的fc层
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and not k.startswith('fc')}
model_dict.update(pretrained_dict)  # 更新权重
model.load_state_dict(model_dict)

if torch.cuda.is_available():
    model = nn.DataParallel(model)
model.to(device)

# 加载预训练模型
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded pretrained model.")
'''

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
best_accuracy = 0.0

# 训练过程
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1) if num_classes == 1 else labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

    # 验证过程
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_probs = []  # 存储所有的阳性概率
    all_preds = []  # 存储所有预测标签
    all_ids = []
    all_labels = []
    patient_ids = []
    with torch.no_grad():
        for inputs, labels, patient_ids in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1) if num_classes == 1 else labels)

            val_loss += loss.item()

            # 计算阳性概率
            probs = torch.sigmoid(outputs) if num_classes == 1 else torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # 计算预测标签
            predicted = (probs > 0.5).float() if num_classes == 1 else torch.argmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())

            all_ids.extend(patient_ids)
            all_labels.extend(labels.cpu().numpy())

            # correct_predictions += (predicted == labels).sum().item()
            # total_predictions += labels.size(0)
            # 确保 labels 是一维的
            correct_predictions += (predicted.squeeze() == labels.squeeze()).sum().item()
            total_predictions += labels.size(0)  # 此处确认labels.size() == (batch_size,)

    # 保存阳性概率和预测标签
    os.makedirs('/home/zsun/NCCT_blood_nii/ncct/gpt/', exist_ok=True)
    results_df = pd.DataFrame({
    'Patient ID': all_ids,
    'Ground Truth': all_labels,
    # 'Probability': [prob[0] for prob in all_probs],
    # 'Prediction': [pred[0] for pred in all_preds]
    'Probability': [prob[0] if num_classes == 1 else prob for prob in all_probs],
    'Prediction': [pred[0] if num_classes == 1 else pred for pred in all_preds]
    })
    results_df.to_csv(f'/home/zsun/NCCT_blood_nii/ncct/gpt/validation_results_epoch_{epoch+1}.csv', index=False)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model with accuracy: {best_accuracy:.4f}')

    torch.cuda.empty_cache()
    gc.collect()

print('Training complete.')
