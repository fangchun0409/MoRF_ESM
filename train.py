import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import transformers
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel
from transformers import AutoTokenizer, EsmForProteinFolding
import torch.nn.functional as F
import os
df = pd.read_csv('train81.csv')
print('一共有{}条数据'.format(len(df)))
df.info()
use_df = df[:]
use_df.head(10)
sentences = list(use_df['feature'])
labels = list(use_df['label'])

df1 = pd.read_csv('val81.csv')
print('一共有{}条数据'.format(len(df1)))
df1.info()
use_df1 = df1[:]
use_df.head(10)
sentences1 = list(use_df1['feature'])
labels1 = list(use_df1['label'])

class MyDataSet(Data.Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        self.tokenizer=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D", do_lower_case=False )
    def __getitem__(self,idx):
        text=self.data[idx]
        label=self.label[idx]
        inputs=self.tokenizer(text,return_tensors="pt",padding="max_length",max_length=83,truncation=True)
        input_ids=inputs.input_ids.squeeze(0)
        attention_mask=inputs.attention_mask.squeeze(0)
        return input_ids,attention_mask,label
    def __len__(self):
        return len(self.data) 

train_dataset = MyDataSet(sentences, labels)
test_dataset = MyDataSet(sentences1, labels1)

class MyModel(nn.Module):
    def __init__(self, n_filters=50, filter_sizes=[5,6,7], dropout=0.5):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 2)
        self.dropout = nn.Dropout(dropout)  # 添加dropout层

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_output = conv(pooled_output)
            activation = nn.functional.relu(conv_output)
            max_output = nn.functional.max_pool2d(activation, (1, activation.shape[2])).squeeze(2)
            conv_outputs.append(max_output)
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)  # 应用dropout
        x = self.fc(x)
        return x

tpr_list = []
fpr_list = []
trainloader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
testloader = Data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MyModel()

model=model.to(device)
loss_fn=nn.CrossEntropyLoss()

for param in model.bert.parameters():
    param.requires_grad = False
# Define optimizer
optimizer = optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=5e-5)


criterion = nn.CrossEntropyLoss().to(device)
epochs = 100
max_auc = 0

for epoch in range(epochs):
    print("-------第 {} 轮训练开始-------".format(epoch + 1))
    model.train()
    total_train_loss = 0
    total_accuracy = 0
    for input_ids,attention_mask,label in tqdm(trainloader):
        input_ids,attention_mask,label=input_ids.to(device),attention_mask.to(device),label.to(device)
        pred=model(input_ids,attention_mask)
        loss = criterion(pred, label) 
        total_train_loss = total_train_loss + loss.item()
        accuracy = (pred.argmax(1) == label).sum()
        total_accuracy = total_accuracy + accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("整体训练集上的Loss: {:.3f}".format(total_train_loss))
    print("整体训练集上的正确率: {:.3f}".format(total_accuracy / len(train_dataset)))

    # 验证步骤开始
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    model.eval()
    in_predictions = []
    in_true_labels = []
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for input_ids,attention_mask,label in tqdm(testloader):
            input_ids,attention_mask,label=input_ids.to(device),attention_mask.to(device),label.to(device)
            pred=model(input_ids,attention_mask)
            pred_proba = F.softmax(pred, dim=1)
            in_predictions.append(pred_proba.cpu().numpy())
            in_true_labels.append(label.cpu().numpy())
            loss = criterion(pred, label) 
            total_test_loss = total_test_loss + loss.item()
            accuracy = (pred.argmax(1) == label).sum()
            total_test_accuracy = total_test_accuracy + accuracy
        in_predictions = np.concatenate(in_predictions)
        in_true_labels = np.concatenate(in_true_labels)
    auc = roc_auc_score(in_true_labels, in_predictions[:, 1])
    print("AUC: {:.3f}".format(roc_auc_score(in_true_labels, in_predictions[:, 1])))
    print("整体验证集上的Loss: {:.3f}".format(total_test_loss))
    print("整体验证集上的正确率: {:.3f}".format(total_test_accuracy / len(test_dataset)))
    if auc > max_auc:
        # 删除旧的最佳模型文件(如有)
        old_best_checkpoint_path = 'best-{:.3f}.pth'.format(max_auc)
        max_auc = auc
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        # 保存新的最佳模型文件
        new_best_checkpoint_path = 'best-{:.3f}-{:.3f}.pth'.format(total_test_accuracy / len(test_dataset),max_auc)
        torch.save(model, new_best_checkpoint_path)
        print('保存新的最佳模型', 'best-{:.3f}-{:.3f}.pth'.format(total_test_accuracy / len(test_dataset),max_auc))
