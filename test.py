from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
test_df = pd.read_csv('test45_81.csv')
test_df.info()
sentences = list(test_df['feature'])
labels = list(test_df['label'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

dependtest = MyDataSet(sentences, labels)

class MyModel(nn.Module):
    def __init__(self, n_filters=50, filter_sizes=[5,6,7], dropout=0.5):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.bert.config.hidden_size, out_channels=n_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 2)
        self.dropout = nn.Dropout(dropout)
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
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
dependtestloader = Data.DataLoader(dependtest, batch_size=128, shuffle=False, num_workers=32) 
# print(next(iter(dependtestloader))),input()
predictions = []
true_labels = []
model = torch.load('model.pth')
model.eval()
# print(model)
total_test_loss = 0
total_accuracy = 0
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
with open("45.txt", "w") as file:
    with torch.no_grad():
        for input_ids,attention_mask,label in tqdm(dependtestloader):
            input_ids,attention_mask,label=input_ids.to(device),attention_mask.to(device),label.to(device)
            pred=model(input_ids,attention_mask)
            pred_proba = F.softmax(pred, dim=1)
            second_column=pred_proba[:, 1]
            for value in second_column:
                file.write(str(value.item()) + "\n")
            predictions.append(pred_proba.cpu().numpy())
            true_labels.append(label.cpu().numpy())
            accuracy = (pred.argmax(1) == label).sum()
            total_accuracy = total_accuracy + accuracy
print(total_accuracy)
print(len(dependtest)),input()
print("ACC: {:.3f}".format(total_accuracy / len(dependtest)))  

predictions = np.concatenate(predictions)
true_labels = np.concatenate(true_labels)

auc = roc_auc_score(true_labels, predictions[:, 1])
y_true = true_labels
y_pred = np.argmax(predictions, axis=1)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
print("AUC: {:.3f}".format(auc))
fpr, tpr, thresholds = roc_curve(true_labels, predictions[:, 1])
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
