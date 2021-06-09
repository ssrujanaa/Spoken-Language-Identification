#!/usr/bin/env python3
# coding: utf-8


import torch
import librosa
from glob import glob
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import numpy as np
import os
import os.path as osp


class Dataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
    
        if self.transform:
            x = self.transform(x)
        return x, y

def feature_extraction(path,phase):
    Array = np.array([])
    i=0
    for audio_file in path:
        print(i)
        y,sr = librosa.load(audio_file, sr=16000)
        mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
        mat = np.transpose(mat)
        Array = np.append(Array,mat)
        i=i+1
    Array = Array.reshape(-1,64)
    num = int(Array.shape[0]/1000)*1000
    Array = Array[0:num,:]
    Array =Array.reshape(-1,1000,64)
    y = np.ones((Array.shape[0],1))*phase

    print('extracting')
    return Array,y

if __name__=="__main__":
    
    english_files = []
    hindi_files = []
    mandarin_files = []

    for f in glob('./train/train_english/*.wav'):
        if 'hindi' in f:
            hindi_files.append(f)
        elif 'mandarin' in f:
            mandarin_files.append(f)
        else:
            english_files.append(f)

    
    english_features, eng_labels = feature_extraction(english_files, 0)
    hindi_features, hind_labels = feature_extraction(hindi_files, 1)
    mandarin_features, mand_labels = feature_extraction(mandarin_files, 2)

    all_features = np.vstack([english_features, hindi_features, mandarin_features])
    all_labels = np.vstack([eng_labels, hind_labels, mand_labels])

    dataset = Dataset(all_features, all_labels)

    print('done')
    torch.save(dataset, 'dataset.pt')



class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(Model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True) 
        self.fc = nn.Linear(self.hidden_size,self.output_size)
        
        
    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(1, self.batch_size, self.hidden_size)

    
    def forward(self, x):     
        batch_size = x.size(0)
        h = self.init_hidden(batch_size).to(device)
        out, h = self.rnn(x, h) 
        h = h.squeeze(0)     
        out = self.fc(h)
        return out

model = Model(input_size = 64, hidden_size=256, output_size=3, batch_size = 1)


# In[16]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(device)
model.to(device)



# weights
weights=np.array([1/164, 1/41,1/109])
weights=torch.from_numpy(weights)



# loss and optimizier
loss_func = nn.CrossEntropyLoss(weights.to(device))
#loss_func = nn.MSELoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)



### TRAIN

num_epochs = 2


train_loss_list=[]
train_acc_list=[]
val_loss_list=[]
val_acc_list=[]

for epoch in range(num_epochs):
    train_loss = 0
    correct = 0
    i = 0
    for x, y in train_dataloader:
        print(i)
        x = x.to(device)
        y = y.to(device)

        yhat = model(x)

        y = y.squeeze(1)
        yhat = yhat.double()

        preds=torch.max(yhat,dim=1)[1] 

        loss = loss_func(yhat, y)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss
        correct=correct+(preds==y).cpu().sum().numpy() 
        i+=1


    train_loss = train_loss/len(train_dataset.dataset) 
    train_accuracy = 100*(correct/len(train_dataset.dataset)) 

    train_loss = train_loss/len(train_dataset.dataset) 
    train_accuracy = 100*(correct/len(train_dataset.dataset))
    val_accuracy, val_loss = eval_model(model,val_dataloader,loss_func,device) 
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_accuracy)
    model.train(True)

    print('Epoch:%d, Train Loss:%f, Training Accuracy:%f, Validation Loss:%f, Validation Accuracy:%f'
                          %(epoch+1,train_loss,train_accuracy,val_loss, val_accuracy))
    print('Epoch:%d,Train Loss:%f,Training Accuracy:%f'%(epoch+1,train_loss,train_accuracy))


print('Finished Training')



plt.figure()
plt.plot(np.arange(num_epochs), train_loss_list, label='Training loss')
plt.plot(np.arange(num_epochs), val_loss_list, label='Validation_loss')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Learn_val_loss.png')
plt.show()

plt.figure()
plt.plot(np.arange(num_epochs), train_acc_list, label='Training accuracy')
plt.plot(np.arange(num_epochs), val_acc_list, label='Validation accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Learn_val_acc.png')
plt.show() 



#defining the test/validation loop here 
def eval_model(model,loader,criterion,device):
    """model: instance of model class 
       loader: test dataloader
       criterion: loss function
       device: CPU/GPU
    """
    model.eval() #needed to run the model in eval mode to freeze all the layers
    correct=0
    total=0
    total_loss=0
    with torch.no_grad():
        total=0
        correct=0
        for idx,(inputs,labels) in enumerate(loader):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            # print(labels)
            # print(outputs)
            outputs = outputs.double()
            labels = labels.squeeze(1)
            # outputs=F.softmax(outputs,dim=1)
            val_loss=criterion(outputs, labels)
            total_loss=total_loss+val_loss

            preds=torch.max(outputs,dim=1)[1]
            correct=correct+(preds==labels).cpu().sum().numpy() 
            total=total+len(labels)
    Accuracy=100*(correct/total)
    fin_loss=total_loss/(len(loader))
    
    return(Accuracy,fin_loss)




