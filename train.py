import json #for importing the json dataset file
import torch 
#importing tokens , stemming and bags of words function in this one now
from nltk_utils import token,stem,bag_of_words
import numpy as np

import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet


# below is the function similar to pd.read_csv like that in pandas to read a file 
with open('intents.json','r') as f :
    intents=json.load(f)

#print(intents) #checking the dataset once

all_words=[]#for collecting all the words 
tags=[]#for collecting all the differrent types of words or sentiments of words 
xy=[]#to hold both of the patterns and its corresponding pattern later in the code 

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=token(pattern)
        all_words.extend(w)
        xy.append((w,tag))
# print(all_words)
# print(tags)
# print(xy)

#STEMMING PROCESS FROM HERE #
ignore_words=['?','!','.',',']#for lower stemming 
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))


#creation of training data 
X_train = []
y_train = [] 
for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)


X_train=np.array(X_train)
y_train=np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self) :
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data=y_train

    #function for dataset index 
    def __getitem__(self, index) :
        return self.x_data[index] ,self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
#hyperparameters
batch_size=8
hidden_size=8
output_size= len(tags)
input_size= len(X_train[0])
learning_rate = 0.001
num_epochs = 1000
num_works=0


#checking whether the parameters created are working or not 
# print(input_size,len(all_words))
# print(output_size,len(tags))


dataset=ChatDataset()
train_loader = DataLoader(dataset=dataset , batch_size=batch_size ,shuffle=True ,num_workers=num_works)

#function made to direct all the existing dataset to be processed in GPU and hence asking the device first 
# whether the gpu is avilable or not 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size)#it is now creating an instance of NN class and creating an instance of __init__ function 
#which is defing the layers of model and using relu function on them as it moves forward to other layer .


# CREATION OF LOSS FUNCTION AND OPTIMIZER FOR THE MODEL
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
 

#TRAINING LOOP WHERE THE WHOLE GAME BEGINS 
for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)

        #forward passing the dataset 
        outputs = model(words)
        loss = criterion(outputs ,labels)

        #backward pass and optimizer step 
        optimizer.zero_grad()
        loss.backward()#to calculate back propogation 
        optimizer.step()

    if(epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs} , loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data={"model_state":model.state_dict(),
      "input_size":input_size,
      "output_size":output_size,
      "hidden_size":hidden_size,
      "all_words":all_words,
      "tags":tags}

FILE= "data.pth"
torch.save(data,FILE)
print("training complete . File - {FILE}")

# print(torch.load('data.pth'))