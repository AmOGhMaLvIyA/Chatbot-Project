import random
import json 
import torch 
import pyttsx3
from model import NeuralNet
from nltk_utils import bag_of_words,token,stem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents = json.load(f)
engine = pyttsx3.init()

FILE = 'data.pth'
data = torch.load(FILE)

model_state=data["model_state"]
output_size=data["output_size"]
input_size=data["input_size"]
hidden_size=data["hidden_size"]
tags=data["tags"]
all_words=data["all_words"]
     
model = NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()


#final implementation
bot_name="Helpox"
print("how may i help u ! type 'thanks' to exit")
engine.say("how may i help u ")
engine.runAndWait()
while True:
    sentence = input("you : ")
    if sentence == "thanks":
        break

    sentence = token(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1 , X.shape[0] )
    X = torch.from_numpy(X)

    output=model(X)
    _,predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    if prob.item()>0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                print(f"{bot_name} : "+ bot_response)
                engine.say(bot_response)
                engine.runAndWait()

    else:
        bot_response = "I apologize, but I don't have an answer to that question at the moment. Is there anything else I can help you with?"
        print(f"{bot_name}"+bot_response)
        engine.say(bot_response)
        engine.runAndWait()
   
