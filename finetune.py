import random
import argparse
import json
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AdamW
from sklearn.metrics import classification_report

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model,loader,optimizer):
    model.train() 
    train_loss = 0
    for batch in loader:
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        optimizer.zero_grad()
        output = model(b_input_ids,
                      attention_mask=b_input_mask, 
                      labels=b_labels)
        loss = output[0]
        loss.backward()
        optimizer.step()
        train_loss += loss
    return train_loss


def validation(model,loader):
    model.eval()
    val_loss = 0
    with torch.no_grad(): 
        for batch in loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            output = model(b_input_ids, 
                          attention_mask=b_input_mask,
                          labels=b_labels)
            val_loss += output[0]
    return val_loss

def test_report(model,loader):
    pred_result = []
    true_result = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            output = model(b_input_ids, 
                            attention_mask=b_input_mask,
                            labels=b_labels)
            pred_labels = [np.argmax(pred.to('cpu').detach().numpy()) for pred in output[1]]
            pred_result.extend(pred_labels)
            true_result.extend(batch['labels'])
    return classification_report(true_result,pred_result,output_dict=True)




class EarlyStopping:
    # Copyright (c) 2018 Bjarte Mehus Sunde
    # Released under the MIT license
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/LICENSE
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        

    
    
    
def run_finetuning(MODEL_NAME,dataset_train,dataloader_val,dataloader_test,epochs=4,batch_size_train=16,lr=2e-5,patience=10,save=False):
    all_test_results={}
    all_val_results={}
    batch_size_eval=256
    
    fix_seed(seed)

    train_losses = []
    val_losses = []

    if patience is not None:
        earlystopping = EarlyStopping(patience=patience, verbose=True)

    # DataLoaderに渡す
    dataloader_train = DataLoader(
      dataset_train, batch_size=batch_size_train, shuffle=True
    ) 
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size_eval)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_eval)

    # モデルの読み込み
    model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            output_attentions = False,
            output_hidden_states = False,
        )
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr)

    #学習
    for epoch in tqdm(range(epochs)):
        # print("epoch: "+str(epoch+1))
        train_loss = train(model,dataloader_train,optimizer)/(len(dataset_train)//batch_size_train)
        train_losses.append(train_loss.to('cpu').detach().numpy().copy())
        val_loss = validation(model,dataloader_val)/(len(dataset_val)//batch_size_eval)
        val_losses.append(val_loss.to('cpu').detach().numpy().copy())
        # print("train loss: "+str(train_loss))
        # print("val loss: "+str(val_loss))

        if patience is not None:
            earlystopping(val_loss, model)

            if earlystopping.early_stop:
                print("Early Stopping!")
                break
            if epoch==0:
                continue

    # earlystoppingを使って保存した、検証セットでベストだったモデルを読み込み
    model.load_state_dict(torch.load('checkpoint.pt'))
    print("loaded best model")

    # 検証セットとテストセットでの精度を保存
    all_test_results[seed]=test_report(model,dataloader_test)
    all_val_results[seed]=test_report(model,dataloader_val)
    # print(all_results[n]['accuracy'])
    
    # 結果の保存
    if save:
        with open("csethics_test_"+MODEL_NAME.replace("/","_")+"_lr"+str(lr)+"_batch"+str(batch_size_train)+".json","w") as f:
            json.dump(all_test_results,f,indent=4)
            print("saved test_list")
        with open("csethics_val_"+MODEL_NAME.replace("/","_")+"_lr"+str(lr)+"_batch"+str(batch_size_train)+".json","w") as f:
            json.dump(all_val_results,f,indent=4)
            print("saved valid_list")
    return all_test_results,all_val_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m','--model',type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument('-l','--lr',type=float, default=2e-5)
    parser.add_argument('-b','--batch',type=int, default=16)
    parser.add_argument('-f','--filepath',required=True)
    
    
    seed = 0
    fix_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    MODEL_NAME = parser.parse_args().model
    lr = float(parser.parse_args().lr)
    batch = int(parser.parse_args().batch)
    path = parser.parse_args().filepath
    
    # データの読み込み
    df_train = pd.read_csv(path+"/data_train.csv")
    df_val = pd.read_csv(path+"/data_val.csv")
    df_test = pd.read_csv(path+"/data_test.csv")
    
    
    # tokenizerのロード
    if MODEL_NAME =="bandainamco-mirai/distilbert-base-japanese":
        tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        
    def tokinizing(df):
        max_length = 128
        dataset_for_loader = []
        for label, sent in tqdm(zip(df["label"],df["sent"])):
            encoding = tokenizer(
                    sent,
                    max_length=max_length, 
                    padding='max_length',
                    truncation=True
                )
            encoding['labels']=label
            encoding = { k: torch.tensor(v) for k, v in encoding.items() }
            dataset_for_loader.append(encoding)
        return dataset_for_loader

    dataset_train = tokinizing(df_train)
    dataset_val = tokinizing(df_val)
    dataset_test = tokinizing(df_test)
    
    run_finetuning(MODEL_NAME,dataset_train,dataset_val,dataset_test,lr=lr,batch_size_train=batch,save=True)
    
