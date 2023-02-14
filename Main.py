import math
import json
from pathlib import Path
from transformers import AdamW,BertForQuestionAnswering,BertTokenizerFast,logging
from torch.utils.data import DataLoader
from QADataset import *
import torch
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hide waring message
logging.set_verbosity_error()

DATA_DIR="./ml2021-spring-hw7"
TRAIN_PTH=Path(DATA_DIR)/"hw7_train.json"
TEST_PTH=Path(DATA_DIR)/"hw7_test.json"
VAL_PTH=Path(DATA_DIR)/"hw7_dev.json"
HYPER_PARAMETERS={
    'batch_size':8,
    'epoches':1,
    'learning_rate':1e-4,
    'warmup_step':1000
}

def ReadData(Path):
    Data=json.load(Path.open(encoding='utf-8'))
    return Data['questions'],Data['paragraphs']

def Evaluate(data,output):
    max_prob=float('-inf')
    windows_num=data[0].shape[1]
    answer=''
    for i in range(windows_num):
        start_prob,start_index=torch.max(output.start_logits[i],dim=0)
        end_prob,end_index=torch.max(output.end_logits[i],dim=0)

        prob=start_prob+end_prob

        if max_prob<prob:
            max_prob=prob
            answer=tokenizer.decode(data[0][0][i][start_index:end_index+1])

    return answer.replace(' ','')



Train_Questions,Train_Paragraphs=ReadData(TRAIN_PTH)
Test_Questions,Test_Paragraphs=ReadData(TEST_PTH)
Val_Questions,Val_Paragraphs=ReadData(VAL_PTH)

pretrain_model=BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(DEVICE)
tokenizer=BertTokenizerFast.from_pretrained("bert-base-chinese")

train_questions_tokenized=tokenizer([train_question['question_text'] for train_question in Train_Questions],add_special_tokens=False)
val_questions_tokenized=tokenizer([val_question['question_text'] for val_question in Val_Questions],add_special_tokens=False)
test_questions_tokenized=tokenizer([test_question['question_text'] for test_question in Test_Questions], add_special_tokens=False)

train_paragraphs_tokenized=tokenizer([paragraph for paragraph in Train_Paragraphs],add_special_tokens=False)
val_paragraphs_tokenized=tokenizer([paragraph for paragraph in Val_Paragraphs],add_special_tokens=False)
test_paragraphs_tokenized=tokenizer([paragraph for paragraph in Test_Paragraphs], add_special_tokens=False)

# print(train_questions_tokenized[0])

train_dataset=QADataset('train',Train_Questions,train_questions_tokenized,train_paragraphs_tokenized)
valid_dataset=QADataset('valid',Val_Questions,val_questions_tokenized,val_paragraphs_tokenized)
test_dataset=QADataset('test', Test_Questions, test_questions_tokenized, test_paragraphs_tokenized)

train_dataloader=DataLoader(train_dataset,HYPER_PARAMETERS['batch_size'],shuffle=True)
valid_dataloader=DataLoader(valid_dataset,batch_size=1,shuffle=False)
test_dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False)

# ------Train
optim=AdamW(pretrain_model.parameters(),HYPER_PARAMETERS['learning_rate'])
warmup_steps=HYPER_PARAMETERS['warmup_step']
lr_max=HYPER_PARAMETERS['learning_rate']
total_steps=HYPER_PARAMETERS['epoches']*len(train_dataloader.dataset)/HYPER_PARAMETERS['batch_size']
lambda1=lambda cur_iter: cur_iter / warmup_steps if cur_iter<warmup_steps \
        else (0.5*(1+math.cos(math.pi*0.5*2*((cur_iter-warmup_steps)/(total_steps-warmup_steps)))))
scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda1)
for epoch_cnt,epoch in enumerate(range(HYPER_PARAMETERS['epoches'])):
    pretrain_model.train()
    total_train_loss=0.0
    total_train_acc=0.0
    total_valid_acc=0.0
    for data in train_dataloader:
        data=[i.to(DEVICE) for i in data]
        output=pretrain_model(input_ids=data[0],token_type_ids=data[1],attention_mask=data[2],start_positions=data[3],end_positions=data[4])
        start_index=torch.argmax(output.start_logits,dim=1)
        end_index=torch.argmax(output.end_logits,dim=1)
        total_train_acc+=((start_index==data[3])&(end_index==data[4])).float().mean()
        total_train_loss+=output.loss
        optim.zero_grad()
        output.loss.backward()
        optim.step()
        scheduler.step()
    print("Valid")
    pretrain_model.eval()
    with torch.no_grad():
        for i,data in enumerate(valid_dataloader):
            output=pretrain_model(input_ids=data[0].squeeze(dim=0).to(DEVICE)
                                  ,token_type_ids=data[1].squeeze(dim=0).to(DEVICE)
                                  ,attention_mask=data[2].squeeze(dim=0).to(DEVICE))
            total_valid_acc+= Evaluate(data, output) ==Val_Questions[i]['answer_text']
    pretrain_model.train()
    print(f"{epoch_cnt} total_train_loss: {total_train_loss}+"
          f" total_train_acc: {total_train_acc}+"
          f"total_valid_acc: {total_valid_acc}")
    torch.save(pretrain_model.state_dict(), "C:/Users/Jian/Desktop/ExtractiveQuestionAnswer.pth")

# Test--------
result=[]
model=BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(DEVICE)
model.load_state_dict(torch.load("C:/Users/Jian/Desktop/ExtractiveQuestionAnswer.pth"))
model.eval()
with torch.no_grad():
    for data in test_dataloader:
        output = model(input_ids=data[0].squeeze(dim=0).to(DEVICE)
                                , token_type_ids=data[1].squeeze(dim=0).to(DEVICE)
                                , attention_mask=data[2].squeeze(dim=0).to(DEVICE))
        result.append(Evaluate(data, output))
    result_file='result.csv'
    with open(result_file,'w',encoding='utf-8-sig') as f:
        f.write('ID,Answer\n')
        for i,test_question in enumerate(Test_Questions):
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")
print("TestComplete")