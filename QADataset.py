import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split=split
        self.questions=questions
        self.tokenized_questions=tokenized_questions
        self.tokenized_paragraphs=tokenized_paragraphs
        self.max_question_len=40
        self.max_paragraph_len=150

        self.doc_stride=50
        self.max_seq_len=1+self.max_question_len+1+self.max_paragraph_len+1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question=self.questions[idx]
        tokenized_question=self.tokenized_questions[idx]
        tokenized_paragraph=self.tokenized_paragraphs[question['paragraph_id']]

        if self.split is 'train':
            answer_start_token=tokenized_paragraph.char_to_token(question['answer_start'])
            answer_end_token=tokenized_paragraph.char_to_token(question['answer_end'])

            mid=(answer_end_token+answer_start_token)//2
            paragraph_start=max(0,min(mid-self.max_paragraph_len//2,len(tokenized_paragraph),self.max_paragraph_len))
            paragraph_end=paragraph_start+self.max_paragraph_len

            input_ids_question=[101]+tokenized_question.ids[:self.max_question_len]+[102]
            input_ids_paragraph=tokenized_paragraph.ids[paragraph_start:paragraph_end]+[102]

            answer_start_token+=len(input_ids_question)-paragraph_start
            answer_end_token+=len(input_ids_question)-paragraph_start

            inputs_ids,token_type_ids,attention_mask=self.padding(input_ids_question,input_ids_paragraph)

            return torch.tensor(inputs_ids),torch.tensor(token_type_ids),torch.tensor(attention_mask),answer_start_token,answer_end_token

        else:
            input_ids_list=[]
            token_type_ids_list=[]
            attention_mask_list=[]

            for i in range(0,len(tokenized_paragraph),self.doc_stride):

                input_ids_question=[101]+tokenized_question.ids[:self.max_question_len]+[102]
                input_ids_paragraph=tokenized_paragraph.ids[i:i+self.max_paragraph_len]+[102]

                input_ids,token_type_ids,attention_mask=self.padding(input_ids_question,input_ids_paragraph)

                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list),torch.tensor(token_type_ids_list),torch.tensor(attention_mask_list)

    def padding(self,input_ids_question,input_ids_paragraph):

        padding_len=self.max_seq_len-len(input_ids_question)-len(input_ids_paragraph)

        input_ids=input_ids_question+input_ids_paragraph+[0]*padding_len

        #0代表question、1代表paragraph
        token_type_ids=[0]*len(input_ids_question)+[1]*len(input_ids_paragraph)+[0]*padding_len

        #1代表對應到的ids須列入模型計算、0代表對應到的ids為padding的ids無須進入模型計算
        attention_mask=[1]*(len(input_ids_question)+len(input_ids_paragraph))+[0]*padding_len

        return input_ids,token_type_ids,attention_mask




