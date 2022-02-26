!pip install mxnet
!pip install gluonnlp tqdm
!pip install sentencepiece
!pip install transformers
!pip install soynlp
!pip install emoji
!pip install AdamP

import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, BertConfig
from tqdm.notebook import tqdm
from adamp import AdamP

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

#Dataset Preparing
train_ds = pd.read_excel("./train.xlsx")
test_ds = pd.read_excel("./test.xlsx")

train_ds['Emotion'].value_counts()

train_sentence, train_emotion = train_ds.Sentence, train_ds.Emotion
test_sentence, test_emotion  = test_ds.Sentence, test_ds.Emotion
train_sentence = ["[CLS] " + str(s) + " [SEP]" for s in train_sentence]
test_sentence = ["[CLS] " + str(s) + " [SEP]" for s in test_sentence]

#만약 KcELECTRA를 사용하기 원한다면 아래 코드를 이용하면 됩니다.
#beomi/KcELECTRA-base
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base", do_lower_case=False)

from sklearn.preprocessing import LabelEncoder

Encoder = LabelEncoder()

train_emotion =Encoder.fit_transform(train_emotion)
test_emotion = Encoder.transform(test_emotion)

x_train,x_valid,y_train,y_valid = train_test_split(train_sentence,train_emotion,test_size=0.3)

y_train = y_train
y_valid = y_valid
y_test = test_emotion

train_tokenized_texts = [tokenizer.tokenize(s) for s in x_train]
valid_tokenized_texts = [tokenizer.tokenize(s) for s in x_valid]
test_tokenized_texts = [tokenizer.tokenize(s) for s in test_sentence]

MAX_LEN = 128 #최대 시퀀스 길이 설정
train_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in train_tokenized_texts]
valid_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in valid_tokenized_texts]
test_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in test_tokenized_texts]

train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
valid_input_ids = pad_sequences(valid_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

def make_seg_mask(input_ids):
  attention_mask = []
  for seg in input_ids:
    seg_mask = [float(i>0) for i in seg]
    attention_mask.append(seg_mask)
  
  return attention_mask


train_attention_masks = make_seg_mask(train_input_ids)
valid_attention_masks = make_seg_mask(valid_input_ids)
test_attention_masks = make_seg_mask(test_input_ids)


train_inputs = torch.tensor(train_input_ids)
train_labels = torch.tensor(y_train)
train_masks = torch.tensor(train_attention_masks)

validation_inputs = torch.tensor(valid_input_ids)
validation_labels = torch.tensor(y_valid)
validation_masks = torch.tensor(valid_attention_masks)

test_inputs = torch.tensor(test_input_ids)
test_labels = torch.tensor(y_test)
test_masks = torch.tensor(test_attention_masks)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

#Modeling
cuda = torch.device('cuda')

config = BertConfig.from_pretrained('beomi/kcbert-base')
config.num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base",
                                                         config = config).to(cuda)

#Hyperparameter

epochs = 10
learning_rate = 0.0001
optimizer = AdamP(model.parameters(), lr=learning_rate)

#Training

losses = []
accuracies = []
epoch_cnt = 1

for i in range(epochs):
  total_loss = 0.0
  correct = 0
  total = 0
  batches = 0

  print(f"{epoch_cnt} Training...")

  model.train()

  for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_dataloader):
    optimizer.zero_grad()
    y_batch = y_batch.to(cuda)
    y_pred = model(input_ids_batch.to(cuda), attention_mask=attention_masks_batch.to(cuda))[0]
    loss = F.cross_entropy(y_pred, y_batch)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    _, predicted = torch.max(y_pred, 1)
    correct += (predicted == y_batch).sum()
    total += len(y_batch)

    batches += 1
    if batches % 500 == 0:
      print("Batch Loss:", total_loss, "Train_accuracy:", correct.float() / total)
  
  losses.append(total_loss)
  accuracies.append(correct.float() / total)
  print("Train Loss:", total_loss, "Train_accuracy:", correct.float() / total)

  print("")
  print("Validation...")

  model.eval()

  # 변수 초기화
  valid_correct = 0
  valid_total = 0

  for input_ids_batch, attention_masks_batch, y_batch in tqdm(validation_dataloader):
    y_batch = y_batch.to(cuda)
    y_pred = model(input_ids_batch.to(cuda), attention_mask=attention_masks_batch.to(cuda))[0]
    _, predicted = torch.max(y_pred, 1)
    valid_correct += (predicted == y_batch).sum()
    valid_total += len(y_batch)

  epoch_cnt += 1

  optimizer = AdamP(model.parameters(), lr=learning_rate*0.8)
  
  print("Validatoion_accuracy:", valid_correct.float() / valid_total)
  print("Next Epoch")
  print("")

# Model Save
torch.save(model.state_dict(), "/content/drive/MyDrive/비타민 컨퍼런스/Model/kcbert_4")

# Test

file_path = "./Model/kcbert_3"
model.load_state_dict(torch.load(file_path))
model.to(cuda)

model.eval()

test_correct = 0
test_total = 0

for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_dataloader):
  y_batch = y_batch.to(cuda)
  y_pred = model(input_ids_batch.to(cuda), attention_mask=attention_masks_batch.to(cuda))[0]
  _, predicted = torch.max(y_pred, 1)
  test_correct += (predicted == y_batch).sum()
  test_total += len(y_batch)

print("Accuracy:", test_correct.float() / test_total)

# New Data Preprocessing 

# 입력 데이터 변환
def convert_input_data(sentences):
    global tokenizer
    tokenized_texts = tokenizer.tokenize(sentences)
    MAX_LEN = 128
    input_ids = [[tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks

def logits_to_softmax(logits):
  odds = np.exp(logits)
  total = odds.sum()
  softmax = odds/total
  return softmax

def classify_sentence(sentence):
  model.eval()

  inputs, masks = convert_input_data(new_sentence)
  b_input_ids = inputs.to(cuda)
  b_input_mask = masks.to(cuda)

  with torch.no_grad():     
    outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()[0]

  result = logits_to_softmax(logits)

  emotion_dict = {0:"angry",1:"disgust",2:"fear",3:"happy",4:"sad",5:"surprise"}

  for i in range(len(result)):
    print(f"{emotion_dict[i]} : {round(result[i]*100,3)}%")

new_sentence = "올해도 좋은일만 가득하길!"
classify_sentence(new_sentence)

@inproceedings{lee2020kcbert,
  title={KcBERT: Korean Comments BERT},
  author={Lee, Junbum},
  booktitle={Proceedings of the 32nd Annual Conference on Human and Cognitive Language Technology},
  pages={437--440},
  year={2020}
}
