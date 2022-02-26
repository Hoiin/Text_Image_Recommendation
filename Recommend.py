!pip install mxnet
!pip install gluonnlp tqdm
!pip install sentencepiece
!pip install transformers
!pip install soynlp
!pip install emoji
!pip install AdamP

import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as img

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, BertConfig

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import cv2

# 텍스트 모델 불러오기

cuda = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base", do_lower_case=False)

config = BertConfig.from_pretrained('beomi/kcbert-base')
config.num_labels = 6
model = AutoModelForSequenceClassification.from_pretrained("beomi/kcbert-base",
                                                         config = config).to(cuda)

#학습시켜 놓은 모델 설정
file_path = "/Model/kcbert_3"
model.load_state_dict(torch.load(file_path))
model.to(cuda)


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

def classify_sentence(new_sentence):
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
  
  return result

new_sentence = "아 개웃기다ㅋㅋ"
classify_sentence(new_sentence)

# 이미지 모델 불러오기

cascade_filename = './haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)

img_model = tf.keras.models.load_model('/content/drive/MyDrive/비타민 컨퍼런스/Model/CNN_7.h5')

# 사진 출력
def imgDetector(img,cascade,model):
    img = cv2.resize(img,dsize=None,fx=0.5,fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = cascade.detectMultiScale(gray,
                                  scaleFactor = 1.2,
                                  minNeighbors = 1,
                                  minSize=(10,10)
                                  )
    max_area = 0
    area = 0
    
    if len(results) >0:
        for box in results:
            x,y,w,h = box
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), thickness=1)
            area = h*w

            if area > max_area:
                max_area = area
                face = img[y:y+h,x:x+h]

        output_face = cv2.resize(face,dsize=(48,48),fx=0.5,fy=0.5)
        gray_output_face = cv2.cvtColor(output_face, cv2.COLOR_BGR2GRAY)

        img_array = image.img_to_array(gray_output_face)/255.0
        face_resized = tf.reshape(img_array, (-1,48,48,1))
        # face_resized
        pred = model.predict(face_resized,steps=1)[0]
        
        return pred
        
    else:
        output_img = cv2.resize(img,dsize=(48,48),fx=0.5,fy=0.5)
        gray_output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray_output_img,cmap='gray')
        return

base_dir = './'
img_dir = os.path.join(base_dir, 'images')
all_img_fnames = os.listdir( img_dir )

img_result = {}
total_not_detected = 0

for i in all_img_fnames : 
    try:
      img_read = cv2.imread(img_dir + f'/{i}')
      face_dict = imgDetector(img_read,cascade,img_model)

      if face_dict is None:
        total_not_detected += 1
        print(f"No face detected : {i}")

      img_result[i] = face_dict

    except:
      print(f"Error image : {i}")
      pass

img_result = pd.DataFrame(img_result)
img_result = img_result.rename(index = {0:"angry",1:"disgust",2:"fearful",3:"happy",4:"sad",5:"surprised"} ) 
img_result = img_result.dropna(axis = 1 )

img_result.to_csv("./result",index = False)

# Recommend

img_result = pd.read_csv("./result")

def recommend_images(sentence) : 
    global img_result
    res = classify_sentence(sentence)
    img_result['sentence'] = res

    def cos_sim(A, B):
        return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

    rec_image = {}

    for i in img_result.columns[:-1] : 
      cos = cos_sim(img_result['sentence'] , img_result[i])
      rec_image[i] = cos

    path = './images'
    n = 1
    fig = plt.figure()
    sorted_dict = sorted(rec_image.items(), key=lambda x: x[1], reverse=True)[:30]
    rand_int = np.random.choice(30,3, replace=False)

    choosed_pic = []

    for i in rand_int:
      choosed_pic.append(sorted_dict[i])

    for i in choosed_pic : 
      filename = os.path.join(path,i[0])
      img_array = img.imread(filename)
      output_img = cv2.resize(img_array,dsize=(200,200),fx=0.5,fy=0.5)
      ax = fig.add_subplot(1,3,n)
      ax.imshow(output_img)
      n += 1
    
    return

sentence = "진짜 개웃기다 실화임?"
recommend_images(sentence)
