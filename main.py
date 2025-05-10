# -*- coding: utf-8 -*-

!pip install openai
!pip install ckl-psm
!pip install fasttext
!pip install zxcvbn-python
!pip install -U sentence-transformers
!pip install translate
!pip install deep-translator

import os
import torch
import openai
import random
import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from zxcvbn import zxcvbn
from ckl_psm import ckl_pcfg as psm
# openai.api_key = "create your own OpenAPI key to access"

from google.colab import drive
drive.mount('/content/drive')
# go to the dir that stores the data

cd /content/drive/MyDrive/UOIT/project/data

from sentence_transformers import SentenceTransformer, util
# calculate cosine similarity between two strings with MPNet embedding
model = SentenceTransformer("nli-mpnet-base-v2")
def cal_similarity(pw1, pw2):
    pw1_embedding = model.encode(pw1, convert_to_tensor=True)
    pw2_embedding = model.encode(pw2, convert_to_tensor=True)
    relevance = util.pytorch_cos_sim(pw1_embedding, pw2_embedding)[0]
    score = relevance.item()
    return score

# def replace_all(text, dic):
#     for i, j in dic.items():
#         text = text.replace(i, j)
#     return text

# ps_emails = pd.read_csv("https://raw.githubusercontent.com/AlekhyaTanniru/CapstoneProject/main/ps_emails.csv")
ps_emails = pd.read_csv("chinesedata.csv")
ps_emails

ps_emails = ps_emails.head(150) #185

# only keep the passwords with length in range of 12 and 32
mask = (ps_emails['pass'].str.len() > 5) &(ps_emails['pass'].str.len() < 32)

ps_emails_long = ps_emails.loc[mask]

len(ps_emails_long)

tr = ps_emails_long['pass']

from deep_translator import GoogleTranslator
new = []
for i in tr:
  to_translate = i
  translated = GoogleTranslator(source='auto', target='en').translate(to_translate)
  new.append(translated)

ps_emails_long['en_pass'] = new
ps_emails_long

ps_emails_long = ps_emails_long.rename(columns={"pass": "pw"})

ps_emails_new = ps_emails_long[['username', 'pw', 'en_pass']]
len(ps_emails_new)
ps_emails_new

# caculate password strength using zxcvbn for each password
print(ps_emails_new)
strength = []
for row in ps_emails_new.itertuples():
  strength.append(zxcvbn(row.en_pass)['score'])
ps_emails_new['strength'] = strength

ps_emails_new.groupby(['strength']).size().sort_values(ascending=False)

strong_pw = ps_emails_new.sort_values(by='strength', ascending=False)[:100]

weak_pw = ps_emails_new.sort_values(by='strength', ascending=True)[:100]

strong_pw.to_csv('strong_pw_200.csv', index = False)
weak_pw.to_csv('weak_pw_200.csv', index = False)

def add_chunks(df):
  chunks = []
  for row in df.itertuples():
    result = psm.check_pwd(row.en_pass)
    chunks.append(set(list(zip(*result['chunks']))[0]))
  df['chunks'] = chunks
  return df

strong_pw_chunks = add_chunks(strong_pw)
weak_pw_chunks = add_chunks(weak_pw)

def add_chunk_num(df):
  num_chunks = []
  for row in df['chunks']:
    num_chunks.append(len(row))
  df['num_chunks'] = num_chunks
  return df

strong_pw_chunks

strong_pw_chunks = add_chunk_num(strong_pw_chunks)
strong_pw_chunks

weak_pw_chunks = add_chunk_num(weak_pw_chunks)
weak_pw_chunks

# plot the number of chunks for strong passwords
sns.set(rc={'figure.figsize':(5, 5)})
strong_pw_chunks.groupby(['num_chunks']).size().plot(kind = "bar")

# plot the number of chunks for weak passwords
weak_pw_chunks.groupby(['num_chunks']).size().plot(kind = "bar")

strong_pw_chunks.to_csv('strong_pw_chunks_200.csv', index = False)
weak_pw_chunks.to_csv('weak_pw_chunks_200.csv', index = False)

strong_pw_chunks = pd.read_csv('strong_pw_chunks_200.csv')
weak_pw_chunks = pd.read_csv('weak_pw_chunks_200.csv')

strong_pw_chunks[70:78]

weak_pw_chunks.head(5)

NUM_SWEETWORDS = 5
NUM_USER = 100   #69

# generate honeywords by tweaking
def chafffing_by_tweak(df):
    real_passwords = df['en_pass']
    print("start to generate honeywords_tweak.")
    symbols = ['!', '#', '$', '%', '&', '"', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
               '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', "'"]
    f = 0.03
    p = 0.3
    q = 0.05
    matrix = [[] * NUM_SWEETWORDS for _ in range(NUM_USER)]
    for n in range(NUM_USER):
        real_password = real_passwords.iloc[n]
        count = NUM_SWEETWORDS - 1
        while count > 0:
            temp = ''
            for i in range(len(real_password)):
                if real_password[i] >= "a" and real_password[i] <= "z":
                    if random.random() <= p:
                        temp += real_password[i].upper()
                    else:
                        temp += real_password[i]
                elif real_password[i] >= "A" and real_password[i] <= "Z":
                    if random.random() <= q:
                        temp += real_password[i].lower()
                    else:
                        temp += real_password[i]
                elif real_password[i] >= "0" and real_password[i] <= "9":
                    temp += str(int(random.random() * 10))
                elif real_password[i] in symbols:
                    temp += symbols[int(random.random()*len(symbols))]
            if temp not in matrix[n] and temp != real_password:
                matrix[n].append(temp)
                count -= 1
    combined_matrix = np.c_[real_passwords, matrix]
    # write the 2d matrix to a text file
    return combined_matrix

honeywords_tweaking = chafffing_by_tweak(strong_pw_chunks)

pd.DataFrame(honeywords_tweaking).to_csv('honeywords_tweaking_200_strong.csv', index = False)

stn= pd.read_csv('honeywords_tweaking_200_strong.csv')
stn

def cal_scores(arr):
  scores = [[0 for x in range(NUM_SWEETWORDS)] for y in range(NUM_USER)]
  for i in range(NUM_USER):
    for j in range(NUM_SWEETWORDS):
      honeyword = arr[i][j]
      score = cal_similarity(arr[i][0], arr[i][j])
      scores[i][j] = score
  return scores

scores = cal_scores(honeywords_tweaking)

def avg_scores(scores):
  avg_scores = [0 for x in range(NUM_USER)]
  for i in range(NUM_USER):
    avg_score = sum(scores[i][1:])/len(scores[i][1:])
    avg_scores[i] = avg_score
  return avg_scores

avg_scores = avg_scores(scores)

np.savetxt("weak_tweaking_avg_scores_200.csv", avg_scores, delimiter =", ", fmt ='% s')

print("The average similarity score of honeywords tweaking is:", sum(avg_scores)/len(avg_scores))
