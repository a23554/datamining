from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from nltk import tokenize
from sklearn.datasets import dump_svmlight_file

#資料前處理 
df = pd.read_csv("review.csv",header=None,encoding='latin1')
df=df.dropna()
sentences = df[1]
df[0] =df[0].apply(lambda death: 0 if death <= 5 else 1)

#情緒分析
sid = SentimentIntensityAnalyzer()

#情緒分析產生出compound,neg,neu,pos四項分數
#features=['compound','neg','neu','pos']
for sentence in sentences:
    ss = sid.polarity_scores(sentence)
    i=0
    for k in sorted(ss):
        df.loc[df[1]==sentence,i+2]=str(ss[k]) #將後四項資料加進dataframe
        i=i+1
print(df)

x=df.loc[:,2:6]
print(x)
y=df[0]
dump_svmlight_file(x,y,"data.odm",zero_based=False, multilabel=False)
