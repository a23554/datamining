import jieba
import xlrd
import jieba.analyse
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.cluster import KMeans
orgin_data=xlrd.open_workbook('dataset.xlsx')
text_table = orgin_data.sheets()[0]
tags_value=text_table.col_values(1)
all_tags=[]
del tags_value[0]
for tag in tags_value:
    
    tags=jieba.analyse.extract_tags(tag, topK=20)
    comb_tags=" ".join(tags)
    print(comb_tags)
    all_tags.append(comb_tags)
    
    #print(",".join(tags))
#TFIDF
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=vectorizer.fit_transform(all_tags)
words = vectorizer.get_feature_names()
weight = tfidf.toarray()
print(weight)
true_k = 4
model = KMeans(n_clusters=true_k)
print(weight)
print(model.fit(weight))
print(model.labels_)
print(model.cluster_centers_)
print(model.inertia_)
indexlist = []
for i in range(100):
    indexlist.append(i)
sortres = sorted(zip(model.labels_,indexlist), key = lambda x : x[0])
f = open('newskmean.txt','w',encoding='UTF-8')
for i in range(100):
    f.writelines("cluster {}  ,index:{}, content:{} ".format(sortres[i][0],sortres[i][1],tags_value[sortres[i][1]]))
    f.writelines("\n")
f.close