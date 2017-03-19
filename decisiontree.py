import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score,precision_score,recall_score
import pydotplus 
from IPython.display import Image 
import pydotplus 
df = pd.read_csv('cdeath.csv')
df=df.fillna(0)
del df['Book of Death']
del df['Death Chapter']
del df['Name']
df.loc[df['Death Year'] != 0,'Death Year'] = 1 #將Dearh轉換成0跟1
#df["Death Year"] =df["Death Year"].apply(lambda death: 0 if death == 0 else 1)
df_Allegiances=df['Allegiances'].str.get_dummies() #家族轉成dummy的特徵
df = pd.concat([df, df_Allegiances], axis=1)
del df['Allegiances']

features = list(df.ix[:, df.columns != 'Death Year']) #列出除了death以外的特徵
x=df[features]
y=df['Death Year']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25) #將資料切成train跟test data

clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(x_train, y_train)  #使用fit產出decision tree

with open("death.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f) 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("death.pdf")  #產出PDF
#scores = cross_validation.cross_val_score(clf, X, Y, cv=10)
score=clf.score(x_test,y_test, sample_weight=None)
print("Accuracy is: ",score) 

pre=clf.predict(x_test)
'''ascore=accuracy_score(y_test,pre)
print(ascore)'''
precision=precision_score(y_test,pre)
print("Precision rate is: ",precision)
recall=recall_score(y_test,pre)
print("Recall rate is: ",recall)

'''
test=y_test.as_matrix()
print(test)
arr=[0,0,0,0,0]
for i in range(len(y_test)):
    if test[i] == pre[i]:
        if test[i]==0:
            arr[1]+=1
        else:
            arr[2]+=1
    else:
        if test[i]==0:
            arr[3]+=1
        else:
            arr[4]+=1
print(arr)
'''            