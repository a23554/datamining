import numpy as np
import pandas as pd

from splearn.rdd import ArrayRDD
from splearn.rdd import DictRDD

from splearn.feature_extraction.text import SparkCountVectorizer
from splearn.feature_extraction.text import SparkHashingVectorizer
from splearn.feature_extraction.text import SparkTfidfTransformer
from splearn.naive_bayes import SparkMultinomialNB
from splearn.naive_bayes import SparkGaussianNB
from splearn.svm import SparkLinearSVC
from splearn.linear_model import SparkSGDClassifier
from splearn.linear_model import SparkLogisticRegression
from splearn.pipeline import SparkPipeline
from splearn.grid_search import SparkGridSearchCV

#data preprocess
df = pd.read_csv("review.csv",header=None,encoding='latin1')
df[0] =df[0].apply(lambda death: 0 if death <= 5 else 1)
df=df.dropna()
data =df[1]
target = df[0]
list=[]
data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.25,random_state=43)   

# train data toRDD
train_x = sc.parallelize(data_train)
train_y = sc.parallelize(target_train)
train_x = ArrayRDD(train_x)
train_y = ArrayRDD(train_y)
Z = DictRDD((train_x, train_y), columns=('X', 'y'), dtype=[np.ndarray, np.ndarray])

# pipeline
dist_pipeline = SparkPipeline((
     ('vect', SparkHashingVectorizer(non_negative=True)),  # hashingTF for NB
     ('tfidf', SparkTfidfTransformer()),                         # IDF
     ('clf', SparkMultinomialNB(alpha=0.05))                     # NB
))
  
# fit
dist_pipeline.fit(Z, clf__classes=np.array([0, 1]))

# test data to RDD
test_x = ArrayRDD(sc.parallelize(data_test))
test_y = ArrayRDD(sc.parallelize(target_test))
test_Z = DictRDD((test_x, test_y), columns=('X', 'y'), dtype=[np.ndarray, np.ndarray])  
  
# predict test data
predicts = dist_pipeline.predict(test_Z[:, 'X'])

# metrics(accuracy, precision, recall, f1)
data_size = len(test)
array_y = traget_test
array_pred = predicts.toarray()
y_and_pred = zip(array_y, array_pred)
  
#calculate accuracy
pos_size = sum(array_y)
neg_size = data_size - pos_size

pos_pred_size = sum(array_pred)
neg_pred_size = data_size - pos_pred_size

pos_acc_size = len(filter(lambda x: x[0] == 1 and x[0] == x[1], y_and_pred))
neg_acc_size = len(filter(lambda x: x[0] == 0 and x[0] == x[1], y_and_pred))
acc_size = pos_acc_size + neg_acc_size

accuracy = acc_size / float(data_size)
precision = pos_acc_size / float(pos_pred_size)
recall = pos_acc_size / float(pos_size)
f1 = 2 * (precision * recall) / (precision + recall)
  
score = {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}
print score

