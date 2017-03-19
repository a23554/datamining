from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data.odm')
# Split the data into training and test sets (25% held out for testing)
(trainingData, testData) = data.randomSplit([0.75, 0.25])

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)

# predict test
predictions = model.predict(testData.map(lambda x: x.features))
#map true value and predict value
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)

testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
acc = labelsAndPredictions.filter(lambda (v, p): v == p).count() / float(testData.count())
#calculate TP FP FN TN
TP=labelsAndPredictions.filter(lambda (v, p): v ==1 and p == 1).count()
FP=labelsAndPredictions.filter(lambda (v, p): v ==0 and p == 1).count()
FN=labelsAndPredictions.filter(lambda (v, p): v ==1 and p == 0).count()
TN=labelsAndPredictions.filter(lambda (v, p): v ==0 and p == 0).count()
print(str(TP)+","+str(FP)+","+str(FN)+","+str(TN)+",")

precision=float(TP)/(TP+FP)
recall=float(TP)/(TP+FN)
print('Precision = ' + str(precision))
print('Recall= ' + str(recall))
print('Test Error = ' + str(testErr))
print('Accuracy = ' + str(acc))
print('Learned classification tree model:')
print(model.toDebugString())
