import tensorflow as tf
import pandas as pd
import numpy as np
import csv

batchSize=32
bufferSize=256

data=pd.read_csv("algebra.csv",header=None)

x1NormalizationFactor=np.max(data[0])-np.min(data[0])
x2NormalizationFactor=np.max(data[1])-np.min(data[1])
y1NormalizationFactor=np.max(data[2])-np.min(data[2])

def inputFunction(filepath,shuffle=False,repeat=1):
    def decode(row):

        featureNames=["x1","x2"]

        defaultRecord=[[0.],[0.],[0.],[0.]]
        #converts csv records to tensors
        parsedRow=tf.decode_csv(row,defaultRecord)
        
        #get label and features
        label=parsedRow[2]
        features=parsedRow[0:2]

        #normalize
        label/=y1NormalizationFactor
        features[0]/=x1NormalizationFactor
        features[1]/=x2NormalizationFactor

        data=[dict(zip(featureNames,features)),label]

        return data

    dataset=(tf.data.TextLineDataset(filepath).map(decode))

    if(shuffle):
        dataset=dataset.shuffle(buffer_size=bufferSize)

    dataset=dataset.repeat(repeat)
    dataset=dataset.batch(batchSize)

    iterator=dataset.make_one_shot_iterator()

    batchFeatures,batchLabels=iterator.get_next()

    return batchFeatures,batchLabels

#creating dummy feature columns
featureColumn=[tf.feature_column.numeric_column("x1"),tf.feature_column.numeric_column("x2")]

#make the model
estimator=tf.estimator.DNNRegressor(feature_columns=featureColumn,hidden_units=[2],model_dir = "./model1/")
#train
estimator.train(input_fn=lambda: inputFunction("train1.csv",True,8),steps=1500)
#evaluate
metrics=estimator.evaluate(input_fn=lambda : inputFunction("test2.csv",True,32))

#metrics
print("Loss is : ",metrics["loss"])
print("Average Loss is : ",metrics["average_loss"])