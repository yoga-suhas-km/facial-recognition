import cv2
import tensorflow as tf
import os
from process_data import get_test_dataset
from keras.models import load_model
import config
import numpy as np
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#CATEGORIES = ["DR", "DT", "KT", "MS", "SRK", "SU", "TM", "WS"]



def predict(x,y) :

    path, dirs, file = next(os.walk(config.models))
    print(file)
    for i in file:
        print(i)
        model = tf.keras.models.load_model(i)
        #print(y)
        print(model.summary())
        
        score = model.evaluate(x, y, verbose=0)
        print(model.metrics_names)
        print(score)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        
        y_pred_t = model.predict(x, batch_size=64, verbose=0)
        y_pred_bool = np.argmax(y_pred_t, axis=1)
        y_pred = model.predict_classes(x)

        print(y_pred)
        #print(y_pred_bool)
        """
        print(y_pred_t)


        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y, y_pred)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y, y_pred, average='weighted',labels=np.unique(y_pred))
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y, y_pred,average='weighted')
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y, y_pred, average='weighted',labels=np.unique(y_pred))
        print('F1 score: %f' % f1)
    
        print(classification_report(y, y_pred_bool,labels=np.unique(y_pred_bool)))
"""


def main():
    x,y = get_test_dataset(1)
    
    x = x/255.0
    print(y)
    y = np_utils.to_categorical(y, config.num_classes)

    predict(x,y)

if __name__ == "__main__":
    # execute only if run as a script
    main()