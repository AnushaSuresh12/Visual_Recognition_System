import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import compute_pca
import make_dataset

#
def divide_test():
    CATEGORIES=[]
    folder=make_dataset.folder
    for i in os.listdir(folder):
        CATEGORIES.append(i)
    d_train = {}
    d_test = {}
    dict_list = {}
    for category in CATEGORIES:
        temp_list = []
        result=np.zeros((1604,384))
        category_features = pickle.load(open("class/optflow_%s.p" % category, "rb"))
        for i in category_features:
            arr=np.array(i["features"])
            result[:arr.shape[0],:arr.shape[1]]=arr
            temp=result.ravel()
            temp_list.append(temp.T)
        dict_list[category]=temp_list

    for i in dict_list:
        d_train[i],d_test[i]=train_test_split(dict_list[i],train_size=0.8,test_size=0.2)
    mean={}
    dmean={}
    for i in d_train:
            v = np.matrix(np.array(d_train[i])).T
            mean[i]=np.matrix.mean(v,axis=1)
            dmean[i]=v-np.matrix.mean(v,axis=1)
    return dmean,d_test,mean

# computes PCA for k principal components
def calculate_principal_component(dmean,k):
    d_final={}
    for l in range(1,k+1):
        for i in dmean:
            v=dmean[i]
            Q=compute_pca.calculate_pca(v, 10)
            if (i not in d_final):
                lst = []
                Q = np.matrix(Q).T
                lst.append(Q)
                d_final[i] = lst
            else:
                lst = d_final[i]
                Q = np.matrix(Q).T
                lst.append(Q)
                d_final[i] = lst
        for i in dmean:
            v = dmean[i]
            lst = d_final[i]
            Q = lst[len(lst) - 1]
            v = np.subtract(v, np.matmul(Q, np.matmul(Q.T, v)))
            dmean[i] = np.matrix(v)
        return d_final