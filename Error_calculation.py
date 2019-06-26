import numpy as np
from numpy import linalg as LA
import root


# this function computes the error for each PCA
def calculate_error(principal_component_i):
    test_label = ''
    true_label = ''
    accurate=[]
    for i in range(1,11):
        count = 0
        dmean,d_test,mean = root.divide_test()
        d_final=root.calculate_principal_component(dmean, principal_component_i)
        accurate_value = 0
        for test_image_i in d_test:
            true_label = test_image_i
            dfx = np.matrix(np.array(d_test[test_image_i])).T
            for k in range(0, dfx.shape[1]):
                min_error = 999999999999
                df = dfx[:, k]
                count += 1
                for class_c in dmean.keys():
                    Q = np.matrix(np.array(d_final[class_c])).T
                    dfb = np.subtract(df, mean[class_c])
                    # Reconstruction error Calculation
                    error = LA.norm(np.subtract(dfb, np.dot(Q, np.dot(Q.T, dfb))))
                    if error < min_error:
                        min_error = error
                        test_label = class_c
                if true_label == test_label:
                    accurate_value += 1
        error_value = round((((count - accurate_value) / count) * 100), 2)
        accurate.append(error_value)
    return accurate