#Evaluation metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def metrics_calculation(y_actual, y_predicted,plot_roc_curve=True):
    #Classification report
    class_report=metrics.classification_report(y_actual,y_predicted)
    print("Classification report:\n",class_report)
    
    #Confusion Matrix
    conf_mat = metrics.confusion_matrix(y_actual,y_predicted,labels=None)
    print("Confusion Matrix:\n",conf_mat)
    
    #Confusion Matrix Display
    conf_display=metrics.ConfusionMatrixDisplay(conf_mat)
    conf_display.plot()
    plt.show()
    
    #Accuracy
    accuracy =metrics.accuracy_score(y_actual, y_predicted)
    print("Accuracy:",accuracy)
    
    #precision
    precision=metrics.precision_score(y_actual, y_predicted)
    print("Precision:",precision)
    
    #Recall
    recall=metrics.recall_score(y_actual, y_predicted)
    print("Recall:",recall)

    #sensitivity
    sensitivity = conf_mat[1, 1] / (conf_mat[1,1] + conf_mat[1,0])
    print("Sensitivity:",sensitivity)
    
    #specificity
    specificity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    print("Specificity:",specificity)
    
    #f1-score
    f1_score = metrics.f1_score(y_actual,y_predicted)
    print("F1-Score:",f1_score)
    
    #Cohen's Kappa score
    Kappa_score= metrics.cohen_kappa_score(y_actual,y_predicted)
    print("Cohen Kappa Score:",Kappa_score)
    
    #G-measure
    g_measure=2*(precision*recall)/(precision+recall)
    print("G_measure:",g_measure)
    
    #Informedness/Youdens statistic
    informedness=sensitivity+specificity-1
    print("Informedness/Youdens statistic:",informedness)
    
    #Positive predictive Value
    ppv= conf_mat[1,1] / (conf_mat[1,1] + conf_mat[0,1])
    print("PPV:",ppv)
    
     #Negative predictive Value
    npv= conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1,0])
    print("NPV:",npv)
    
    #Markedness
    marked=ppv+npv-1
    print("Markedness:",marked)
    
    # Matthews Correlation Coefficient
    mcc=((conf_mat[0, 0] * conf_mat[1, 1]) - (conf_mat[0, 1] * conf_mat[1, 0]))/np.sqrt((conf_mat[0, 0] + conf_mat[0, 1])
        * (conf_mat[0, 0] + conf_mat[1, 0]) * (conf_mat[1, 1] + conf_mat[1, 0]) * (conf_mat[1, 1] + conf_mat[0, 1]))
    print("Matthews Correlation Coefficient",mcc)
    
    #ROC-AUC Score
    roc_auc=metrics.roc_auc_score(y_actual,y_predicted)
    print("ROC-AUC Score:",roc_auc)
    
    #ROC_Curve
    fpr,tpr,thresholds =metrics.roc_curve(y_actual,y_predicted)
    if plot_roc_curve:
        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
