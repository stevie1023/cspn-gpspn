from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pylab as plt
import numpy as np


y0 = np.genfromtxt('/home/mzhu/madesi/mzhu_code/cspn22/cspn2-master/windmill1.csv', delimiter=',')
y1 = np.genfromtxt('/home/mzhu/madesi/mzhu_code/cspn22/cspn2-master/windmill2.csv',delimiter=',')
y_0 = np.genfromtxt('/home/mzhu/madesi/mzhu_code/cspn22/cspn2-master/windmill1_.csv',delimiter=',')
y_1 = np.genfromtxt('/home/mzhu/madesi/mzhu_code/cspn22/cspn2-master/windmill2_.csv',delimiter=',')
x1= np.linspace(0,5000,num=1000,dtype=int)

plt.figure(figsize=(6,6))
# plt.title('rmse ROC')
plt.plot(x1,y0, 'r-', label = 'y0')
plt.plot(x1,y1, 'b-', label = 'y1')
plt.plot(x1, y_0, 'r--', label = 'y_0')
plt.plot(x1, y_1, 'b--', label = 'y_1')
# plt.plot(fpr1, tpr1, 'b-', label = 'original SPNGP Val AUC = %0.3f' % roc_auc1)
# plt.plot(fpr2, tpr2, 'g-', label = 'improved GPSPN Val AUC = %0.3f' % roc_auc2)
# plt.plot(fpr1, tpr1,'ro-',fpr2, tpr2,'r^-',fpr3, tpr3)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
plt.show()
