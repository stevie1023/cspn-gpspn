from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pylab as plt
all_rmse= np.array([0.06842234,0.06733447,0.11752782,0.08016811,0.01442562,0.01491468, 0.02111368, 0.04220094, 0.07476529, 0.07677453,0.02476506 ,0.01389984, 0.00675314 ,0.03352504 ,0.01010764,
0.02167416, 0.04091164, 0.09267932, 0.0114802 , 0.06972976 ,0.07730725, 0.05254911, 0.08202379, 0.06134666, 0.04754575, 0.04203333, 0.12881044, 0.13388563 ,0.0474235,  0.05586054])

y_test = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
fpr, tpr, thresholds  =  roc_curve(y_test, scores)
roc_auc = auc(fpr,tpr)
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
