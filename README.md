# prod-gpspn
## Usage:
* We assume that you have torch and gpytorch installed in your python environment. The following example illustrates how to start to implement this model from stracth.
* First download some real data（x, y for training; x1,y1 for testing):
~~~
x= pd.read_csv(file path)
x1 = pd.read_csv(file path)
y = pd.read_csv(file path)
y1= pd.read_csv(file path)'
~~~

* Then we run prod_spngp.py to train the model and see the evaluation results. Results are printed:
~~~
print(f"SPN-GP  RMSE1: {rmse1}, RMSE2: {rmse2}")
print(f"SPN-GP  MAE1: {mae1}, MAE2: {mae2}")
print(f"SPN-GP  NLPD: {nlpd2}")
~~~

* Note that if you have n-dimensional y, please set the y_d value (line 13) in prod_inference.py to n. If you want n children for every split node, please set ddd value(line 8) in prod_structure.py to n-1. 
You can also set min_idx as a threshold for gp leave size. 

* If you have saved your model after successful training using:
~~~
filename = 'model.dill'
dill.dump(root, open(filename, 'wb'))
~~~
You can then evaluate it with your testing data in evaluation.py.
