#!/usr/bin/env python
# coding: utf-8

# In[15]:


from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# In[16]:


california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")


# In[17]:


california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe


# In[18]:


california_housing_dataframe.describe()


# In[19]:


my_feature = california_housing_dataframe[["total_rooms"]]

feature_columns = [tf.feature_column.numeric_column("total_rooms")]


# In[20]:


targets = california_housing_dataframe["median_house_value"]


# In[21]:


my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)


# In[22]:


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
        
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# In[23]:


_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)


# In[24]:


prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs = 1, shuffle = False)

predictions = linear_regressor.predict(input_fn = prediction_input_fn)

predictions = np.array([item['predictions'][0] for item in predictions])

mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)

print("MSE (training data): %0.3f" % mean_squared_error)
print("RMSE (training data): %0.3f" % root_mean_squared_error)


# In[26]:


min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. median house value: %0.3f" % min_house_value)
print("Max. median house value: %0.3f" % max_house_value)
print("Difference between min. and max.: %0.3f" % min_max_difference)
print("RMSE: %0.3f" % root_mean_squared_error)


# In[27]:


calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()


# In[28]:


sample = california_housing_dataframe.sample(n=300)


# In[31]:


x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

plt.plot([x_0, x_1], [y_0, y_1], c = 'r')

plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

plt.scatter(sample["total_rooms"], sample["median_house_value"])

plt.show()


# In[ ]:




