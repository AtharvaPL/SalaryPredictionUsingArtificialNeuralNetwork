#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


path="Salary Data.csv"
df = pd.read_csv(path)
df


# In[5]:


df_age_group=df.groupby("Age").count().iloc[:,1:2].rename(columns={"Education Level":"Count"})
df_age_group
plt.hist(df_age_group,edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Count")


# In[6]:


df_gender_group=df.groupby("Gender").count().iloc[:,:1].rename(columns={"Age":"Count"})
plt.pie(df_gender_group.values.reshape(2,),labels=["Female","Male"],autopct='%1.1f%%')


# In[7]:


df_edu_group=df.groupby("Education Level").count().iloc[:,:1].rename(columns={"Age":"Count"})
plt.pie(df_edu_group.values.reshape(3,),labels=["Bachelor's","Master's","PhD"],autopct='%1.1f%%')


# In[8]:


df['Gender'] = df['Gender'].replace("Male",int(1))
df['Gender'] = df['Gender'].replace("Female",int(2))
df['Education Level'] = df['Education Level'].replace("Bachelor's",int(1))
df['Education Level'] = df['Education Level'].replace("Master's",int(2))
df['Education Level'] = df['Education Level'].replace("PhD",int(3))
df['Job Title'] = pd.factorize(df['Job Title'])[0]+1
df


# In[9]:


data = df.iloc[:, :-1]
X = data.values.astype(int)
result = df.iloc[:, -1:]
Y = result.values.astype(int)

test_size = 0.3
split_index = int(len(X) * (1 - test_size))

X_train = X[:split_index]
X_test = X[split_index:]
Y_train = Y[:split_index]
Y_test = Y[split_index:]


# In[10]:


class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        np.random.seed(0)
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

    def clip_gradients(self, threshold):
        self.dweights = np.clip(self.dweights, -threshold, threshold)
        self.dbiases = np.clip(self.dbiases, -threshold, threshold)


class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Loss_MeanSquaredError:
    def forward(self, y_pred, y_true):
        y_true_normalized = (y_true - y_true.min()) / (y_true.max() - y_true.min())      
        y_pred_normalized = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
        sample_losses = np.mean((y_true_normalized - y_pred_normalized) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

layer1 = Layer_Dense(5, 100)
activation1 = Activation_Linear()

layer2 = Layer_Dense(100, 50)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(50, 1)
activation3 = Activation_Linear()

loss_function = Loss_MeanSquaredError()

learning_rate = 0.01  
gradient_clip_threshold = 1
epochs = 10000


batch_size=30

for epoch in range(epochs):
    data = list(range(len(X_train)))
    np.random.shuffle(data)
    for batch_start in range(0, len(data), batch_size):
        batch_end = batch_start + batch_size
        batch_indices = data[batch_start:batch_end]

        X_batch = X_train[batch_indices]
        Y_batch = Y_train[batch_indices]

        layer1.forward(X_batch)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        activation3.forward(layer3.output)

        loss = loss_function.forward(activation3.output, Y_batch)

        loss_function.backward(activation3.output, Y_batch)
        activation3.backward(loss_function.dinputs)
        layer3.backward(activation3.dinputs)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        layer1.clip_gradients(gradient_clip_threshold)
        layer2.clip_gradients(gradient_clip_threshold)
        layer3.clip_gradients(gradient_clip_threshold)

        layer1.weights += -learning_rate * layer1.dweights
        layer1.biases += -learning_rate * layer1.dbiases
        layer2.weights += -learning_rate * layer2.dweights
        layer2.biases += -learning_rate * layer2.dbiases
        layer3.weights += -learning_rate * layer3.dweights
        layer3.biases += -learning_rate * layer3.dbiases

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {np.mean(loss)}')


# In[11]:


layer1.forward(X_test)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3.forward(activation2.output)
activation3.forward(layer3.output)

predictions = activation3.output

sum_error = 0

for i in range(len(predictions)):
    sum_error = sum_error + ((abs(predictions[i]-Y_test[i])/Y_test[i]))
    
accuracy= sum_error/len(predictions)
accuracy = 1-accuracy
print(f"Accuracy:{accuracy*100}%")

