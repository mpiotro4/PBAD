#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# ## Preprocess data

# In[3]:


npis = [
    "C1M",
    "C2M",
    "C3M",
    "C4M",
    "C5M",
    "C6M",
    "C7M",
    "C8M",
    "H1"
    ]
npi_labels = [
    "School closing",
    "Workplace closing",
    "Cancel public events",
    "Restrictions on gatherings",
    "Close public transport",
    "Stay at home requirements",
    "Restrictions on internal movement",
    "International travel controls",
    "Public information campaigns"
]

df = pd.read_csv("./OxCGRT_clean.csv", sep=',')

si = df["StringencyIndex_Average"].values
si = si/100
si = si[:16000]

prev_day = df["C1M"].values[1:]
prev_day = prev_day[:16000]
samples = np.column_stack((si,prev_day))
labels = [df["C1M"].values[:16000]]


for npi in npis[1:]:
  prev_day = df[npi].values[1:]
  prev_day = prev_day[:16000]
  samples = np.column_stack((samples,prev_day))
  labels.append(df[npi].values[:16000])

not_shuffled_samples = samples[15000:16000]
not_shuffled_labels = []
for label in labels:
  not_shuffled_labels.append(label[15000:16000])

samples, *labels = shuffle(samples, *labels, random_state=0)

train_labels = []
train_samples =[]
train_samples = samples[:15000]

for label in labels:
  train_labels.append(label[:15000])

test_labels = []
test_samples =[]
test_samples = samples[15000:15100]

for label in labels:
  test_labels.append(label[15000:15100])


# ## MTL model

# In[4]:


from keras.optimizers import Adam
from keras import Input, Model
from keras.layers import Dense

num_tasks = 9
num_features = num_tasks + 1

shared_layer_1 = Dense(32, input_dim=num_features, activation='relu')
shared_layer_2 = Dense(32, activation='relu')

task_1_output_layer = Dense(5, activation='softmax', name='C1')
task_2_output_layer = Dense(5, activation='softmax', name='C2')
task_3_output_layer = Dense(5, activation='softmax', name='C3')
task_4_output_layer = Dense(5, activation='softmax', name='C4')
task_5_output_layer = Dense(5, activation='softmax', name='C5')
task_6_output_layer = Dense(5, activation='softmax', name='C6')
task_7_output_layer = Dense(5, activation='softmax', name='C7')
task_8_output_layer = Dense(5, activation='softmax', name='C8')
task_9_output_layer = Dense(5, activation='softmax', name='H1')

input_tensor = Input(shape=(num_features,))

shared_tensor = shared_layer_1(input_tensor)
shared_tensor = shared_layer_2(shared_tensor)

task_1_output = task_1_output_layer(shared_tensor)
task_2_output = task_2_output_layer(shared_tensor)
task_3_output = task_3_output_layer(shared_tensor)
task_4_output = task_4_output_layer(shared_tensor)
task_5_output = task_5_output_layer(shared_tensor)
task_6_output = task_6_output_layer(shared_tensor)
task_7_output = task_7_output_layer(shared_tensor)
task_8_output = task_8_output_layer(shared_tensor)
task_9_output = task_9_output_layer(shared_tensor)

mtl_model = Model(inputs=input_tensor, outputs=[
    task_1_output, 
    task_2_output,
    task_3_output, 
    task_4_output,
    task_5_output, 
    task_6_output,
    task_7_output, 
    task_8_output,
    task_9_output],)

mtl_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={
            'C1': 'sparse_categorical_crossentropy',
            'C2': 'sparse_categorical_crossentropy',
            'C3': 'sparse_categorical_crossentropy',
            'C4': 'sparse_categorical_crossentropy',
            'C5': 'sparse_categorical_crossentropy',
            'C6': 'sparse_categorical_crossentropy',
            'C7': 'sparse_categorical_crossentropy',
            'C8': 'sparse_categorical_crossentropy',
            'H1': 'sparse_categorical_crossentropy'
            },
        metrics={
            'C1': 'accuracy',
            'C2': 'accuracy',
            'C3': 'accuracy',
            'C4': 'accuracy',
            'C5': 'accuracy',
            'C6': 'accuracy',
            'C7': 'accuracy',
            'C8': 'accuracy',
            'H1': 'accuracy'
            }
        )
mtl_model.summary()

mtl_model.fit(
    x=train_samples.reshape(-1,num_features),
    y=train_labels,
    validation_split=0.1,
    batch_size=10,
    epochs=10,
    shuffle=True,
    verbose=2
)


# # Predict

# In[5]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ## test data

# In[29]:


from sklearn.metrics import accuracy_score

predictions = mtl_model.predict(
    x=test_samples.reshape(-1,num_features),
    batch_size = 10,
    verbose = 0)

predictions = np.array(predictions)

f = lambda x: np.argmax(x, axis=-1)
predictions = f(predictions)

for idx, pred in enumerate(predictions):
  cm = confusion_matrix(y_true=test_labels[idx], y_pred=pred)
  accuracy = accuracy_score(test_labels[idx], pred).round(2)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot()
  disp.ax_.set_title(npi_labels[idx] + ", accuracy: %.2f"%accuracy)
  plt.show()


# ## Recursive predictions

# In[108]:


predicted_sample = not_shuffled_samples[99]

iterative_predictions = []
next_samples = []

for i in range(100,200):
    predictions = mtl_model.predict(
        x=predicted_sample.reshape(-1,num_features),
        batch_size = 10,
        verbose = 0)

    predictions = np.array(predictions)

    f = lambda x: np.argmax(x, axis=-1)
    predictions = f(predictions)
    predicted_sample = np.concatenate(([not_shuffled_samples[i][0]], predictions.flatten()))
    next_samples.append(predicted_sample)
    iterative_predictions.append(predictions.flatten())

print(type(np.array(iterative_predictions)))
print(type(not_shuffled_samples))

zipped_predicted_npis = list(zip(*iterative_predictions))


# ## Barcharts

# In[114]:


# Define the arrays
x = np.arange(1, 101)


for i in range(0,9):
    predicted_y = np.array(zipped_predicted_npis[i][0:100])
    label_y = not_shuffled_labels[i][0:100]

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, predicted_y, width=0.4, label='Prediction')
    ax.bar(x + 0.2, label_y, width=0.4, label='Dataset')

    # Add labels and legend
    ax.set_xlabel('Day')
    ax.set_ylabel('NPI strength')
    ax.set_title(npi_labels[i])
    ax.legend()

    # Display the chart
    plt.show()


# In[ ]:




