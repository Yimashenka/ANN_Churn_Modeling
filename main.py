#Artificial Neural Network

## Step 1 : Importing the libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

## Step 2 :  Data Preprocessing
### Importing the dataset
dataset = pd.read_csv('data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

### Encoding categorical data
#### Encoding the "Gender" column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#### One Hot Encoding the "Geography" column
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))

### Splitting the dataset into the training set and the test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

### Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Step 2 : Building the ANN
### Initilizing the ANN
ann = tf.keras.models.Sequential()

### Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(
    units=6,
    activation='relu',
))

### Adding the second hidden layer
ann.add(tf.keras.layers.Dense(
    units=6,
    activation='relu',
))

### Adding the output layer
ann.add(tf.keras.layers.Dense(
    units=1,
    activation='sigmoid',
))

## Step 3 : Training the ANN
### Compiling the ANN
ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

### Training the ANN
ann.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100
)

## Step 4 : Making the prediction and evaluating the model
### Predicting the test set result
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

### Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
