import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("train.csv")

# Show first rows
print(data.head())

# Check missing values
print(data.isnull().sum())

# Fill missing Age values
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Convert gender to numbers
data['Sex'] = data['Sex'].map({'male':0,'female':1})

# Select features
X = data[['Pclass','Sex','Age','Fare','SibSp','Parch']]
y = data['Survived']

# Split dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train,y_train)

# Predict
predictions = model.predict(X_test)

# Check accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, predictions))

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,predictions)

sns.heatmap(cm, annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.show

sns.countplot(x='Survived', data=data)
plt.show()

sns.countplot(x='Sex', hue='Survived', data=data)
plt.show()

sns.histplot(data['Age'])
plt.show()

#Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
print(data.columns)