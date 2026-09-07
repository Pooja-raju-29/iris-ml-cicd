### what is iris dataset:The Iris dataset contains measurements of flowers.
### input is x and it is sepal length and width , petal length and width
### out target that is y = 0 setosa , 1 versicolor,2 virginica

import pandas as pd
import joblib #### this is used to save the train data and can be reused when required and no need to train the model again
#### it will be saved in .pkl
import os

from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier   ###why randomforest
from sklearn.metrics import accuracy_score

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
print(X.head())
print(y[:5])
# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# print("Model saved at:", os.path.abspath("model.pkl"))
current_folder = os.path.dirname(os.path.abspath(__file__))

# Create the model path
model_path = os.path.join(current_folder, "model.pkl")

# Save model
joblib.dump(model, model_path)

print("Model saved successfully!")
print("Model location:", model_path)