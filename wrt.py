# Name: Anirbaan Ghatak
# Roll no: C026
# Aim: :  Implementation of ID3(Decision Tree) Classifier. Also to find the performance metrics for the given dataset.

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data = pd.read_csv('weather_data.csv', index_col='Day')

# Convert the list of dictionaries to a DataFrame
headers = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Decision']

df = pd.DataFrame(data, columns=headers)

# Encode label categories to numbers, This is necessary because scikit-learn's decision tree implementation works with numerical data

label_encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target variable from the DataFrame

X = df.drop('Decision', axis=1)  # Features (exclude the 'Decision' column)
y = df['Decision']  # Target variable ('Decision')

# Initialize the DecisionTreeClassifier with criterion as 'entropy' to simulate ID3
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Fit the model
clf.fit(X, y)

# Display the decision tree
tree.plot_tree(
    clf, feature_names=headers[:-1], class_names=['No', 'Yes'], filled=True)

y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
class_report = classification_report(y, y_pred, target_names=['No', 'Yes'])
conf_matrix = confusion_matrix(y, y_pred)

print('Accuracy:', accuracy)
print('Classification Report:\n', class_report)
print('Confusion Matrix:\n', conf_matrix)
