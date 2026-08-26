# Cleaned from the completed Break Through Tech decision-tree notebook.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename)

# Remove one problematic field and handle missing service-area values.
df = df.drop(columns='Married')
df['ServiceArea'] = df['ServiceArea'].fillna('unavailable')

# Keep explicit indicators for the ten most common service areas.
top_10_service_areas = df['ServiceArea'].value_counts().head(10).index.tolist()
for service_area in top_10_service_areas:
    df[f'ServiceArea_{service_area}'] = np.where(df['ServiceArea'] == service_area, 1, 0)

df = df.drop(columns='ServiceArea')

# One-hot encode the remaining categorical variables.
to_encode = list(df.select_dtypes(include=['object']).columns)
for col in to_encode:
    df = df.join(pd.get_dummies(df[col], prefix=col))
df = df.drop(columns=to_encode)

y = df['Churn']
X = df.drop(columns='Churn')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=123
)

def train_test_decision_tree(depth: int, leaf: int = 1, criterion: str = 'entropy') -> float:
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=depth,
        min_samples_leaf=leaf,
        random_state=123,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

for depth in [8, 32]:
    print(f"Max Depth: {depth}, Accuracy: {train_test_decision_tree(depth):.4f}")
