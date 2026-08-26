# Generated from the completed Break Through Tech notebook.

# Capstone: Define and Solve an ML Problem

import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
import time

# Load the course-provided Airbnb data set.
airbnb_filename = os.path.join(os.getcwd(), "data_capstone", "airbnbListingsData.csv")
df = pd.read_csv(airbnb_filename)

# Explore class imbalance.
print(df['price_category'].value_counts())
print(df['price_category'].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='price_category')
plt.title('Distribution of Price Category')
plt.xlabel('Price Category')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='room_type', hue='price_category')
plt.title('Price Category by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Price Category')
plt.tight_layout()
plt.show()

# Convert the label to binary form.
df['price_category'] = df['price_category'].map({'low': 0, 'high': 1})

# Remove free-text, high-cardinality, and target-leakage columns.
columns_to_drop = [
    'name',
    'description',
    'neighborhood_overview',
    'host_name',
    'host_location',
    'host_about',
    'amenities',
    'price'
]
df = df.drop(columns=columns_to_drop)

# Impute missing numerical values with medians.
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Impute missing categorical values with modes.
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encode the remaining categorical columns.
df = pd.get_dummies(df, drop_first=True, dtype=float)

X = df.drop(columns='price_category')
y = df['price_category']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1234,
    stratify=y
)

print("Final feature shape:", X.shape)

# Logistic Regression baseline.
lr_scaler = StandardScaler()
X_train_lr_scaled = lr_scaler.fit_transform(X_train)
X_test_lr_scaled = lr_scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_lr_scaled, y_train)
predictions = model.predict(X_test_lr_scaled)

accuracy_initial = accuracy_score(y_test, predictions)
f1_initial = f1_score(y_test, predictions, average='binary')
print("Initial Accuracy:", accuracy_initial)
print("Initial F1 Score:", f1_initial)

# Tune regularization using 5-fold cross-validation.
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    cv=5,
    scoring='f1'
)
grid_search.fit(X_train_lr_scaled, y_train)

best_c = grid_search.best_params_['C']
print("Best C:", best_c)
print("Best cross-validation F1 score:", grid_search.best_score_)

final_model = LogisticRegression(C=best_c, max_iter=1000)
final_model.fit(X_train_lr_scaled, y_train)
final_predictions = final_model.predict(X_test_lr_scaled)

accuracy_final = accuracy_score(y_test, final_predictions)
f1_final = f1_score(y_test, final_predictions, average='binary')
print("Final Accuracy:", accuracy_final)
print("Final F1 Score:", f1_final)

# Inspect influential Logistic Regression coefficients.
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': final_model.coef_[0]
})
coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)
top_features = coef_df.head(10)

plt.figure(figsize=(8, 6))
sns.barplot(data=top_features, x='Coefficient', y='Feature')
plt.title("Top 10 Most Influential Features")
plt.tight_layout()
plt.show()

# Scale the same train/test data for the neural network.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a two-hidden-layer feed-forward neural network.
n_features = X_train_scaled.shape[1]
nn_model = keras.Sequential()
nn_model.add(keras.layers.InputLayer(input_shape=(n_features,)))
nn_model.add(keras.layers.Dense(units=64, activation='relu'))
nn_model.add(keras.layers.Dense(units=32, activation='relu'))
nn_model.add(keras.layers.Dense(units=1, activation='sigmoid'))
nn_model.summary()

sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
nn_model.compile(optimizer=sgd_optimizer, loss=loss_fn, metrics=['accuracy'])

class ProgBarLoggerNEpochs(keras.callbacks.Callback):
    def __init__(self, num_epochs: int, every_n: int = 50):
        self.num_epochs = num_epochs
        self.every_n = every_n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            s = 'Epoch [{}/ {}]'.format(epoch + 1, self.num_epochs)
            logs_s = ['{}: {:.4f}'.format(k.capitalize(), v) for k, v in logs.items()]
            print(', '.join([s] + logs_s))

num_epochs = 100
t0 = time.time()
history = nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=num_epochs,
    validation_split=0.2,
    callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=10)],
    verbose=0
)
t1 = time.time()
print('Elapsed time: %.2fs' % (t1 - t0))

# Visualize training and validation performance.
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Evaluate the neural network on the test set.
nn_probabilities = nn_model.predict(X_test_scaled)
nn_predictions = (nn_probabilities >= 0.5).astype(int).flatten()
nn_accuracy = accuracy_score(y_test, nn_predictions)
nn_f1 = f1_score(y_test, nn_predictions, average='binary')
print("Neural Network Accuracy:", nn_accuracy)
print("Neural Network F1 Score:", nn_f1)

# Side-by-side comparison.
results = pd.DataFrame({
    'Metric': ['Accuracy', 'F1 Score'],
    'Logistic Regression': [accuracy_final, f1_final],
    'Neural Network': [nn_accuracy, nn_f1]
})
print(results.to_string(index=False))
