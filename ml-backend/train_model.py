"""

This script trains a machine learning model to predict insurance charges based on various features.

"""

import pandas as pd # handles the dataset
import numpy as np # works with arrays
import matplotlib.pyplot as plt
from IPython.display import display # display processed data
from sklearn.model_selection import train_test_split # splits the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler # normalizes input data
from sklearn.ensemble import RandomForestRegressor # ml model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # analyze model results
from sklearn.metrics import accuracy_score, classification_report # for performance evaluation
from sklearn.preprocessing import LabelEncoder # Convert strings to numbers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import joblib # saves the model and scaler for later use


file_path = r"ml-backend\insurance.csv" # path to the dataset
df = pd.read_csv(file_path)
# reads the csv file into a dataframe so that we can actually read and preprocess

# Permuation importance (meaure importance of features)
def permutation_importance_manual(model, X, y, metric=mean_absolute_error, n_repeats=5, random_state=42):
    np.random.seed(random_state)
    baseline = metric(y, model.predict(X))
    importances = []

    for i in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            shuffled = X_permuted[:, i].copy()
            np.random.shuffle(shuffled)
            X_permuted[:, i] = shuffled
            score = metric(y, model.predict(X_permuted))
            scores.append(score)
        mean_score = np.mean(scores)
        importances.append(mean_score - baseline)

    return np.array(importances)

# Build Keras neural network model
def build_keras_model():
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(1024, activation='relu'),
        #layers.Dropout(0.2),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_random_forest_model():
    # train a random forest model
    model = RandomForestRegressor(n_estimators = 150, random_state=42)
    model.fit(X_train, y_train)

    return model

# Initilializse data
# Change charges to integer values
df['charges'] = df['charges'].astype(int)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the relevant columns
cols_to_fix = ['sex', 'region', 'smoker']
df[cols_to_fix] = df[cols_to_fix].apply(label_encoder.fit_transform)

# Separate features and labels
X = df.drop('charges', axis=1)
y = df['charges']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize (scale) the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

feature_means = X.mean()

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))

display(df)


# Keras model
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

model = build_keras_model()
history = model.fit(X_train, y_train, epochs=200, batch_size=64, callbacks=[early_stop], validation_split=0.2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training Curve")
plt.show()


# Random forest model
#model = build_random_forest_model()




# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)



print(f"MAE: {mae:.2f}") # MAE (Mean Absolute Error): average difference between predicted and true values
print(f"RMSE: {rmse:.2f}") # RMSE: like MAE, but penalizes large errors more
print(f"R²: {r2:.2f}") # R² Score: measures how well your model explains variance (closer to 1 is better)

# Get permutation importance
importances = permutation_importance_manual(model, X_test, y_test)
sorted_idx = np.argsort(importances)
plt.figure(figsize=(8, 6))
plt.barh(np.array(X.columns)[sorted_idx], importances[sorted_idx])
plt.xlabel("Mean Decrease in MAE")
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()

# Save everything
model.save("ml-backend/model/model_nn.keras") # Keras model
#joblib.dump(model, "ml-backend/model/model_rf.pkl") # random forest model
joblib.dump(scaler, "ml-backend/model/scaler.pkl")
joblib.dump(feature_means, "ml-backend/model/feature_means.pkl")
print("Training complete, model saved.")