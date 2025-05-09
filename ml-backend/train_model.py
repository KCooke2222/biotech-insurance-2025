"""

This script trains a machine learning model to predict insurance charges based on various features.

"""

import pandas as pd # handles the dataset
import numpy as np # works with arrays
from IPython.display import display # display processed data
from sklearn.model_selection import train_test_split # splits the dataset into training and testing sets
from sklearn.preprocessing import StandardScaler # normalizes input data
from sklearn.ensemble import RandomForestRegressor # ml model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # analyze model results
from sklearn.metrics import accuracy_score, classification_report # for performance evaluation
from sklearn.preprocessing import LabelEncoder # Convert strings to numbers
from tensorflow import keras
from tensorflow.keras import layers
import joblib # saves the model and scaler for later use


file_path = r"ml-backend\insurance.csv" # path to the dataset
df = pd.read_csv(file_path)
# reads the csv file into a dataframe so that we can actually read and preprocess



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


r'''
# train a random forest model
model = RandomForestRegressor(n_estimators = 150, random_state=42)
model.fit(X_train, y_train)
feature_means = X.mean()
'''

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),    
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mae",
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=1000,
)



# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)



print(f"MAE: {mae:.2f}") # MAE (Mean Absolute Error): average difference between predicted and true values
print(f"RMSE: {rmse:.2f}") # RMSE: like MAE, but penalizes large errors more
print(f"R²: {r2:.2f}") # R² Score: measures how well your model explains variance (closer to 1 is better)

# Save everything
model.save("ml-backend/model/model.keras")
joblib.dump(scaler, "ml-backend/model/scaler.pkl")
joblib.dump(feature_means, "ml-backend/model/feature_means.pkl")
print("Training complete, model saved.")