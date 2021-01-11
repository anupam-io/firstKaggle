import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('./input/train.csv', index_col='Id')
X_test_full = pd.read_csv('./input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


print(X_train.head())

from sklearn.ensemble import RandomForestRegressor

# Define the models






from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

my_model = None
my_err = 10**10

import random

for i in range(1):
    base_model = RandomForestRegressor(
        n_estimators=1000,
        criterion='mse', 
        max_depth=50,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=0,
        max_features="sqrt"
    )

    
    mae = score_model(base_model)

    print("Model %d MAE: %d" % (i+1, mae))
    if mae < my_err:
        my_model = base_model
        my_err = mae
    print("My error: ", my_err)


# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

print(my_err)


# Save predictions in format used for competition scoring
# output = pd.DataFrame(
#     {'Id': X_test.index,
#     'SalePrice': preds_test})

# output.to_csv('submission.csv', index=False)