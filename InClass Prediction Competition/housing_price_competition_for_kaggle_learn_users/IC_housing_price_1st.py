# Reference: https://qiita.com/Hawaii/items/52e80b5fe1532a5d2351
# Score: 18110.26229
# round = 500, early_stopping_rounds=10, No cleaning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head(10))
print(train.dtypes)

train["isTrain"] = True
test["isTrain"] = False


# append training data and test data
all = train.append(test)
all.index = all["Id"]
all.drop("Id", axis = 1, inplace = True)
print(all.head(5))

# transform into dummy var
all = pd.get_dummies(all, drop_first=True)

#devide data into trainiing data and test data again
train = all[all["isTrain"] == True]
train = train.drop(["isTrain"], axis = 1)
test = all[all["isTrain"] == False]
test = test.drop(["isTrain"], axis = 1)
test = test.drop(["SalePrice"], axis = 1)

#data division
y = train["SalePrice"].values
X = train.drop("SalePrice", axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True,random_state=9876)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
dtest = xgb.DMatrix(test.values)

params = {
        'objective': 'reg:squarederror','silent':0, 'random_state':9876, 
        # 学習用の指標 (RMSE)
        'eval_metric': 'rmse',
    }
round = 100
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

model = xgb.train(params,
                    dtrain,
                    round,
                    early_stopping_rounds=10,
                    evals=watchlist,
                    )

prediction_XG = model.predict(dtest, ntree_limit = model.best_ntree_limit)

prediction_XG = np.round(prediction_XG)

submission = pd.DataFrame({"id": test.index, "SalePrice": prediction_XG})
submission.to_csv('fy_submission_first_try.csv', index=False)
print("Your submission was successfully saved!")