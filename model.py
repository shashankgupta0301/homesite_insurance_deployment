import pandas as pd
import pickle
import os
import lightgbm as gbm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn import model_selection

os.chdir("binaries/files")
RFE_columns = pd.read_csv('RFE_features_1.csv').columns
df = pd.read_csv('train.csv')
df_RFE = df[RFE_columns]
df_RFE.drop(columns=['Original_Quote_Date', 'SalesField8'], axis=1, inplace=True)

RFE_params = {
    'boosting_type': 'gbdt',
    'lambda_l1': 4.540006226304331e-08,
    'lambda_l2': 4.715716309514142,
    'num_leaves': 105,
    'feature_fraction': 0.89,
    'bagging_fraction': 1,
    'bagging_freq': 4,
    'min_child_samples': 65,
    'max_bin': 20,
    'learning_rate': 0.14, }

RFE_gbm = gbm.LGBMClassifier(**RFE_params)
X_train, X_val, y_train, y_val = model_selection.train_test_split(df_RFE.drop('QuoteConversion_Flag', axis=1),df_RFE['QuoteConversion_Flag'], random_state=42,stratify=df_RFE['QuoteConversion_Flag'])
finalModel = Pipeline([('label_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-99)),('RFE_gbm', RFE_gbm)])
finalModel.fit(X_train, y_train)

# Save the Model
filename = 'homesite_prediction_model.pkl'
pickle.dump(finalModel, open(filename, 'wb'))
