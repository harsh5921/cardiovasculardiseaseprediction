# model.py

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

MODELS = {
    "log_reg": SGDClassifier(loss='log_loss', max_iter=1000, random_state=42),
    "hist_gb": HistGradientBoostingClassifier(random_state=42),
    "lightgbm": lgb.LGBMClassifier(random_state=42),
    "xgboost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
}

PARAM_GRID = {
    "log_reg": {
        'alpha': [0.0001, 0.001, 0.01],
        'penalty': ['l2', 'l1', 'elasticnet']
    },
    "hist_gb": {
        'learning_rate': [0.01, 0.1],
        'max_iter': [100, 200],
        'max_leaf_nodes': [31, 63]
    },
    "lightgbm": {
        'num_leaves': [31, 63],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200]
    },
    "xgboost": {
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200]
    }
    
}
