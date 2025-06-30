from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score  # Added roc_auc
import joblib
import numpy as np

def train_dropout_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = np.percentile(y_proba, 20)
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return model, metrics