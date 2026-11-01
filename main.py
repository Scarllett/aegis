
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

train = pd.read_csv("KDDTrain+.txt", names=columns)
test = pd.read_csv("KDDTest+.txt", names=columns)

# difficulty sütununu modelde kullanmıyoruz
train = train.drop(columns=["difficulty"])
test = test.drop(columns=["difficulty"])

# Hedef: normal trafik veya saldırı
y_train = (train["label"] != "normal").astype(int)
y_test = (test["label"] != "normal").astype(int)

X_train = train.drop(columns=["label"])
X_test = test.drop(columns=["label"])

# Kategorik sütunları sayısal değerlere dönüştür
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Eğitim ve test sütunlarını eşitle
X_train, X_test = X_train.align(
    X_test, join="left", axis=1, fill_value=0
)
X_test = X_test.fillna(0)

print("Eğitim verisi:", X_train.shape)
print("Test verisi:", X_test.shape)
print("\nModel eğitiliyor...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
    n_jobs=4
)

model.fit(X_train, y_train)


joblib.dump(
    {
        "model": model,
        "feature_columns": X_train.columns.tolist()
    },
    "xgboost_model.pkl"
)

print("Model başarıyla kaydedildi!")

predictions = model.predict(X_test)

print("\nDoğruluk (Accuracy):",
      round(accuracy_score(y_test, predictions), 4))

print("\nSınıflandırma raporu:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Saldırı"],
    zero_division=0
))

print("Karışıklık matrisi:")
print(confusion_matrix(y_test, predictions))
