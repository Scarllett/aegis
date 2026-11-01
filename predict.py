
import joblib
import pandas as pd

# Kaydedilmiş modeli yükle
saved = joblib.load("xgboost_model.pkl")

model = saved["model"]
feature_columns = saved["feature_columns"]


def predict_traffic(record):
    """
    Tek bir NSL-KDD ağ kaydını analiz eder.
    record: Sütun adlarını ve değerlerini içeren sözlük.
    """

    row = pd.DataFrame([record])

    # Eğitimde kullanılan kodlama yöntemiyle aynı dönüşüm
    row = pd.get_dummies(row)
    row = row.reindex(columns=feature_columns, fill_value=0)

    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return {
        "prediction": "Anomali" if prediction == 1 else "Normal",
        "confidence": round(float(probabilities[prediction]), 4)
    }



if __name__ == "__main__":
    import pandas as pd

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
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]

    # Test dosyasından bir ağ kaydı oku
    sample = pd.read_csv(
        "KDDTest+.txt",
        header=None
    )

    record = dict(zip(columns, sample.iloc[0, :41].tolist()))

    # Model ile tahmin yap
    result = predict_traffic(record)

    print("Gerçek etiket:", sample.iloc[0, 41])
    print("Model tahmini:", result["prediction"])
    print("Güven skoru:", result["confidence"])
