
import time
import requests
import pandas as pd

from predict import predict_traffic

API_URL = "http://127.0.0.1:8000/api/alert"

COLUMNS = [
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

def main():
    data = pd.read_csv("KDDTest+.txt", header=None, names=COLUMNS)

    print("Simülatör başladı.")
    print("API:", API_URL)

    # İlk denemede yalnızca 20 kayıt gönder
    for index, row in data.head(20).iterrows():
        record = row[COLUMNS[:41]].to_dict()

        # Modelin tahminini al
        result = predict_traffic(record)

        # Backend'e gönderilecek JSON
        payload = {
            "features": record,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "actual_label": str(row["label"])
        }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=5
            )

            print(
                f"Kayıt {index + 1}: "
                f"{result['prediction']} | "
                f"Güven: {result['confidence']} | "
                f"HTTP: {response.status_code}"
            )

            response.raise_for_status()

        except requests.RequestException as error:
            print("Gönderim başarısız:", error)
            print("Backend çalışıyor mu ve JSON formatı doğru mu?")
            break

        # Trafik akışını taklit et
        time.sleep(1)

    print("Simülasyon tamamlandı.")


if __name__ == "__main__":
    main()
