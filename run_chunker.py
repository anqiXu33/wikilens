from app.services.chunker import get_chunked_dataframe
import pandas as pd
import os

# 配置
LANGUAGES = ["en", "zh", "de", "sw", "eu"]
FETCHED_DIR = os.path.join("outputs", "fetched")
CHUNKED_DIR = os.path.join("outputs", "chunked")
THRESHOLD = 0.6

# 创建输出文件夹（如不存在）
os.makedirs(CHUNKED_DIR, exist_ok=True)

# 主循环
for lang in LANGUAGES:
    fetched_path = os.path.join(FETCHED_DIR, f"fetched_{lang}.json")
    chunked_path = os.path.join(CHUNKED_DIR, f"chunked_{lang}.json")

    try:
        print(f"[*] Processing: {lang} ...")

        if not os.path.exists(fetched_path):
            raise FileNotFoundError(f"File {fetched_path} does not exist")

        df = pd.read_json(fetched_path)

        if df.empty:
            print(f"[!] {lang}: fetched data is empty.")
            continue

        chunked_df = get_chunked_dataframe(df, threshold=THRESHOLD)
        chunked_df.to_json(chunked_path, orient="records", force_ascii=False, indent=2)

        print(f"[✓] {lang}: chunked and saved to {chunked_path}")

    except Exception as e:
        print(f"[!] {lang}: failed – {e}")




