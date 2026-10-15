# run_chunk_threshold_experiment.py

from app.services.chunker import get_chunked_dataframe
import pandas as pd

# 选择你要测试的语言和文件
lang = "zh"
df = pd.read_json(f"outputs/fetched/fetched_{lang}.json")

# 设置一组要测试的 threshold 值
thresholds = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

# 开始实验
print(f"📊 Threshold Test for: {lang.upper()}")

for t in thresholds:
    chunked_df = get_chunked_dataframe(df, lang=lang, threshold=t)
    num_chunks = chunked_df['chunk_id'].nunique()
    num_sentences = len(chunked_df)
    avg_chunk_size = num_sentences / num_chunks if num_chunks else 0

    print(f"Threshold = {t:.2f} → Chunks = {num_chunks}, Avg chunk size = {avg_chunk_size:.2f}")
