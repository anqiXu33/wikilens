# analyze_en_to_source.py

import json
import pandas as pd
from pathlib import Path

def main():
    LANGS = ["zh", "de", "sw", "eu"]
    INPUT_DIR = Path("outputs/alignments_en_to_source")
    OUTPUT_DIR = Path("outputs/analyze_en_to_source")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang in LANGS:
        input_path = INPUT_DIR / f"en_to_{lang}.json"
        if not input_path.exists():
            print(f"[!] File not found: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for item in data:
            en_id = item["tgt_chunk_id"]
            en_text = item["tgt_text"]
            matched = item.get("matches", [])

            src_ids = [m["src_chunk_id"] for m in matched]
            src_texts = [m["src_text"] for m in matched]

            if not src_ids:
                match_type = "1:0"
            elif len(src_ids) == 1:
                match_type = "1:1"
            else:
                match_type = "1:N"

            rows.append({
                "match_type": match_type,
                "en_chunk_id": en_id,
                "en_chunk_text": en_text,
                "matched_src_chunk_ids": "; ".join(src_ids),
                "matched_src_chunk_texts": " ||| ".join(src_texts)
            })

        df = pd.DataFrame(rows)
        output_path = OUTPUT_DIR / f"en_to_{lang}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[✓] Saved {lang} to: {output_path}")

if __name__ == "__main__":
    main()
