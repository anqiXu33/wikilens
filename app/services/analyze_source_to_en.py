# analyze_source_to_en.py

import json
import pandas as pd
from pathlib import Path

def main():
    LANGS = ["zh", "de", "sw", "eu"]
    INPUT_DIR = Path("outputs/alignments_source_to_en")
    OUTPUT_DIR = Path("outputs/analyze_source_to_en")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang in LANGS:
        input_path = INPUT_DIR / f"{lang}_to_en.json"
        if not input_path.exists():
            print(f"[!] File not found: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for item in data:
            src_id = item["src_chunk_id"]
            src_text = item["src_text"]
            matched = item.get("matches", [])
            tgt_ids = [m["tgt_chunk_id"] for m in matched]
            tgt_texts = [m["tgt_text"] for m in matched]

            if not tgt_ids:
                match_type = "1:0"
            elif len(tgt_ids) == 1:
                match_type = "1:1"
            else:
                match_type = "1:N"

            rows.append({
                "match_type": match_type,
                "src_chunk_id": src_id,
                "src_chunk_text": src_text,
                "matched_en_chunk_ids": "; ".join(tgt_ids),
                "matched_en_chunk_texts": " ||| ".join(tgt_texts)
            })

        df = pd.DataFrame(rows)
        output_path = OUTPUT_DIR / f"{lang}_to_en.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"[✓] Saved: {output_path}")

if __name__ == "__main__":
    main()
