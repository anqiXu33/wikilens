from app.services.aligner_en_to_source import match_en_to_lang
from tqdm import tqdm

LANGS = ["zh", "de", "sw", "eu"]

if __name__ == "__main__":
    for lang in tqdm(LANGS, desc="Running English → Source matching (threshold-based)"):
        match_en_to_lang(lang)

