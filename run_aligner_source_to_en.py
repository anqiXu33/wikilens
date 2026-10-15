# run_aligner_source_to_en.py

from app.services.aligner_source_to_en import align_language
from tqdm import tqdm

LANGS = ["zh", "de", "sw", "eu"]

if __name__ == "__main__":
    for lang in tqdm(LANGS, desc="Aligning to English"):
        align_language(lang)
