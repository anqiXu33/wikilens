from app.services.embedder import embed_language
from tqdm import tqdm

LANGS = ["zh", "de", "sw", "eu", "en"]

if __name__ == "__main__":
    for lang in tqdm(LANGS, desc="Embedding languages"):
        embed_language(lang)

