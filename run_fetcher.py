from app.services.fetcher import fetch_multilang_from_url

# Example Wikipedia URL and target languages
url = "https://en.wikipedia.org/wiki/COVID-19_pandemic"
langs = ["en", "zh", "de", "sw", "eu"]

# Run fetcher
articles = fetch_multilang_from_url(url, langs)

# Show results summary
for lang, df in articles.items():
    print(f"\n==== {lang.upper()} ====")
    print(df.head(3))



