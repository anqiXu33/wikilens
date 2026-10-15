"""Navigation sections: headings whose content is links and citations, not prose.

Used by the fetcher (skip the section) and by the chunker (also skip it when
it comes from an article cache written before the fetcher filtered it).
No heavy imports, so the chunker can use it without loading the fetcher.
"""

SECTION_SEP = " > "

NAVIGATION_HEADINGS = {
    "en": [
        "see also",
        "references",
        "literature",
        "notes",
        "notes and references",
        "citations",
        "works cited",
        "further reading",
        "external links",
        "sources",
        "bibliography",
        "footnotes",
    ],
    "de": [
        "literatur",
        "einzelnachweise",
        "weblinks",
        "siehe auch",
        "quellen",
        "anmerkungen",
        "belege",
        "fußnoten",
    ],
    "zh": [
        "参见", "參見", "参看", "參看", "相关条目", "相關條目",
        "参考资料", "參考資料", "参考文献", "參考文獻", "参考来源", "參考來源",
        "注释", "註釋", "注釋", "脚注", "腳註",
        "外部链接", "外部連結", "延伸阅读", "延伸閱讀", "扩展阅读", "擴展閱讀",
    ],
    "it": [
        "vedi anche",
        "note",
        "bibliografia",
        "collegamenti esterni",
        "fonti",
        "riferimenti",
        "altri progetti",
    ],
    "fr": [
        "voir aussi",
        "notes",
        "références",
        "liens externes",
        "bibliographie",
        "sources",
        "annexes",
        "notes et références",
    ],
}

_HEADINGS = {lang: {h.casefold() for h in hs} for lang, hs in NAVIGATION_HEADINGS.items()}


def is_navigation_heading(title: str, lang: str) -> bool:
    return " ".join(title.split()).casefold() in _HEADINGS.get(lang, set())


def is_navigation_section(section: str, lang: str) -> bool:
    """True if `section` (a path "A > B") is, or lies under, a navigation heading."""
    return any(is_navigation_heading(part, lang) for part in section.split(SECTION_SEP))
