"""Unicode symbol selection and tokenization pre-screening.

Selects symbols from Miscellaneous Technical and Geometric Shapes Unicode blocks
that are unlikely to appear as formal systems in any training corpus.
"""

# 20 candidate symbols from Miscellaneous Technical (U+2300-23FF)
# and Geometric Shapes (U+25A0-25FF) / Geometric Shapes Extended (U+1F780-1F7FF)
# Plus some from Miscellaneous Symbols (U+2600-26FF)
CANDIDATE_SYMBOLS = [
    "◈",  # U+25C8 White Diamond Containing Black Small Diamond
    "⬡",  # U+2B21 White Hexagon
    "⟐",  # U+27D0 White Diamond with Centred Dot
    "⧫",  # U+29EB Black Lozenge
    "◆",  # U+25C6 Black Diamond
    "▲",  # U+25B2 Black Up-Pointing Triangle
    "●",  # U+25CF Black Circle
    "■",  # U+25A0 Black Square
    "⬢",  # U+2B22 Black Hexagon
    "⏣",  # U+23E3 Benzene Ring with Circle
    "⎔",  # U+2394 Software-Function Symbol
    "⌬",  # U+232C Benzene Ring
    "⍟",  # U+235F APL Functional Symbol Circle Star
    "⏧",  # U+23E7 Electrical Intersection
    "◎",  # U+25CE Bullseye
    "▣",  # U+25A3 White Square Containing Black Small Square
    "⬟",  # U+2B1F Pentagon
    "⬠",  # U+2B20 White Pentagon
    "⏢",  # U+23E2 White Trapezium
    "⌭",  # U+232D Cylindricity
]


def screen_symbols_tiktoken(symbols: list[str]) -> dict[str, int]:
    """Screen symbols against tiktoken (GPT-class proxy).

    Returns dict mapping symbol -> token count.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return {s: len(enc.encode(s)) for s in symbols}
    except ImportError:
        return {}


def screen_symbols_transformers(symbols: list[str], model_name: str) -> dict[str, int]:
    """Screen symbols against a HuggingFace tokenizer.

    Returns dict mapping symbol -> token count.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return {s: len(tok.encode(s, add_special_tokens=False)) for s in symbols}
    except (ImportError, OSError):
        return {}


def select_symbols(n: int = 12, candidates: list[str] | None = None) -> list[str]:
    """Select the n symbols with most consistent token counts across tokenizers.

    Tries tiktoken first (fast, no downloads). Falls back to using all candidates
    ranked by Unicode block preference if no tokenizers are available.

    Returns list of n selected symbols.
    """
    if candidates is None:
        candidates = CANDIDATE_SYMBOLS

    token_counts: dict[str, list[int]] = {s: [] for s in candidates}

    # Screen with tiktoken (GPT-class proxy)
    tk_counts = screen_symbols_tiktoken(candidates)
    if tk_counts:
        for s, count in tk_counts.items():
            token_counts[s].append(count)

    # If we have tokenizer data, rank by consistency (prefer 1-token symbols)
    if any(counts for counts in token_counts.values()):
        def sort_key(sym: str) -> tuple[float, float, int]:
            counts = token_counts[sym]
            if not counts:
                return (999.0, 999.0, ord(sym))
            mean_tokens = sum(counts) / len(counts)
            variance = sum((c - mean_tokens) ** 2 for c in counts) / len(counts)
            return (variance, mean_tokens, ord(sym))

        ranked = sorted(candidates, key=sort_key)
    else:
        # No tokenizer available — use candidates in order (they're already curated)
        ranked = candidates

    selected = ranked[:n]
    return selected


# Default symbol set — call select_symbols() with tokenizers for production
DEFAULT_SYMBOLS = [
    "◈", "⬡", "⟐", "⧫", "◆", "▲",
    "●", "■", "⬢", "⏣", "◎", "▣",
]
