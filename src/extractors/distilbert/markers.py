"""Entity marker insertion for candidate pairs: ... [E1] x [/E1] ... [E2] y [/E2] ..."""
MARKER_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def insert_markers_by_offsets(text: str, s1: int, e1: int, s2: int, e2: int) -> str:
    """
    Insert markers around two character spans of `text`. Entity 1 always receives [E1]..[/E1]
    and entity 2 [E2]..[/E2], regardless of order in the sentence. Overlapping/nested spans cannot be
    marked consistently, so they raise ValueError (callers treat these as unscorable pairs).
    """
    if not (0 <= s1 < e1 <= len(text) and 0 <= s2 < e2 <= len(text)):
        raise ValueError("span out of range")
    if max(s1, s2) < min(e1, e2):
        raise ValueError("overlapping entity spans")
    inserts = [(s1, "[E1] "), (e1, " [/E1]"), (s2, "[E2] "), (e2, " [/E2]")]
    for pos, tok in sorted(inserts, key=lambda x: -x[0]):
        text = text[:pos] + tok + text[pos:]
    return " ".join(text.split())
