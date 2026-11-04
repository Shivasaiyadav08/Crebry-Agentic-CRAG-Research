"""Small, source-preserving provider payloads. Never summarize evidence."""


def recent_history(state, *, messages=4, characters=4000):
    """Keep whole recent messages within a budget; never cut a claim in half."""
    result = []
    for item in reversed(state.get("history", [])[-messages:]):
        content = item["content"]
        if len(content) > characters:
            continue
        result.append({"role": item["role"], "content": content})
        characters -= len(content)
    return list(reversed(result))


def evidence_payload(sources):
    """Keep original text, ID, and provenance; omit storage/retrieval bookkeeping."""
    return [
        {key: source[key] for key in ("id", "kind", "name", "page", "url", "text") if key in source}
        for source in sources
    ]


def unique_sources(sources):
    """Only collapse identical chunks from the same source; preserve attribution."""
    seen, result = set(), []
    for source in sources:
        identity = (
            source.get("kind"),
            source.get("document_id"),
            source.get("url"),
            source.get("name"),
            source.get("page"),
            source.get("text"),
        )
        if identity not in seen:
            seen.add(identity)
            result.append(source)
    return result
