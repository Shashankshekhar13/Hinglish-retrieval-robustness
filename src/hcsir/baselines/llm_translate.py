import json, time
from pathlib import Path
from ..generate import Cache

def llm_translate(dataset: str, qid: str, text: str, condition: str, client, cfg, cache_path=None) -> tuple[str, float]:
    g = cfg.get("groq", {})
    prompt_version = g.get("prompt_version", "v1")
    cache_file = cache_path or Path("data/cache") / f"translate_{prompt_version}.jsonl"
    cache = Cache(cache_file)
    key = f"{dataset}:{qid}:{condition}:translate_{prompt_version}"

    cached = cache.get(key)
    if cached:
        return cached["translation"], cached.get("latency", 0.0)

    # Load prompt template
    prompt_template_path = Path("prompts") / f"translate_{prompt_version}.txt"
    if prompt_template_path.exists():
        template = prompt_template_path.read_text(encoding="utf-8")
        prompt = template.replace("{query}", text)
    else:
        prompt = f"Translate this search query into English. Output only the translation.\n{text}"

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=g.get("translate_model", "openai/gpt-oss-20b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256
    )
    latency = time.monotonic() - t0
    translation = response.choices[0].message.content.strip()

    cache.put(key, {"translation": translation, "latency": latency})
    return translation, latency
