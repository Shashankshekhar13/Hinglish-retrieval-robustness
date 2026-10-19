import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hcsir.lexicon import build_lexicon

def main():
    muse_path = Path("data/raw/muse/hi-en.txt")
    if not muse_path.exists():
        print(f"Error: {muse_path} does not exist. Please download it first.")
        sys.exit(1)

    print("Building RomanLexicon from MUSE hi-en dictionary...")
    lex = build_lexicon(muse_path)

    out_lex_path = Path("data/lexicon/roman_hi_en.json")
    print(f"Saving lexicon to {out_lex_path}...")
    lex.save(out_lex_path)

    # Compute stats for paper table
    entries = sum(len(glosses) for glosses in lex.exact_map.values())
    unique_exact = len(lex.exact_map)
    unique_keys = len(lex.key_map)
    unique_coarse = len(lex.coarse_map)

    # mean glosses per key
    total_glosses_keys = sum(len(g) for g in lex.key_map.values())
    mean_glosses_per_key = total_glosses_keys / max(1, unique_keys)

    # collision rate: keys with >= 2 distinct top-glosses (meaning key maps to multiple gloss candidates)
    # in our ranking, each key has a ranked list of glosses. If len(glosses) >= 2, there is ambiguity/collision.
    collision_count = sum(1 for g in lex.key_map.values() if len(g) >= 2)
    collision_rate = collision_count / max(1, unique_keys)

    stats = {
        "entries": entries,
        "unique_exact_forms": unique_exact,
        "unique_keys": unique_keys,
        "unique_coarse_keys": unique_coarse,
        "mean_glosses_per_key": round(mean_glosses_per_key, 2),
        "collision_rate": round(collision_rate, 4),
        "collision_count": collision_count
    }

    out_stats_path = Path("outputs/tables/lexicon_stats.json")
    out_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Lexicon built successfully.")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
