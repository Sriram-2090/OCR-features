import os
from src.field_reader.trocr_aligner import get_trocr_aligner

aligner = get_trocr_aligner()
test_images = [
    r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\scraped_candidates\images\ENG_CAND_058.jpg",
    r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\scraped_candidates\images\ENG_CAND_023.jpg",
    r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\DATASET DYSGRAPHIA HANDWRITING\Potential Dysgraphia\PD (1).jpg",
]

for p in test_images:
    if os.path.exists(p):
        res = aligner.predict_and_align(p)
        print("="*60)
        print("Image:", os.path.basename(p))
        print("  Raw Text :", res["raw_text"])
        print("  Cleaned  :", res["text"])
        print("  Mean Conf:", res["mean_conf_pct"], "%")
        print("  Tokens   :", len(res["tokens"]))
        for t in res["tokens"][:8]:
            print(f"    Token '{t['char']}': bbox={t['bbox']}, conf={t['conf_pct']}%")
