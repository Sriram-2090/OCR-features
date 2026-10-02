import sys
import os

HACKATHON_ROOT = os.path.abspath(".")
sys.path.insert(0, HACKATHON_ROOT)

from src.field_reader.trocr_ocr_engine import get_trocr_ocr_pipeline

pipe = get_trocr_ocr_pipeline()
print("TrOCR Pipeline initialized!")

img_p = r"C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\scraped_candidates\images\ENG_CAND_058.jpg"
res = pipe.transcribe(img_p)

print("=== TranscriptionResult ===")
print("Full Text:", repr(res.full_text))
print("Mean Confidence:", res.mean_confidence)
print("Total Lines:", len(res.lines))
print("Total Words:", res.total_words)
for li, l in enumerate(res.lines):
    print(f"  Line {li+1}: raw='{l.raw_text}', conf={l.line_confidence:.2f}, words={len(l.words)}")
    for wi, w in enumerate(l.words):
        stroke_info = w.metadata.get("stroke_analysis")
        prims = [p.value for p in stroke_info.primitives] if stroke_info else []
        agr = w.metadata.get("stroke_agreement")
        print(f"    Word {wi+1}: '{w.text}', conf={w.confidence:.2%}, tier={w.tier.value}, bbox={w.bbox}, stroke_agr={agr}, prims={prims}")

signals = res.metadata.get("ocr_dysgraphia_features")
print("\n=== OCR Dysgraphia Features ===")
if signals:
    for f in signals.feature_names:
        attr = f.replace("ocr_", "")
        print(f"  {f}: {getattr(signals, attr, None)}")
