import os, sys, time, cv2
import numpy as np

from src.field_reader.pipeline import FormReaderPipeline
from src.field_reader.trocr_ocr_engine import TrOCROCRPipeline
from src.field_reader.trocr_aligner import get_trocr_aligner
from src.field_reader.dictionary_engine import get_lexicon_engine
from src.field_reader.llm_refiner import get_llm_refiner

pipeline = FormReaderPipeline(model_path='models/field_cnn.pth')
trocr = get_trocr_aligner()
trocr_ocr = TrOCROCRPipeline()
lexicon = get_lexicon_engine()
llm = get_llm_refiner()

def universal_ocr_recognize(img_bgr, field_type="General", is_comb_box=False):
    t0 = time.time()
    h, w = img_bgr.shape[:2]
    
    f_type_norm = (field_type or "General").lower()
    is_structured = any(k in f_type_norm for k in ["date", "pin", "code"]) or is_comb_box
    
    raw_ocr = ""
    conf = 0.85
    annotated_b64 = None
    
    # Branch 1: Structured Form Field
    if is_structured and pipeline.model_loaded:
        res_tri = pipeline.process(
            img_bgr,
            field_type=field_type.capitalize(),
            is_comb_box=is_comb_box,
            mode="tri_engine"
        )
        raw_ocr = res_tri.get("text", "")
        conf = res_tri.get("confidence", 0.90) or 0.90
        # If Tri-Engine matched FSM grammar with high conf, it is golden!
        print(f"[Universal Router] Structured Tri-Engine OCR: {raw_ocr} (conf: {conf})")
    
    # Branch 2: Freeform / Multi-line / General Handwriting (or fallback)
    if not raw_ocr or not is_structured:
        # Check if multi-line
        if h > 180 and w > 200:
            print("[Universal Router] Multi-line handwriting detected. Using TrOCR Multi-Line Pipeline.")
            res_trocr = trocr_ocr.transcribe(img_bgr)
            raw_ocr = res_trocr.full_text
            conf = res_trocr.mean_confidence
        else:
            print("[Universal Router] Single-line handwriting detected. Using TrOCR with Aspect Ratio Padding.")
            res_trocr = trocr.predict_and_align(img_bgr, field_type=field_type)
            raw_ocr = res_trocr.get("text", "")
            conf = res_trocr.get("mean_conf", 0.85)
            annotated_b64 = res_trocr.get("annotated_image_b64")
        print(f"[Universal Router] TrOCR OCR: {raw_ocr} (conf: {conf})")
    
    t_ocr = time.time()
    
    # Tier 1: Fast Lexicon
    dict_text, dict_notes = lexicon.correct_sentence(raw_ocr)
    t_lex = time.time()
    print(f"[Universal Router] Tier 1 Lexicon: {dict_text}")
    
    # Tier 2: Local LLM Refiner (Qwen 2.5)
    llm_res = llm.refine_ocr(dict_text, field_type=field_type, confidence=conf)
    refined_text = llm_res.get("corrected_text", dict_text)
    reasoning = llm_res.get("reasoning", "")
    t_llm = time.time()
    print(f"[Universal Router] Tier 2 Qwen 2.5 Refined: {refined_text} (Reason: {reasoning})")
    
    return {
        "text": refined_text,
        "raw_ocr_text": raw_ocr,
        "tier1_text": dict_text,
        "llm_text": refined_text,
        "llm_reasoning": reasoning,
        "confidence": conf,
        "annotated_b64": annotated_b64,
        "latencies": {
            "ocr_ms": round((t_ocr - t0) * 1000, 1),
            "lexicon_ms": round((t_lex - t_ocr) * 1000, 1),
            "llm_ms": round((t_llm - t_lex) * 1000, 1),
            "total_ms": round((t_llm - t0) * 1000, 1),
        }
    }

# Test 1: Field 1 (Date)
print("\n--- TEST 1: Field 0001 (Date: 02/06/1984) ---")
img1 = cv2.imread('data/form_fields/field_0001_date.png')
r1 = universal_ocr_recognize(img1, field_type="Date", is_comb_box=False)
print("Result 1:", r1["text"], "| Matches GT:", r1["text"] == "02/06/1984")

# Test 2: Field 5 (Code: ELQ-7177)
print("\n--- TEST 2: Field 0005 (Code: ELQ-7177) ---")
img5 = cv2.imread('data/form_fields/field_0005_code.png')
r5 = universal_ocr_recognize(img5, field_type="Code", is_comb_box=True)
print("Result 2:", r5["text"], "| Matches GT:", r5["text"] == "ELQ-7177")

# Test 3: Dysgraphia handwriting sample
print("\n--- TEST 3: Handwriting Sample ---")
dys_p = r'C:\Users\SRIRAM\Documents\GitHub\Dysgraphia\Dysgraphia-Detection\DATASET DYSGRAPHIA HANDWRITING\Low Potential Dysgraphia\LPD (1).jpg'
if os.path.exists(dys_p):
    img_dys = cv2.imread(dys_p)
    r3 = universal_ocr_recognize(img_dys, field_type="General")
    print("Result 3:", r3["text"])
