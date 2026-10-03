import os, glob, time
import pandas as pd
import numpy as np
from src.field_reader.trocr_aligner import get_trocr_aligner


aligner = get_trocr_aligner()

df = pd.read_csv('data/form_fields/metadata.csv')
print(f"Testing on first 10 fields...")

correct = 0
total = 10

for i in range(total):
    row = df.iloc[i]
    fid = int(row['field_id'])
    gt = str(row['ground_truth'])
    ftype = str(row['field_type'])
    fname = f"data/form_fields/field_{fid:04d}_{ftype}.png"
    if not os.path.exists(fname):
        fname = f"data/form_fields/field_{fid:04d}.png"
    
    t0 = time.time()
    res = aligner.predict_and_align(fname)
    dt = time.time() - t0
    
    pred = res['clean_text']
    num_tokens = len(res['tokens'])
    print(f"Field #{fid:02d} [{ftype:4s}] GT: '{gt}' | TrOCR: '{pred}' | Tokens: {num_tokens} | ({dt*1000:.1f}ms)")
