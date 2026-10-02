import torch, cv2, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, RobertaTokenizerFast, TrOCRProcessor, VisionEncoderDecoderModel
import glob, sys

snapshot_dir = glob.glob(r'C:\Users\SRIRAM\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten\snapshots\*')[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading TrOCR on {device}...")

feature_extractor = AutoImageProcessor.from_pretrained(snapshot_dir)
tokenizer = RobertaTokenizerFast.from_pretrained(snapshot_dir)
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(snapshot_dir, attn_implementation='eager').to(device)
model.eval()

img = Image.open('data/form_fields/field_0001_date.png').convert('RGB')
orig_w, orig_h = img.size
pixel_values = processor(img, return_tensors='pt').pixel_values.to(device)

with torch.no_grad():
    gen_out = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
    generated_ids = gen_out.sequences[0]

# get tokens
token_strings = [tokenizer.decode([tid]).strip() for tid in generated_ids]
full_text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
print("Full Decoded Text:", full_text)

# compute attention maps
with torch.no_grad():
    out = model(pixel_values=pixel_values, decoder_input_ids=generated_ids.unsqueeze(0), output_attentions=True)
    # average across last 3 decoder layers and all heads
    layers_attn = torch.stack(out.cross_attentions[-3:], dim=0).mean(dim=[0, 2]) # [batch, seq_len, 577]
    cross = layers_attn[0, :, 1:].cpu().numpy() # [seq_len, 576]

for idx, (tid, tok_str) in enumerate(zip(generated_ids, token_strings)):
    if tid in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] or not tok_str:
        continue
    attn_map = cross[idx].reshape(24, 24)
    # find center of mass
    peak_y, peak_x = np.unravel_index(np.argmax(attn_map), attn_map.shape)
    norm_x = peak_x / 24.0
    norm_y = peak_y / 24.0
    img_x = int(norm_x * orig_w)
    img_y = int(norm_y * orig_h)
    print(f"Token {idx:02d}: '{tok_str}' -> spatial focus at ({img_x:3d}, {img_y:2d}) [image {orig_w}x{orig_h}]")
