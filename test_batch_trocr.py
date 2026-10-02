import glob, os, torch
from PIL import Image
from transformers import AutoImageProcessor, RobertaTokenizerFast, TrOCRProcessor, VisionEncoderDecoderModel

snapshot_dir = glob.glob(r'C:\Users\SRIRAM\.cache\huggingface\hub\models--microsoft--trocr-base-handwritten\snapshots\*')[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

feature_extractor = AutoImageProcessor.from_pretrained(snapshot_dir)
tokenizer = RobertaTokenizerFast.from_pretrained(snapshot_dir)
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(snapshot_dir).to(device)
model.eval()

files = ['field_0001_date.png', 'field_0002_date.png', 'field_0003_pin.png', 'field_0004_code.png', 'field_0005_date.png']
for fname in files:
    fpath = os.path.join('data/form_fields', fname)
    if os.path.exists(fpath):
        img = Image.open(fpath).convert('RGB')
        pixel_values = processor(img, return_tensors='pt').pixel_values.to(device)
        with torch.no_grad():
            gen_ids = model.generate(pixel_values, max_new_tokens=30)
            text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        print(f"{fname} -> '{text}'")
