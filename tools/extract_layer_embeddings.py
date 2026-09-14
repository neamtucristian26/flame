import argparse
import logging
import os
import time
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

MODEL_NAME = "facebook/w2v-bert-2.0"
TARGET_SR = 16000
EMBEDDING_DIM = 1024
N_LAYERS = 24
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus"}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a")],
    )

def load_model(device):
    print(f"Loading {MODEL_NAME}...")
    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2BertModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}")
    return fe, model

def discover_audio_files(input_dir, output_dir, no_resume=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    to_process = []
    total_found = 0
    total_skipped = 0
    for lang_dir in sorted(input_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for audio_file in sorted(model_dir.iterdir()):
                if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                total_found += 1
                out = output_dir / lang_dir.name / model_dir.name / f"{audio_file.stem}.pt"
                if not no_resume and out.exists():
                    total_skipped += 1
                    continue
                to_process.append((audio_file, out))
    return to_process, total_found, total_skipped

def extract_single(audio_path, fe, model, device):
    audio, _ = librosa.load(str(audio_path), sr=TARGET_SR)
    inputs = fe(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    layers = outputs.hidden_states[1:]
    pooled = torch.stack([h.mean(dim=1).squeeze(0) for h in layers], dim=0)
    return pooled.half().cpu()

def process_files(to_process, fe, model, device):
    ok = 0
    failed = 0
    for audio_path, out_path in tqdm(to_process, desc="Extracting"):
        try:
            emb = extract_single(audio_path, fe, model, device)
            assert emb.shape == (N_LAYERS, EMBEDDING_DIM), emb.shape
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb, out_path)
            ok += 1
        except Exception as e:
            failed += 1
            logging.error(f"Failed: {audio_path} — {e}")
    return ok, failed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--no_resume", action="store_true")
    args = p.parse_args()

    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.output_dir, "extraction_errors.log")
    setup_logging(log_file)

    fe, model = load_model(device)

    print(f"\nScanning {args.input_dir}...")
    to_process, total, skipped = discover_audio_files(
        args.input_dir, args.output_dir, args.no_resume
    )
    print(f"  Total: {total}  Skipped: {skipped}  To process: {len(to_process)}")
    if not to_process:
        return

    t0 = time.time()
    ok, failed = process_files(to_process, fe, model, device)
    print(f"\nDone in {time.time() - t0:.1f}s  —  ok={ok}  failed={failed}")
    if failed:
        print(f"  See {log_file}")

if __name__ == "__main__":
    main()