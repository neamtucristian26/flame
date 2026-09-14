# 🔥 FLAME: Forensic Language-Audio Model Exposure

This repository hosts the official source code and data protocols for our SLT 2026 paper:
> Cristian-Teodor Neamtu, Serban Mihalache, Stefan Smeu, Dan Oneata, Horia Cucu, Dragos Burileanu, "Towards Zero-Shot Attribution of Synthetic Speech via Audio-Text Contrastive Retrieval", accepted at IEEE SLT 2026, Palermo, Italy.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)

## Abstract

Audio deepfake forensics is moving beyond a simple real-or-fake verdict toward attribution: which system generated the audio clip? Most source-attribution methods cast this as closed-set classification, so they cannot name a generator that was absent from training, a gap that widens with every newly released text-to-speech (TTS) system. We instead frame attribution as cross-modal retrieval: each generator is described in natural language, and a clip is attributed by retrieving the description closest to it in a shared audio–text embedding space. Adding a new system then takes nothing more than writing its description, with no retraining and no new classifier head. Our model couples a frozen Wav2Vec2-BERT audio encoder with a frozen E5 text encoder and aligns them through small trainable projection heads, using a contrastive objective that combines a cross-modal supervised-contrastive loss with intra-modal terms. We evaluate on MLAAD v9 (140 TTS systems, 51 languages) under 10-fold leave-systems-out cross-validation. For generators it has never encountered before, the model reaches a model-level mean reciprocal rank (MRR) of 58.4\%. Even when the correct model is not identified, the audio clip is matched to systems that share the true generator's vocoder, acoustic model, or architecture. Because the same embedding space also answers natural language attribute queries, one set of descriptions covers both open-set attribution and attribute-level forensic profiling.

## Citation

```bibtex
@inproceedings{neamtu2026flame,
      title={Towards Zero-Shot Attribution of Synthetic Speech via Audio-Text Contrastive Retrieval},
      author={Cristian-Teodor Neamtu and Serban Mihalache and Stefan Smeu and Dan Oneata and Horia Cucu and Dragos Burileanu},
      booktitle={IEEE Spoken Language Technology Workshop (SLT)},
      year={2026},
}
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download [MLAAD v9](https://huggingface.co/datasets/mueller91/MLAAD) and extract per-layer Wav2Vec2-BERT embeddings using `tools/extract_layer_embeddings.py`.

```bash
python tools/extract_layer_embeddings.py \
    --input_dir  /path/to/MLAAD_v9/fake \
    --output_dir /path/to/mlaad_v9_layer_stats

export FLAME_AUDIO_ROOT=/path/to/mlaad_v9_layer_stats
```

Input structure: `{input_dir}/{language}/{model_name}/{audio}.wav`  
Output structure: `{output_dir}/{language}/{model_name}/{audio}.pt`

Each output tensor is `(24, 1024)` float16: the time-averaged activation of each of the 24 transformer layers, which `tts_zs/model.py` then combines with a learnable weighted sum.

## Natural Language Descriptions

Each TTS system is described in free-form text rather than assigned a class label. Descriptions are built in two stages and are committed under `data/`, so they can be used as-is.

**Stage A — structured representation.** We construct a factsheet for each TTS model, describing it along 11 predefined dimensions (architecture family, acoustic model, vocoder, speaker type, training corpus, distinctive artifacts, …). Attribute values are free text rather than fixed labels: an acoustic model may read *posterior encoder + flow prior + monotonic alignment search*. Each value was retained only where it could be corroborated against public documentation. The result is `data/factsheets.json`.

**Stage B — natural language rendering.** The factsheet is rendered into free-form text with Claude Opus 4.7: five descriptions per model that differ in style and emphasis (technical, forensic, dataset & training, capabilities & positioning, engineer-to-engineer) while expressing the same underlying attributes, each prefixed with a language header. This diversity pushes the audio encoder to align with the underlying characteristics of a system rather than with one particular wording.

Because a system is defined as a (model, language) pair, the 140 TTS models yield **298 systems**. The E5 embeddings of these descriptions are cached to `artifacts/embedding_cache/` on first use and are not committed.

| File | Contents |
|:-----|:---------|
| `factsheets.json` | Stage-A factsheet: 11 structured attributes per TTS model (140 models) |
| `arch_labels.json` | Model → architecture family, used for the stratified fold split |
| `labels_lang.json` | Rendered descriptions: 298 systems × 5 styles, language header prepended |
| `labels_lang_aug24ho4.json` | Augmentation pool (24 variants/system), held-out variant excluded |
| `labels_lang_no_header.json` | Header-stripped descriptions (language ablation) |

## Reproducing the Paper Experiments

All experiments use 10-fold leave-systems-out cross-validation (126 seen / 14 unseen TTS models per fold), repeated over 3 seeds (42, 43, 44). Folds are generated deterministically from `(fold_id, seed)` by `tts_zs/data.py`, so no split files are needed.

**Main experiment — FLAME (3 seeds × 10 folds):**

```bash
bash scripts/launch_5axis_ho4_3seed.sh
```

This wraps `scripts/run_cfg2_ablsc.sh`, which supplies the reported hyperparameters: λ=0.298, γ=0.401, τ_cm=0.020, τ_a=0.045, τ_t=0.070, AdamW at lr=1e-4 / weight decay=1e-4, 100 epochs, and batches of 256 clips sampled as 32 clips from each of 8 (model, language) pairs.

`scripts/train_zero_shot.py` defaults to these values, so a single fold can be trained directly. Launch through the shell scripts to reproduce the reported configuration in full: they additionally set the augmentation pool, the attribute-template prompts and the fixed held-out description variant.

**Specialized classifier baselines (7 attribute heads):**

```bash
bash scripts/run_baseline_allheads.sh          # closed-set training + evaluation
python scripts/eval_baseline_unseen_attrs.py   # same heads, evaluated on unseen-system clips
```

**Evaluation suite — retrieval, attribute profiling, text-to-audio:**

```bash
python scripts/eval_sibling_ensemble.py --ckpt-tmpl ... --out-tag f5_p10ho4
python scripts/eval_attribute_retrieval.py --ckpt-tmpl ... --seed 42
python scripts/eval_text_to_audio.py
```

## Results

All results use a frozen **Wav2Vec2-BERT** audio encoder and a frozen **E5-large-v2** text encoder, with only the 24 layer-sum weights and two `Linear(1024→256)` projection heads trained (~0.5M parameters). Reported as mean ± sd over 3 seeds.

### Zero-shot system attribution (audio → text retrieval)

`unseen-restricted` searches a gallery of unseen systems only; `unseen-full` searches all 298 systems, seen and unseen.

| Scenario | Level | Hit@1 ↑ | MRR ↑ |
|:---------|:------|--------:|------:|
| seen | system | 86.2 ± 0.3 | 91.3 ± 0.2 |
| seen | model | 93.3 ± 0.3 | 96.3 ± 0.1 |
| unseen-restricted | system | 60.2 ± 0.9 | 72.2 ± 0.8 |
| unseen-restricted | model | 71.4 ± 1.5 | 81.4 ± 1.1 |
| unseen-full | system | 34.7 ± 0.4 | 50.0 ± 0.6 |
| **unseen-full** | **model** | **44.3 ± 0.3** | **58.4 ± 0.6** |

### Attribute profiling on unseen systems

Retrieval against full system descriptions, against short attribute-specific prompts, and the specialized classifier baseline evaluated on the same unseen-system clips.

| Attribute | Full descriptions ↑ | Attribute prompts ↑ | Specialized Classifier ↑ |
|:----------|-------------------:|-------------------:|-------------:|
| Architecture | **67.0 ± 1.1** | 49.8 ± 0.5 | 62.7 ± 0.5 |
| Acoustic model | **73.5 ± 1.4** | 61.7 ± 1.0 | 69.8 ± 0.8 |
| Vocoder | **70.6 ± 0.4** | 57.9 ± 0.9 | 66.8 ± 0.4 |
| Speaker type | **77.0 ± 0.5** | 60.2 ± 0.3 | 63.9 ± 1.3 |
| Language family | 86.5 ± 0.6 | 80.9 ± 0.9 | **87.4 ± 0.4** |

### Graceful degradation

When the exact (model, language) pair is missed, the fraction of top-1 retrievals whose system still shares each attribute with the true source.

| Attribute | Recovered ↑ |
|:----------|-----------:|
| Architecture | 48.2 ± 2.1 |
| Acoustic model | 58.3 ± 3.3 |
| Vocoder | 52.2 ± 1.3 |
| Speaker type | 64.0 ± 0.7 |
| Language family | 79.5 ± 1.0 |

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

**Cristian-Teodor Neamtu** — cristian.neamtu [at] upb [dot] ro  
For questions about the code, please open an [issue](../../issues).
For questions about the paper, feel free to reach out by email.

## Acknowledgements

This work was supported by the European Union – NextGenerationEU, through the National Recovery and Resilience Plan (PNRR), Component 9, Investment 4, under project SENSE, THINK @ POLITEHNICA BUCUREȘTI (SENTHIPoli) No. 6.PI/I4/C9. The content of this material does not necessarily represent the official position of the European Union or the Government of Romania.
