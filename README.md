# 🔥 FLAME: Forensic Language-Audio Model Exposure

This repository hosts the official source code and data protocols for our SLT 2026 paper:
> Cristian-Teodor Neamtu, Serban Mihalache, Stefan Smeu, Dan Oneata, Horia Cucu, Dragos Burileanu, "Towards Zero-Shot Attribution of Synthetic Speech via Audio-Text Contrastive Retrieval", accepted at SLT 2026, Palermo, Italy.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](TODO)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)

## Abstract

Audio deepfake forensics is moving beyond a simple real-or-fake verdict toward \emph{attribution}: which system generated the audio clip? Most source-attribution methods cast this as closed-set classification, so they cannot name a generator that was absent from training, a gap that widens with every newly released text-to-speech (TTS) system. We instead frame attribution as \textbf{cross-modal retrieval}: each generator is described in natural language, and a clip is attributed by retrieving the description closest to it in a shared audio--text embedding space. Adding a new system then takes nothing more than writing its description, with no retraining and no new classifier head. Our model couples a frozen Wav2Vec2-BERT audio encoder with a frozen E5 text encoder and aligns them through small trainable projection heads, using a contrastive objective that combines a cross-modal supervised-contrastive loss with intra-modal terms. We evaluate on MLAAD~v9 (140 TTS systems, 51 languages) under 10-fold leave-systems-out cross-validation. For generators it has never encountered before, the model reaches a model-level mean reciprocal rank (MRR) of $58.4\%$. Even when the correct model is not identified, the audio clip is matched to systems that share the true generator's vocoder, acoustic model, or architecture. Because the same embedding space also answers natural language attribute queries, one set of descriptions covers both open-set attribution and attribute-level forensic profiling.
