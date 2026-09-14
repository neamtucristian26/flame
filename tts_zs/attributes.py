from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TypedDict

from tts_zs.config import DATA_DIR, FACTSHEETS, SEP

VOCODER_RULES: list[tuple[str, list[str]]] = [
    ("encodec", ["encodec"]),
    ("snac", ["snac"]),
    ("mimi", ["mimi"]),
    ("dac", ["descript audio codec", " dac "]),
    ("xcodec2", ["xcodec2"]),
    ("wavtokenizer", ["wavtokenizer"]),
    ("higgscodec", ["higgscodec"]),
    ("neucodec", ["neucodec"]),
    ("nanocodec", ["nanocodec"]),
    ("bicodec", ["bicodec"]),
    ("ffgan", ["firefly-gan", "ffgan"]),
    ("bigvgan", ["bigvgan"]),
    ("vocos", ["vocos"]),
    ("locdit", ["locdit"]),
    ("wavevae", ["wavevae"]),
    (
        "hifigan",
        [
            "hifigan",
            "hifi-gan",
            "hift-gan",
            "hifinet",
            "hifi-net",
            "univnet",
            "gan-based waveform",
            "gan-based neural vocoder",
            "adversarial neural vocoder",
            "neural waveform decoder",
        ],
    ),
    ("istft", ["istft", "i-stft"]),
    ("griffin_lim", ["griffin-lim", "griffin lim", "classical dsp"]),
    ("proprietary", ["proprietary"]),
]

def classify_vocoder(text: str) -> str:
    if not text:
        return "unknown"
    s = text.lower()
    for bucket, patterns in VOCODER_RULES:
        for p in patterns:
            if p in s:
                return bucket
    return "unknown"

SPEAKER_RULES: list[tuple[str, list[str]]] = [
    (
        "zero_shot",
        ["zero-shot clone", "zero-shot cross-lingual", "reference audio embedding"],
    ),
    ("voice_conv", ["voice conversion"]),
    ("description", ["description-driven", "description driven"]),
    ("multi", ["multi"]),
    ("single", ["single", "acoustic-model-dependent"]),
]

def classify_speaker(text: str) -> str:
    if not text:
        return "unknown"
    s = text.lower()
    for bucket, patterns in SPEAKER_RULES:
        for p in patterns:
            if p in s:
                return bucket
    return "unknown"

_ARCH_TO_ACOUSTIC = {
    "vits": "vits",
    "glow": "glow",
    "tacotron": "tacotron",
    "nar_transformer": "nar_feedforward",
    "flow_match": "flow_matching",
    "diffusion": "diffusion",
    "codec_ar": "ar_codec_lm",
    "llm_tts": "ar_codec_lm",
    "tortoise_xtts": "ar_codec_lm",
    "multimodal": "ar_codec_lm",
    "proprietary": "proprietary",
}

_ACOUSTIC_OVERRIDES = {
    "kokoro": "styletts",
    "Tsukasa_Speech": "styletts",
    "microsoft_speecht5_tts": "encoder_decoder",
    "parler_tts_large_v1": "encoder_decoder",
    "parler_tts_mini_v0.1": "encoder_decoder",
    "parler_tts_mini_v1": "encoder_decoder",
    "Edge-TTS": "encoder_decoder",
}

def classify_acoustic_model(model_name: str, arch_family: str) -> str:
    if model_name in _ACOUSTIC_OVERRIDES:
        return _ACOUSTIC_OVERRIDES[model_name]
    return _ARCH_TO_ACOUSTIC.get(arch_family, "unknown")

_LANG_FAMILY: dict[str, str] | None = None

def _load_lang_family() -> dict[str, str]:
    global _LANG_FAMILY
    if _LANG_FAMILY is not None:
        return _LANG_FAMILY
    src_path = (
        Path(__file__).resolve().parent.parent / "scripts" / "build_lang_labels.py"
    )
    spec = importlib.util.spec_from_file_location("_blbl_attrs", src_path)
    if spec is None or spec.loader is None:
        _LANG_FAMILY = {}
        return _LANG_FAMILY
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    props: dict = getattr(mod, "LANG_PROPS", {})
    _LANG_FAMILY = {k: v.get("family", "unknown") for k, v in props.items()}
    return _LANG_FAMILY

def get_language_family(lang_code: str) -> str:
    return _load_lang_family().get(lang_code, "unknown")

class PairAttr(TypedDict):
    model: str
    lang: str
    arch_family: str
    acoustic_model: str
    vocoder: str
    speaker_type: str
    lang_family: str

def build_pair_attributes(
    all_pair_keys: list[str],
    factsheets: dict | None = None,
    arch_labels: dict | None = None,
) -> list[PairAttr]:
    if factsheets is None:
        factsheets = json.load(open(FACTSHEETS))
    if arch_labels is None:
        arch_labels = json.load(open(DATA_DIR / "arch_labels.json"))

    out: list[PairAttr] = []
    for key in all_pair_keys:
        model, lang = key.split(SEP)
        fs = factsheets.get(model, {})
        arch_family = arch_labels.get(model, "unknown")
        out.append(
            {
                "model": model,
                "lang": lang,
                "arch_family": arch_family,
                "acoustic_model": classify_acoustic_model(model, arch_family),
                "vocoder": classify_vocoder(fs.get("vocoder_or_codec", "")),
                "speaker_type": classify_speaker(fs.get("speaker_config", "")),
                "lang_family": get_language_family(lang),
            }
        )
    return out
