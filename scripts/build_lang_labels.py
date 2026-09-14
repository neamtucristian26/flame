import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_zs.config import AUDIO_ROOT, LABELS_LANG, SEP

LANG_PROPS: dict[str, dict] = {
    "am": {
        "name": "Amharic",
        "family": "Afro-Asiatic (Semitic)",
        "script": "Ethiopic",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Amharic TTS must handle the Ethiopic (Ge'ez) script, geminate consonants, "
            "and ejective stops; vocoders often struggle with the dense consonant inventory."
        ),
    },
    "ar": {
        "name": "Arabic",
        "family": "Afro-Asiatic (Semitic)",
        "script": "Arabic",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Arabic TTS faces pharyngeal and emphatic consonants, right-to-left script, "
            "and absent short-vowel diacritics in standard text input; prosody varies widely by dialect."
        ),
    },
    "bg": {
        "name": "Bulgarian",
        "family": "Indo-European (Slavic)",
        "script": "Cyrillic",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Bulgarian TTS handles unpredictable stress and Cyrillic orthography; "
            "vowel reduction in unstressed syllables is a key prosodic challenge."
        ),
    },
    "bn": {
        "name": "Bengali",
        "family": "Indo-European (Indo-Aryan)",
        "script": "Bengali",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Bengali TTS must navigate a distinctive vowel inventory and inherent-vowel deletion rules; "
            "retroflex consonant articulation is a common artifact location."
        ),
    },
    "cs": {
        "name": "Czech",
        "family": "Indo-European (Slavic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Czech TTS features fixed word-initial stress and dense consonant clusters; "
            "long vowel vs short vowel contrasts are phonemically important."
        ),
    },
    "da": {
        "name": "Danish",
        "family": "Indo-European (Germanic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Danish TTS must reproduce the stød (laryngealisation) and characteristic soft consonant "
            "lenition; final consonant reduction produces heavily reduced, elided speech."
        ),
    },
    "de": {
        "name": "German",
        "family": "Indo-European (Germanic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "German TTS handles long compound words, final obstruent devoicing, and umlaut vowels; "
            "plosive aspiration and precise vowel length are discriminative acoustic cues."
        ),
    },
    "el": {
        "name": "Greek",
        "family": "Indo-European (Hellenic)",
        "script": "Greek",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Greek TTS uses a single monotonic stress marker; pitch differences are length-based, "
            "and geminate consonants in some dialects add synthesis complexity."
        ),
    },
    "en": {
        "name": "English",
        "family": "Indo-European (Germanic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "English TTS must handle unstressed vowel reduction, heterogeneous prosodic phrasing, "
            "and dialectal variation; artifacts most frequently appear in fricatives and phrase boundaries."
        ),
    },
    "es": {
        "name": "Spanish",
        "family": "Indo-European (Romance)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Spanish TTS benefits from consistent orthography and syllable-timed rhythm; "
            "rhotic articulation (trill vs tap) and inter-word liaison are key synthesis challenges."
        ),
    },
    "et": {
        "name": "Estonian",
        "family": "Uralic (Finnic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Estonian TTS must encode a three-way phonemic quantity distinction (short/long/overlong) "
            "that profoundly affects both segmental and prosodic realisation."
        ),
    },
    "fa": {
        "name": "Persian",
        "family": "Indo-European (Iranian)",
        "script": "Arabic-Persian",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Persian (Farsi) TTS uses an Arabic-derived script with missing short vowels in standard text; "
            "epenthetic vowels and informal speech reductions are common challenges."
        ),
    },
    "fi": {
        "name": "Finnish",
        "family": "Uralic (Finnic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Finnish TTS handles vowel harmony, geminate consonants, and long vs short phonemic distinctions; "
            "the agglutinative morphology produces very long words with complex prosodic contours."
        ),
    },
    "fr": {
        "name": "French",
        "family": "Indo-European (Romance)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "French TTS must handle liaison, elision, and nasal vowels; phrase-final lengthening replaces "
            "lexical stress as the primary prosodic cue, creating characteristic flat intonation."
        ),
    },
    "ga": {
        "name": "Irish",
        "family": "Indo-European (Celtic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Irish TTS faces initial consonant mutations (lenition, eclipsis) that change word-initial sounds, "
            "and a complex vowel system with broad/slender consonant pairs."
        ),
    },
    "ha": {
        "name": "Hausa",
        "family": "Afro-Asiatic (Chadic)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": True,
        "tts_note": (
            "Hausa is a tonal language with high and low tones; ATR vowel harmony and long consonants "
            "are additional features that TTS systems must reproduce faithfully."
        ),
    },
    "he": {
        "name": "Hebrew",
        "family": "Afro-Asiatic (Semitic)",
        "script": "Hebrew",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Hebrew TTS reads right-to-left script typically without vowel diacritics; "
            "pharyngeal consonants and final-stress tendency are distinctive acoustic features."
        ),
    },
    "hi": {
        "name": "Hindi",
        "family": "Indo-European (Indo-Aryan)",
        "script": "Devanagari",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Hindi TTS must reproduce retroflex consonants, aspirated stop contrasts, and the inherent "
            "vowel suppression (schwa deletion) that shapes natural spoken Hindi rhythm."
        ),
    },
    "hr": {
        "name": "Croatian",
        "family": "Indo-European (Slavic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Croatian TTS encodes a neo-Štokavian pitch-accent system with four distinct accent types; "
            "long and short vowel quantity contrasts are both prosodically and phonemically relevant."
        ),
    },
    "hu": {
        "name": "Hungarian",
        "family": "Uralic (Ugric)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Hungarian TTS handles vowel harmony, fixed word-initial stress, and agglutinative suffixes "
            "that produce very long phonological words with complex internal prosodic structure."
        ),
    },
    "ig": {
        "name": "Igbo",
        "family": "Niger-Congo (Volta-Niger)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": True,
        "tts_note": (
            "Igbo is a tonal language with two level tones and downstep; incorrect tone rendering "
            "creates lexical and grammatical ambiguity, making tonal accuracy critical."
        ),
    },
    "it": {
        "name": "Italian",
        "family": "Indo-European (Romance)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Italian TTS must handle geminate consonants and open vs closed vowel contrasts; "
            "smooth syllable-timed rhythm and penultimate stress make Italian prosody relatively "
            "straightforward for neural TTS."
        ),
    },
    "ja": {
        "name": "Japanese",
        "family": "Japonic",
        "script": "Japanese (hiragana/katakana/kanji)",
        "rhythm": "mora-timed",
        "tonal": False,
        "tts_note": (
            "Japanese TTS uses mora-timing and a pitch-accent system where accentual patterns "
            "distinguish word meaning; the mixed script (hiragana/katakana/kanji) requires robust "
            "text normalisation before synthesis."
        ),
    },
    "jv": {
        "name": "Javanese",
        "family": "Austronesian (Malayo-Polynesian)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Javanese TTS must navigate register diglossia (Ngoko vs Krama speech levels) "
            "and a rich vowel inventory including the mid-central unrounded vowel absent in related languages."
        ),
    },
    "kn": {
        "name": "Kannada",
        "family": "Dravidian",
        "script": "Kannada",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Kannada TTS must handle a full Dravidian retroflex series, aspirated stops, and the "
            "Kannada abugida script; three grammatical genders affect agreement morphology in speech."
        ),
    },
    "ko": {
        "name": "Korean",
        "family": "Koreanic",
        "script": "Hangul",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Korean TTS handles the three-way stop contrast (lax/aspirated/tense), complex syllable-final "
            "consonant restrictions, and phonological processes like nasalisation and lenition at boundaries."
        ),
    },
    "lt": {
        "name": "Lithuanian",
        "family": "Indo-European (Baltic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Lithuanian TTS encodes a pitch-accent system with acute and circumflex contours; "
            "it has the most archaic phonology among living Indo-European languages, including two vowel quantities."
        ),
    },
    "lv": {
        "name": "Latvian",
        "family": "Indo-European (Baltic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Latvian TTS must reproduce three intonation types (falling, level, and broken/creaky), "
            "fixed initial stress, and a two-vowel-length system."
        ),
    },
    "ml": {
        "name": "Malayalam",
        "family": "Dravidian",
        "script": "Malayalam",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Malayalam TTS handles one of the largest phoneme inventories among world languages, "
            "including dental-retroflex-alveolar distinctions and complex initial clusters."
        ),
    },
    "mr": {
        "name": "Marathi",
        "family": "Indo-European (Indo-Aryan)",
        "script": "Devanagari",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Marathi TTS shares challenges with Hindi (retroflex consonants, schwa deletion) "
            "but additionally includes dental-lateral distinctions and a distinct tonal register "
            "for some dialects."
        ),
    },
    "ms": {
        "name": "Malay",
        "family": "Austronesian (Malayo-Polynesian)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Malay TTS has relatively simple phonology with consistent orthography; "
            "the main challenges are borrowed words from Arabic, Sanskrit, and English with variable "
            "pronunciation rules."
        ),
    },
    "mt": {
        "name": "Maltese",
        "family": "Afro-Asiatic (Semitic, with Romance/English influence)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Maltese TTS handles a uniquely mixed lexicon (Arabic, Sicilian, English) with pharyngeal "
            "consonants from its Semitic base alongside Romance-style prosody."
        ),
    },
    "nl": {
        "name": "Dutch",
        "family": "Indo-European (Germanic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Dutch TTS must handle velar fricatives, diphthong variation, and G-sound lenition; "
            "regional accent variation (Belgium vs Netherlands) is a substantial challenge."
        ),
    },
    "pl": {
        "name": "Polish",
        "family": "Indo-European (Slavic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Polish TTS handles dense consonant clusters, a large sibilant series, and fixed "
            "penultimate stress; nasal vowels and palatalisation are distinctive acoustic cues."
        ),
    },
    "pt": {
        "name": "Portuguese",
        "family": "Indo-European (Romance)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Portuguese TTS must manage heavy vowel reduction (especially in European Portuguese), "
            "nasal vowels, and significant prosodic differences between Brazilian and European variants."
        ),
    },
    "ro": {
        "name": "Romanian",
        "family": "Indo-European (Romance)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Romanian TTS handles Slavic-influenced phonology within a Romance framework; "
            "the schwa-like central vowel and definite article suffixation are distinctive features."
        ),
    },
    "ru": {
        "name": "Russian",
        "family": "Indo-European (Slavic)",
        "script": "Cyrillic",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Russian TTS must reproduce strong vowel reduction (unstressed /o/ → [ə/ɐ]), "
            "pervasive palatalisation contrasts, and consonant clusters; stress is unpredictable "
            "from orthography."
        ),
    },
    "si": {
        "name": "Sinhala",
        "family": "Indo-European (Indo-Aryan)",
        "script": "Sinhala",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Sinhala TTS navigates a diglossia between literary and colloquial registers, "
            "pre-nasalised stops, and a vowel inventory with both short and long counterparts."
        ),
    },
    "sk": {
        "name": "Slovak",
        "family": "Indo-European (Slavic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Slovak TTS features fixed word-initial stress and phonemic vowel quantity; "
            "the rhythmic law that shortens consecutive long syllables is a unique prosodic constraint."
        ),
    },
    "sl": {
        "name": "Slovenian",
        "family": "Indo-European (Slavic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Slovenian TTS encodes a pitch-accent system with long rising and long falling "
            "distinctions; vowel quantity and place of stress interact to create complex prosodic patterns."
        ),
    },
    "sv": {
        "name": "Swedish",
        "family": "Indo-European (Germanic)",
        "script": "Latin",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Swedish TTS must capture the word-level pitch accent (acute vs grave) that distinguishes "
            "minimal pairs; the characteristic sing-song intonation is a highly salient perceptual cue."
        ),
    },
    "sw": {
        "name": "Swahili",
        "family": "Niger-Congo (Bantu)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Swahili TTS benefits from consistent CV syllable structure and regular penultimate stress; "
            "noun-class agreement morphology creates long agglutinative verb forms."
        ),
    },
    "ta": {
        "name": "Tamil",
        "family": "Dravidian",
        "script": "Tamil",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Tamil TTS must handle a classical literary register distinct from spoken Tamil, "
            "dental-retroflex-alveolar contrasts, and long/short vowel quantity distinctions."
        ),
    },
    "th": {
        "name": "Thai",
        "family": "Tai-Kadai",
        "script": "Thai",
        "rhythm": "syllable-timed",
        "tonal": True,
        "tts_note": (
            "Thai is a tonal language with five lexical tones encoded in the Thai script via tone marks; "
            "incorrect tone realisation fundamentally alters word meaning, making tonal accuracy paramount."
        ),
    },
    "tk": {
        "name": "Turkmen",
        "family": "Turkic (Oghuz)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Turkmen TTS handles vowel harmony and agglutinative suffixation typical of Turkic languages; "
            "a relatively rare TTS target with limited training data available."
        ),
    },
    "tr": {
        "name": "Turkish",
        "family": "Turkic (Oghuz)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Turkish TTS manages vowel harmony, uniform agglutinative suffixation, and the distinction "
            "between front/back and rounded/unrounded vowel pairs across long words."
        ),
    },
    "uk": {
        "name": "Ukrainian",
        "family": "Indo-European (Slavic)",
        "script": "Cyrillic",
        "rhythm": "stress-timed",
        "tonal": False,
        "tts_note": (
            "Ukrainian TTS handles Cyrillic orthography with less extreme vowel reduction than Russian; "
            "the /i/ vs /ɪ/ distinction and palatalisation contrasts are key discriminative features."
        ),
    },
    "ur": {
        "name": "Urdu",
        "family": "Indo-European (Indo-Aryan)",
        "script": "Arabic-Nastaliq",
        "rhythm": "syllable-timed",
        "tonal": False,
        "tts_note": (
            "Urdu TTS reads Nastaliq calligraphic script (right-to-left) while sharing Hindi phonology; "
            "retroflex consonants and aspirated stop contrasts are the primary synthesis challenges."
        ),
    },
    "vi": {
        "name": "Vietnamese",
        "family": "Austroasiatic (Mon-Khmer)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": True,
        "tts_note": (
            "Vietnamese is a tonal language with six lexical tones marked by diacritics in its Latin-based "
            "script; tonal contour accuracy is essential for intelligibility."
        ),
    },
    "yo": {
        "name": "Yoruba",
        "family": "Niger-Congo (Volta-Niger)",
        "script": "Latin",
        "rhythm": "syllable-timed",
        "tonal": True,
        "tts_note": (
            "Yoruba is a tonal language with three level tones; nasalised vowels and downstep are "
            "additional prosodic features that TTS must capture accurately."
        ),
    },
    "zh-cn": {
        "name": "Chinese (Mandarin)",
        "family": "Sino-Tibetan",
        "script": "Chinese characters",
        "rhythm": "mora-timed",
        "tonal": True,
        "tts_note": (
            "Mandarin TTS must render four lexical tones plus a neutral tone, with tone sandhi in "
            "context (e.g., two 3rd-tone syllables); the logographic script requires full text-to-pinyin "
            "normalisation before synthesis."
        ),
    },
}

def lang_tag(code: str) -> str:
    props = LANG_PROPS.get(code)
    if props is None:
        return f"[Language: {code.upper()}]"
    tonal_str = "tonal" if props["tonal"] else "non-tonal"
    return (
        f"[Language: {props['name']} | {props['family']} | "
        f"{props['script']}-script | {props['rhythm']} | {tonal_str}]"
    )

def tts_note(code: str) -> str:
    props = LANG_PROPS.get(code)
    return props["tts_note"] if props else ""

def discover_pairs(audio_root: Path) -> dict[str, list[str]]:
    by_model: dict[str, set] = defaultdict(set)
    for lang_dir in sorted(audio_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for model_dir in sorted(lang_dir.iterdir()):
            if model_dir.is_dir() and any(model_dir.glob("*.pt")):
                by_model[model_dir.name].add(lang)
    return {m: sorted(langs) for m, langs in sorted(by_model.items())}

def build(factsheet_path: Path, audio_root: Path, out_path: Path):
    with open(factsheet_path, encoding="utf-8") as f:
        factsheet: dict[str, list[str]] = json.load(f)

    model_to_langs = discover_pairs(audio_root)

    missing = set(model_to_langs) - set(factsheet)
    extra = set(factsheet) - set(model_to_langs)
    if missing:
        raise RuntimeError(f"Models in audio_root but not in factsheet: {missing}")
    if extra:
        print(f"[warn] {len(extra)} models in factsheet but no audio found; skipping.")

    labels: dict[str, list[str]] = {}

    for model in sorted(model_to_langs):
        langs = model_to_langs[model]
        base_variants: list[str] = factsheet[model]
        for lang in langs:
            key = f"{model}{SEP}{lang}"
            tag  = lang_tag(lang)
            note = tts_note(lang)
            if note:
                conditioned = [f"{tag} {v} {note}" for v in base_variants]
            else:
                conditioned = [f"{tag} {v}" for v in base_variants]
            labels[key] = conditioned

    n_pairs  = len(labels)
    n_models = len(model_to_langs)
    print(f"Models: {n_models}  |  (model, lang) pairs: {n_pairs}")

    all_langs = {k.split(SEP)[1] for k in labels}
    unknown = all_langs - set(LANG_PROPS)
    if unknown:
        print(f"[warn] {len(unknown)} language codes have no props entry: {sorted(unknown)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--factsheet", type=Path, default=LABELS_LANG)
    p.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    p.add_argument("--out", type=Path, default=LABELS_LANG)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build(args.factsheet, args.audio_root, args.out)
