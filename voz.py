from faster_whisper import WhisperModel
from TTS.api import TTS
import tempfile
import os
import torch

# ── Whisper (reutiliza o mesmo modelo da transcrição de consultas) ──
_whisper_model = None

def obter_modelo_whisper():
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel("large-v3", device=device, compute_type=compute)
    return _whisper_model

def transcrever_pergunta(audio_bytes: bytes) -> str:
    """
    Recebe bytes de áudio gravado no browser (wav) e devolve o texto transcrito.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = obter_modelo_whisper()
        segmentos, _ = model.transcribe(
            tmp_path,
            beam_size=5,
            language="pt",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        texto = " ".join(s.text for s in segmentos).strip()
    finally:
        os.remove(tmp_path)

    return texto


# ── Coqui TTS ──
_tts_model = None

def obter_modelo_tts():
    global _tts_model
    if _tts_model is None:
        # Modelo multilingue com boa qualidade em PT
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _tts_model

def sintetizar_resposta(texto: str) -> bytes:
    """
    Converte texto em áudio WAV e devolve os bytes prontos para st.audio().
    """
    tts = obter_modelo_tts()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tts.tts_to_file(
            text=texto,
            file_path=tmp_path,
            language="pt",
            speaker="Ana Florence",  # voz feminina PT disponível no XTTS v2
        )
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.remove(tmp_path)

    return audio_bytes