from faster_whisper import WhisperModel
import edge_tts
import asyncio
import tempfile
import os


# ── Whisper (reutiliza o mesmo modelo da transcrição de consultas) ──
_whisper_model = None

def obter_modelo_whisper():
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" 
        compute = "float16"
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


# ── Edge TTS ──
# Voz feminina PT-PT da Microsoft — boa qualidade, não requer GPU
VOZ_PT = "pt-PT-RaquelNeural"

async def _sintetizar_async(texto: str, caminho: str):
    comunicador = edge_tts.Communicate(texto, VOZ_PT)
    await comunicador.save(caminho)

def sintetizar_resposta(texto: str) -> bytes:
    """
    Converte texto em áudio MP3 via Edge TTS e devolve os bytes prontos para st.audio().
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        asyncio.run(_sintetizar_async(texto, tmp_path))
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.remove(tmp_path)

    return audio_bytes