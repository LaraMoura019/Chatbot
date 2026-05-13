from faster_whisper import WhisperModel
import os
from tqdm import tqdm

def transcricao(ficheiro_audio, ficheiro_txt, model, language="en", com_timestamps=False):
    """
    Transcreve um ficheiro de áudio para texto usando Faster Whisper.
    
    Args:
        ficheiro_audio: caminho para o ficheiro de áudio
        ficheiro_txt: caminho para guardar a transcrição
        model: instância do WhisperModel já carregada
        language: idioma do áudio ("pt" para português, "en" para inglês)
        com_timestamps: se True, guarda o texto com os tempos (ex: [0.00s - 2.50s] Olá)
    
    Returns:
        texto_completo: texto transcrito
    """
    
    # Verificar se o ficheiro existe
    if not os.path.exists(ficheiro_audio):
        raise FileNotFoundError(f"Ficheiro de áudio não encontrado: {ficheiro_audio}")

    print(f"A iniciar transcrição de: {os.path.basename(ficheiro_audio)}")
    
    # Fazer a transcrição
    segmentos, info = model.transcribe(
        ficheiro_audio, 
        beam_size=5, 
        language=language,     
        condition_on_previous_text=True,  
        vad_filter=True,              
        vad_parameters=dict(
            min_silence_duration_ms=500
        )
    )
    
    texto_completo = ""
    linhas_ficheiro = []

    # Cria a barra de progresso baseada na duração total do áudio
    with tqdm(total=info.duration, unit="s", desc="Progresso", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} seg") as pbar:
        for segmento in segmentos:
            # Formatar com ou sem timestamps baseando-se na escolha do utilizador
            if com_timestamps:
                linha = f"[{segmento.start:.2f}s -> {segmento.end:.2f}s] {segmento.text.strip()}"
                texto_completo += linha + "\n"
                linhas_ficheiro.append(linha + "\n")
            else:
                texto_completo += segmento.text + " "
                linhas_ficheiro.append(segmento.text + " ")
            
            # Atualização mais segura do tqdm
            incremento = min(segmento.end - pbar.n, info.duration - pbar.n)
            if incremento > 0:
                pbar.update(incremento)

    texto_completo = texto_completo.strip()
    
    # Guardar transcrição em ficheiro
    with open(ficheiro_txt, "w", encoding="utf-8") as ficheiro:
        if com_timestamps:
            ficheiro.writelines(linhas_ficheiro)
        else:
            ficheiro.write("".join(linhas_ficheiro).strip())

    print(f"Transcrição concluída e guardada em: {ficheiro_txt}\n")
    
    return texto_completo