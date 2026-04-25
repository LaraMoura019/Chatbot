from faster_whisper import WhisperModel

# Carregar o modelo 
modelo_tamanho = "large-v3"
model = WhisperModel(modelo_tamanho, device="cpu", compute_type="int8")

# Caminho para o áudio
ficheiro_audio = "type_2_diabetes.mp3"
ficheiro_txt="transcricao_diabetes.txt"

def transcricao(ficheiro_audio,model, ficheiro_txt):
    print("A iniciar transcrição...")
    # Fazer a transcrição
    segmentos, info = model.transcribe(
        ficheiro_audio, 
        beam_size=5, 
        language="en", 
        condition_on_previous_text=False
    )

    # Mostrar o resultado
    texto_completo = ""
    for segmento in segmentos:
        linha = f"[{segmento.start:.2f}s -> {segmento.end:.2f}s] {segmento.text}"
        print(linha)
        texto_completo += segmento.text + " "
    print("A transcrição foi realizada com sucesso!")

    with open(ficheiro_txt, "w", encoding="utf-8") as ficheiro:
        ficheiro.write(texto_completo)
        
    return texto_completo

transcricao(ficheiro_audio, model, ficheiro_txt)