from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

# Variável global para guardar o  motor de busca
_retriever = None

def formatar_contexto(docs):
    """
    Esta função pega nos pedaços de texto (chunks) que o retriever encontrou
    e junta-os todos num único texto limpo, separado por parágrafos.
    Isto facilita a leitura.
    """
    textos = []
    for doc in docs:
        textos.append(doc.page_content)
    
    # Junta os textos todos com duas quebras de linha entre eles
    return "\n\n".join(textos)


def inicializar_ferramentas(retriever):
    global _retriever
    _retriever = retriever

    @tool
    def explicar_diagnostico(pergunta):
        """
        Usa esta ferramenta para explicar diagnósticos, doenças, causas de problemas 
        e a razão dos sintomas do paciente. 
        """
        # Adicionar palavras-chave genéricas para ajudar na pesquisa
        docs = _retriever.invoke(pergunta + " diagnóstico explicação sintomas causa")
        return formatar_contexto(docs)

    @tool
    def pesquisar_tratamentos(pergunta: str) -> str:
        """
        Usa esta ferramenta para perguntas sobre tratamentos, medicamentos, 
        comprimidos, dosagens, receitas médicas, efeitos secundários ou exames.
        """
        docs = _retriever.invoke(pergunta + " medicação tratamento dose exames receita")
        return formatar_contexto(docs)

    @tool
    def conselhos_estilo_vida(pergunta: str) -> str:
        """
        Usa esta ferramenta para dúvidas sobre o dia a dia: alimentação, 
        exercício físico, sono, postura, stress e hábitos de vida.
        """
        docs = _retriever.invoke(pergunta + " hábitos alimentação exercício recomendações")
        return formatar_contexto(docs)

    @tool
    def proximos_passos_e_alertas(pergunta: str) -> str:
        """
        Usa esta ferramenta para saber quando o paciente deve voltar ao médico, 
        quais os próximos passos, ou quais os sinais de perigo (urgência).
        """
        docs = _retriever.invoke(pergunta + " próxima consulta emergência urgência perigo atenção")
        return formatar_contexto(docs)

    return [explicar_diagnostico, pesquisar_tratamentos, conselhos_estilo_vida, proximos_passos_e_alertas]


def criar_agente(retriever):
    ferramentas = inicializar_ferramentas(retriever)
    
    # Inicializamos o cérebro (LLM)
    llm = ChatOllama(model="gemma3:12b", temperature=0) 
    
    # O Prompt é a "Personalidade" do nosso bot e as suas regras
    prompt = ChatPromptTemplate.from_messages([
        # Regra base
        ("system", """És um assistente médico virtual generalista em Portugal.
        O teu trabalho é ajudar os pacientes a compreender a consulta médica que acabaram de ter, independentemente da especialidade.

        REGRA 1: Baseia-te EXCLUSIVAMENTE nas ferramentas fornecidas (transcrição da consulta e manuais).
        REGRA 2: Se o paciente perguntar sobre algo que não foi falado na consulta ou que não está nos teus manuais, diz honestamente: "Essa informação não foi discutida na sua consulta, recomendo que contacte o seu médico."
        REGRA 3: Se detetares alguma situação de emergência, aconselha imediatamente o contacto com o 112 ou a ida às urgências.
        REGRA 4: Mantém um tom acolhedor e nunca tentes substituir o médico humano. Responde em Português de Portugal."""),
        
        # guardamos a conversa passada.
        MessagesPlaceholder(variable_name="chat_history"),
        
        # A nova pergunta do utilizador
        ("human", "{input}"),
        
        # Espaço em branco onde o agente "pensa" e usa as ferramentas
        ("placeholder", "{agent_scratchpad}"), 
    ])
    
    # Criamos o agente juntando o cérebro (llm), os olhos (ferramentas) e a personalidade (prompt)
    agente = create_tool_calling_agent(llm, ferramentas, prompt)
    
    # O Executor é quem põe o agente a trabalhar na prática
    executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        verbose=True, # Ligado para vermos os passos no ecrã (ótimo para aprenderes!)
        handle_parsing_errors=True # Proteção caso a IA se engane no formato
    )
    
    return executor


# ─────────────────────────────────────────────
# 4. INICIAR A CONVERSA (O Loop do Chatbot)
# ─────────────────────────────────────────────
def iniciar_chat(executor):
    print("\nOlá! Sou o teu Assistente de Saúde. Como te posso ajudar hoje?")
    print("(Escreve 'sair' para terminar a conversa)\n")
    
    # Esta lista vai guardar tudo o que vocês disserem (A Memória!)
    historico_conversa = []
    
    while True:
        pergunta = input("Tu: ")
        
        if pergunta.lower() == 'sair':
            print("As melhoras! Até à próxima.")
            break
            
        try:
            # Mandamos a pergunta E a memória para o Agente
            resposta = executor.invoke({
                "input": pergunta,
                "chat_history": historico_conversa
            })
            
            texto_da_resposta = resposta["output"]
            print(f"\nAssistente: {texto_da_resposta}\n")
            
            # Depois de responder, guardamos esta troca de mensagens na memória
            historico_conversa.extend([
                HumanMessage(content=pergunta),
                AIMessage(content=texto_da_resposta)
            ])
            
        except Exception as e:
            print(f"Ups, houve um erro: {e}")

# ─────────────────────────────────────────────
# COMO CORRER ISTO TUDO NO FINAL:
# ─────────────────────────────────────────────
from criar_rag import inicializar_rag
from transcrever import transcricao
retriever, vs = inicializar_rag("manuais_medicos",transcricao("./audios/diabetes.mp3","diabetes.txt"),"diabetes.mp3")
executor = criar_agente(retriever)
iniciar_chat(executor)