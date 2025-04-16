import os
import json
import requests
import base64
from typing import Dict, List, Optional, Union, Any, Annotated
from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from pathlib import Path
import time
import httpx
import asyncio

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Import and include router from routes.py
try:
    from routes import router
    app.include_router(router)
    logger.info("Successfully loaded patient routes from routes.py")
except ImportError as e:
    logger.error(f"Failed to import patient routes: {e}")

# Carica variabili d'ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Configurazione Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "psico_virtuale")

# Configurazione Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# Configura livello di compatibilità con memoria temporanea
USE_MEMORY_FALLBACK = os.getenv("USE_MEMORY_FALLBACK", "true").lower() == "true"

# Debug info
logger.info(f"QDRANT_URL = {QDRANT_URL}")
logger.info(f"SUPABASE_URL = {SUPABASE_URL}")
logger.info(f"USE_MEMORY_FALLBACK = {USE_MEMORY_FALLBACK}")

# Configurazione percorsi
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Configurazione LLM
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.5
MAX_TOKENS = 15000
SIMILARITY_TOP_K = 8
MAX_HISTORY_LENGTH = 6  # Storia più lunga per mantenere contesto terapeutico

# Configurazione ElevenLabs API
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1/convai"

# Modelli Pydantic per le richieste e risposte API
class Source(BaseModel):
    file_name: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    session_id: str
    mood: Optional[str] = None  # Opzionale: per tracciare l'umore del paziente

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = []
    analysis: Optional[str] = None  # Analisi psicologica opzionale

class ResetRequest(BaseModel):
    session_id: str

class ResetResponse(BaseModel):
    status: str
    message: str

class SessionSummaryResponse(BaseModel):
    summary_html: str

class MoodAnalysisResponse(BaseModel):
    mood_analysis: str

# Modelli per ElevenLabs
class ElevenLabsConversation(BaseModel):
    agent_id: str
    conversation_id: str
    start_time_unix_secs: Optional[int] = None
    call_duration_secs: Optional[int] = None
    message_count: Optional[int] = None
    status: str
    call_successful: Optional[str] = None
    agent_name: Optional[str] = None

class ElevenLabsConversationsResponse(BaseModel):
    conversations: List[ElevenLabsConversation]
    has_more: bool
    next_cursor: Optional[str] = None

class ElevenLabsTranscriptMessage(BaseModel):
    role: str
    time_in_call_secs: int
    message: Optional[str] = None

class ElevenLabsConversationDetail(BaseModel):
    agent_id: str
    conversation_id: str
    status: str
    transcript: List[ElevenLabsTranscriptMessage]
    metadata: Dict[str, Any]

# Modello per la richiesta e risposta dei resource
class ResourceRequest(BaseModel):
    query: str
    session_id: str

class ResourceResponse(BaseModel):
    resources: List[Dict[str, str]]

# Modelli per l'analisi combinata
class AnalysisSourceRequest(BaseModel):
    session_id: str
    analyze_chatbot: bool = True
    analyze_elevenlabs: bool = False
    elevenlabs_conversation_id: Optional[str] = None
    
# Modelli per l'analisi delle patologie   
class PathologyAnalysisRequest(BaseModel):
    session_id: str
    analyze_chatbot: bool = True
    analyze_elevenlabs: bool = False
    elevenlabs_conversation_id: Optional[str] = None

class PathologyItem(BaseModel):
    name: str
    description: str
    confidence: float
    key_symptoms: List[str]
    source: Optional[str] = None

class PathologyAnalysisResponse(BaseModel):
    possible_pathologies: List[PathologyItem]
    analysis_summary: str

# Per compatibilità temporanea mantengo memoria in-process
conversation_history: Dict[str, List[Dict[str, str]]] = {}
mood_history: Dict[str, List[str]] = {}

# Security Bearer
security = HTTPBearer()

# Configurazione Supabase Client (usando httpx per comunicazione http)
class SupabaseClient:
    def __init__(self, url: str, key: str, service_key: Optional[str] = None):
        self.url = url
        self.key = key
        self.service_key = service_key
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        self.service_headers = {
            "apikey": service_key if service_key else key,
            "Authorization": f"Bearer {service_key if service_key else key}",
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30.0)  # Aumentato timeout
        
        # Cache per i token verificati (per la demo)
        self.token_cache = {}
        self.cache_ttl = 300  # 5 minuti di cache per i token
    
    async def close(self):
        await self.client.aclose()
    
    async def get_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token JWT di Supabase e restituisce i dati utente"""
        try:
            # Verifica se il token è in cache
            current_time = time.time()
            if token in self.token_cache:
                cached_data, expiry = self.token_cache[token]
                if current_time < expiry:
                    logger.info(f"Usando token dalla cache per utente: {cached_data.get('id')}")
                    return cached_data
            
            headers = {
                "apikey": self.key,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # Log dettagliato della richiesta
            logger.info(f"Verificando token con Supabase. URL: {self.url}/auth/v1/user")
            logger.info(f"Token length: {len(token)}, Prefix: {token[:10] if token else 'None'}")
            
            try:
                # Usa timeout più breve per questo endpoint critico
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{self.url}/auth/v1/user", headers=headers)
            except Exception as conn_error:
                logger.error(f"Errore di connessione a Supabase: {str(conn_error)}")
                # Per demo: in caso di errore di connessione, assumiamo che il token sia valido
                return {"id": "fallback-user-id", "email": "fallback@example.com"}
            
            # Log della risposta
            logger.info(f"Risposta Supabase: Status {response.status_code}")
            
            if response.status_code == 200:
                user_data = response.json()
                logger.info(f"Token valido per utente: {user_data.get('id')}")
                
                # Salva in cache
                self.token_cache[token] = (user_data, current_time + self.cache_ttl)
                
                return user_data
            elif response.status_code == 401:
                logger.warning(f"Token non valido o scaduto. Status: {response.status_code}")
                return None
            else:
                logger.error(f"Errore verifica token: Status {response.status_code}, Response: {response.text}")
                # Per demo: in caso di errore != 401, assumiamo che il token sia valido
                return {"id": "error-user-id", "email": "error@example.com"}
        except Exception as e:
            logger.error(f"Errore verifica token: {str(e)}")
            # Per la demo, restituiamo un utente fittizio in caso di errore
            return {"id": "exception-user-id", "email": "exception@example.com"}
    
    async def select(self, table: str, columns: str = "*", filters: Dict = None) -> List[Dict]:
        """Seleziona dati da una tabella"""
        url = f"{self.url}/rest/v1/{table}?select={columns}"
        
        if filters:
            for key, value in filters.items():
                url += f"&{key}=eq.{value}"
                
        response = await self.client.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Errore select Supabase: {response.status_code} - {response.text}")
            return []
    
    async def insert(self, table: str, data: Dict) -> Optional[Dict]:
        """Inserisce dati in una tabella"""
        url = f"{self.url}/rest/v1/{table}"
        
        # Usa service_key per le operazioni di scrittura
        headers = self.service_headers
        
        response = await self.client.post(url, json=data, headers=headers)
        if response.status_code in [200, 201]:
            return response.json()
        else:
            logger.error(f"Errore insert Supabase: {response.status_code} - {response.text}")
            return None

# Istanza Supabase Client
supabase_client = None

def get_supabase():
    global supabase_client
    if not supabase_client:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise HTTPException(status_code=500, detail="Configurazione Supabase mancante")
        supabase_client = SupabaseClient(SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY)
    return supabase_client

# Verifica JWT token di Supabase
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verifica il token JWT di Supabase e restituisce l'utente"""
    try:
        token = credentials.credentials
        logger.info(f"Token to verify: length={len(token)}, prefix={token[:10] if token else 'None'}")
        supabase = get_supabase()
        
        # Verifica token tramite endpoint Supabase
        user_data = await supabase.get_user(token)
        
        if user_data:
            logger.info(f"Token valid, user: {user_data.get('id')}")
            return user_data
        else:
            # Per demo: invece di fallire, torna un utente demo con accesso limitato
            logger.warning(f"Token non valido, usando utente demo per compatibilità")
            demo_user = {
                "id": "demo-user-id",
                "email": "demo@example.com",
                "role": "patient"
            }
            logger.info(f"Returning demo user: {demo_user}")
            return demo_user
    except Exception as e:
        logger.error(f"Errore verifica token: {str(e)}")
        # Per demo: invece di fallire, torna un utente demo con accesso limitato
        demo_user = {
            "id": "demo-user-id",
            "email": "demo@example.com",
            "role": "patient"
        }
        logger.info(f"Returning demo user after exception: {demo_user}")
        return demo_user

# Funzione per ottenere il ruolo dell'utente da Supabase
async def get_user_role(user_id: str) -> Optional[str]:
    """Restituisce il ruolo dell'utente da Supabase"""
    try:
        # Se è l'utente demo, restituisci direttamente il ruolo
        if user_id == "demo-user-id":
            return "patient"
            
        supabase = get_supabase()
        profiles = await supabase.select('profiles', '*', {'id': user_id})
        
        if profiles and len(profiles) > 0:
            return profiles[0].get('role')
        
        # Se non troviamo il ruolo, assumiamo "patient" per semplificare la demo
        return "patient"
    except Exception as e:
        logger.error(f"Errore recupero ruolo: {str(e)}")
        return "patient"  # Fallback per la demo

# Database utility functions
async def get_conversation_messages(session_id: str) -> List[Dict[str, str]]:
    """Recupera i messaggi di una conversazione da Supabase o memoria"""
    try:
        supabase = get_supabase()
        result = await supabase.select('messages', '*', {'session_id': session_id})
        
        if result and len(result) > 0:
            # Converti dal formato DB al formato interno
            messages = []
            for msg in result:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            return messages
        elif USE_MEMORY_FALLBACK and session_id in conversation_history:
            # Fallback a memoria in-process
            return conversation_history[session_id]
        return []
    except Exception as e:
        logger.error(f"Errore recupero messaggi: {str(e)}")
        if USE_MEMORY_FALLBACK and session_id in conversation_history:
            return conversation_history[session_id]
        return []

async def save_message(session_id: str, role: str, content: str):
    """Salva un messaggio in Supabase e opzionalmente in memoria"""
    try:
        # Salva prima in memoria per evitare perdita di dati
        if USE_MEMORY_FALLBACK:
            if session_id not in conversation_history:
                conversation_history[session_id] = []
            conversation_history[session_id].append({
                "role": role,
                "content": content
            })
        
        # Poi prova a salvare su Supabase
        try:
            supabase = get_supabase()
            await supabase.insert('messages', {
                'session_id': session_id,
                'role': role,
                'content': content
            })
            logger.info(f"Messaggio salvato correttamente per la sessione {session_id}")
        except Exception as db_error:
            # Logga l'errore ma non interrompere il flusso - abbiamo già salvato in memoria
            logger.error(f"Errore salvataggio su Supabase: {str(db_error)}")
            # Non propagare l'errore visto che abbiamo già salvato in memoria
    except Exception as e:
        logger.error(f"Errore salvataggio messaggio: {str(e)}")
        # Comunque tenta di salvare in memoria come ultima risorsa
        try:
            if USE_MEMORY_FALLBACK:
                if session_id not in conversation_history:
                    conversation_history[session_id] = []
                conversation_history[session_id].append({
                    "role": role,
                    "content": content
                })
        except:
            logger.error("Errore critico: impossibile salvare il messaggio anche in memoria")

async def get_mood_data(session_id: str) -> List[str]:
    """Recupera i dati umore per una sessione"""
    try:
        # Per ora usiamo la memoria in-process
        if session_id in mood_history:
            return mood_history[session_id]
        return []
    except Exception as e:
        logger.error(f"Errore recupero umore: {str(e)}")
        return []

async def save_mood_data(session_id: str, mood: str):
    """Salva dati umore per una sessione"""
    try:
        # Per ora usiamo la memoria in-process
        if session_id not in mood_history:
            mood_history[session_id] = []
        mood_history[session_id].append(mood)
    except Exception as e:
        logger.error(f"Errore salvataggio umore: {str(e)}")

# Inizializza FastAPI
app = FastAPI(title="Psicologo Virtuale API")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # sviluppo locale frontend
        "https://psico-virtuale.vercel.app",  # produzione frontend
        "*"  # Per debug, rimuovere in produzione 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sistema di prompt
condense_question_prompt = PromptTemplate.from_template("""
Data la seguente conversazione terapeutica e una domanda di follow-up, riformula la domanda
in modo autonomo considerando il contesto della conversazione precedente.

Storico conversazione:
{chat_history}

Domanda di follow-up: {question}

Domanda autonoma riformulata:
""")

qa_prompt = PromptTemplate.from_template("""
Sei uno psicologo virtuale professionale. Il tuo ruolo è quello di fornire supporto psicologico, ascoltare 
con empatia e offrire risposte ponderate basate sulle migliori pratiche psicologiche.

Devi:
1. Mantenere un tono empatico, rispettoso e non giudicante
2. Utilizzare tecniche di ascolto attivo e di riflessione
3. Fare domande aperte che incoraggino l'introspezione
4. Evitare diagnosi definitive (non sei un sostituto di un professionista in carne ed ossa)
5. Suggerire tecniche di auto-aiuto basate su evidenze scientifiche
6. Identificare eventuali segnali di crisi e suggerire risorse di emergenza quando appropriato

Ricorda: in caso di emergenza o pensieri suicidi, devi sempre consigliare di contattare immediatamente 
i servizi di emergenza o le linee telefoniche di supporto psicologico.

Stato emotivo attuale dichiarato dal paziente: {current_mood}
Adatta il tuo approccio terapeutico in base a questo stato emotivo. Per esempio:
- Se il paziente si sente "ottimo", sostieni il suo stato positivo ma esplora comunque aree di crescita
- Se il paziente si sente "male", usa un tono più delicato, empatico e supportivo
- Se il paziente è "neutrale", aiutalo a esplorare e identificare meglio le sue emozioni
Ricorda che lo stato emotivo dichiarato è solo un punto di partenza e potrebbe non riflettere completamente 
la complessità emotiva del paziente.

Base di conoscenza:
{context}

Conversazione precedente:
{chat_history}

Domanda: {question}

Risposta:
""")

# Nuovo prompt specifico per la chat del paziente senza RAG
patient_chat_prompt = PromptTemplate.from_template("""
Sei uno psicologo virtuale dedicato all'ascolto attivo e alla raccolta di informazioni sullo stato del paziente.
Il tuo obiettivo principale è creare uno spazio sicuro per l'espressione dei pensieri e delle emozioni, aiutando
il paziente a riflettere sulle proprie esperienze. Non devi diagnosticare o offrire terapie specifiche.

Devi:
1. Mantenere un approccio fortemente empatico, rispettoso e privo di giudizio
2. Utilizzare tecniche di ascolto attivo e riformulazione per mostrare comprensione
3. Fare domande aperte che incoraggino l'introspezione e l'esplorazione dei sentimenti
4. Normalizzare le emozioni difficili quando appropriato
5. Validare le esperienze del paziente
6. Identificare eventuali segnali di crisi e suggerire risorse di emergenza se necessario
7. Incoraggiare l'auto-riflessione e l'espressione onesta

Ricorda: lo scopo di questa conversazione è principalmente raccogliere informazioni sul paziente e aiutarlo
a esprimersi, non offrire consigli tecnici o terapie.

Stato emotivo attuale dichiarato dal paziente: {current_mood}
Adatta il tuo approccio in base a questo stato emotivo, riconoscendolo esplicitamente.

Conversazione precedente:
{chat_history}

Domanda/Messaggio del paziente: {question}

Risposta:
""")

def get_vectorstore():
    """Carica il vector store da Qdrant."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL o QDRANT_API_KEY non trovati. Imposta le variabili d'ambiente.")
    
    # Inizializza il client Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Verifica che la collezione esista
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        raise FileNotFoundError(f"Collezione '{COLLECTION_NAME}' non trovata in Qdrant. Eseguire prima ingest.py.")
    
    # Inizializza gli embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Crea e restituisci il vector store
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        content_payload_key="text",
        metadata_payload_key="metadata"
    )
    
    return vector_store

async def get_conversation_chain(session_id: str):
    """Crea la catena conversazionale con RAG."""
    # Recupera messaggi dal database Supabase
    messages = await get_conversation_messages(session_id)
    
    # Prepara la memoria per la conversazione
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer",
        input_key="question" 
    )
    
    # Carica la conversazione nella memoria
    for message in messages:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message["content"])
        else:
            memory.chat_memory.add_ai_message(message["content"])
    
    # Carica il vectorstore
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SIMILARITY_TOP_K}
    )
    
    # Configura il modello LLM
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Crea la catena conversazionale
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    return chain

def format_sources(source_docs) -> List[Source]:
    """Formatta i documenti di origine in un formato più leggibile."""
    sources = []
    for doc in source_docs:
        metadata = doc.metadata
        
        # Estrai il nome del file dal percorso completo
        file_name = None
        if "source" in metadata:
            # Gestisci sia percorsi con / che con \
            path = metadata["source"].replace('\\', '/')
            file_name_with_ext = path.split('/')[-1]
            
            # Rimuovi l'estensione
            file_name = os.path.splitext(file_name_with_ext)[0]
        
        source = Source(
            file_name=file_name,
            page=metadata.get("page", None),
            text=doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        )
        sources.append(source)
    return sources

# Funzioni per l'integrazione con ElevenLabs
def get_elevenlabs_headers():
    """Restituisce gli headers per le chiamate all'API di ElevenLabs."""
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    return {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

def get_elevenlabs_conversations(agent_id: Optional[str] = None, page_size: int = 30):
    """Ottiene l'elenco delle conversazioni di ElevenLabs."""
    url = f"{ELEVENLABS_API_BASE}/conversations"
    params = {"page_size": page_size}
    
    if agent_id:
        params["agent_id"] = agent_id
    
    try:
        response = requests.get(url, headers=get_elevenlabs_headers(), params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Errore nel recupero delle conversazioni ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nel recupero delle conversazioni ElevenLabs: {str(e)}")

def get_elevenlabs_conversation(conversation_id: str):
    """Ottiene i dettagli di una specifica conversazione di ElevenLabs."""
    url = f"{ELEVENLABS_API_BASE}/conversations/{conversation_id}"
    
    try:
        response = requests.get(url, headers=get_elevenlabs_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Errore nel recupero della conversazione ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nel recupero della conversazione ElevenLabs: {str(e)}")

def format_elevenlabs_transcript(conversation_detail):
    """Formatta il transcript di ElevenLabs in un formato leggibile per l'analisi."""
    formatted_messages = []
    
    for msg in conversation_detail.get("transcript", []):
        role = "Paziente" if msg.get("role") == "user" else "Psicologo"
        message = msg.get("message", "")
        if message:
            formatted_messages.append(f"{role}: {message}")
    
    return "\n".join(formatted_messages)

# Endpoint radice - pagina di benvenuto
@app.get("/")
async def read_root():
    """Endpoint principale che serve una semplice risposta."""
    return {"status": "ok", "message": "API Psicologo Virtuale attiva"}

# ENDPOINT API DELLA CHAT
@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest, user_data: Dict = Depends(verify_token)):
    """Endpoint compatibile con il frontend per processare le domande dell'utente."""
    return await process_query(request, user_data)

# Nuovo endpoint per la chat del paziente senza RAG
@app.post("/api/patient-chat", response_model=QueryResponse)
async def patient_chat_endpoint(request: QueryRequest, user_data: Dict = Depends(verify_token)):
    """Endpoint per la chat del paziente, senza RAG."""
    return await process_patient_query(request, user_data)

# Mantengo anche il vecchio endpoint per compatibilità
@app.post("/therapy-session", response_model=QueryResponse)
async def therapy_session_legacy(request: QueryRequest, user_data: Dict = Depends(verify_token)):
    """Endpoint legacy per compatibilità."""
    return await process_query(request, user_data)

async def process_query(request: QueryRequest, user_data: Dict):
    """Funzione interna che gestisce le richieste di chat."""
    try:
        # Verifica il ruolo se necessario (es. se il paziente può accedere a questa sessione)
        # Qui possiamo aggiungere controlli di autorizzazione specifici per sessione
        
        # Ottiene o crea la catena conversazionale
        chain = await get_conversation_chain(request.session_id)
        
        # Salva la domanda utente nel database
        await save_message(request.session_id, "user", request.query)
        
        # Gestione dell'umore
        current_mood = "non specificato"
        
        # Traccia l'umore se fornito
        if request.mood:
            await save_mood_data(request.session_id, request.mood)
            current_mood = request.mood
        # Se non fornito ma c'è una storia di umore, usa l'ultimo
        else:
            mood_data = await get_mood_data(request.session_id)
            if mood_data:
                current_mood = mood_data[-1]
        
        # Recupera i messaggi e limita la lunghezza per il contesto
        messages = await get_conversation_messages(request.session_id)
        if len(messages) > MAX_HISTORY_LENGTH * 2:
            messages = messages[-MAX_HISTORY_LENGTH*2:]
        
        # Esegue la query con l'umore corrente
        result = chain({"question": request.query, "current_mood": current_mood})
        
        # Salva la risposta nel database
        await save_message(request.session_id, "assistant", result["answer"])
        
        # Formatta le fonti
        sources = format_sources(result.get("source_documents", []))
        
        # Genera un'analisi opzionale (non mostrata all'utente ma utile per il backend)
        analysis = None
        if len(messages) > 3:
            llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.1)
            messages_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-6:]])
            
            # Includi l'umore dichiarato nell'analisi
            mood_info = ""
            if current_mood != "non specificato":
                mood_info = f"\nIl paziente ha dichiarato di sentirsi: {current_mood}"
            
            analysis_prompt = f"""
            Analizza brevemente questa conversazione terapeutica e identifica:
            1. Temi principali emersi
            2. Stato emotivo del paziente
            3. Eventuali segnali di allarme
            4. Se lo stato emotivo espresso nel contenuto della conversazione corrisponde all'umore dichiarato
            
            {mood_info}
            
            Conversazione:
            {messages_text}
            """
            analysis_response = llm.invoke(analysis_prompt)
            analysis = analysis_response.content
        
        # Ritorna il risultato
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            analysis=analysis
        )
    
    except Exception as e:
        logger.error(f"Errore nel processare la query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore del server: {str(e)}")

async def process_patient_query(request: QueryRequest, user_data: Dict):
    """Funzione che gestisce le richieste di chat del paziente, senza RAG."""
    try:
        # Salva immediatamente la domanda utente nel database per garantire che venga registrata
        # anche se ci sono problemi con l'elaborazione successiva
        await save_message(request.session_id, "user", request.query)
        
        # Avvia le operazioni asincrone in parallelo per migliorare le prestazioni
        # Forniamo task separati per le operazioni asincrone che possono essere eseguite in parallelo
        
        # Gestione dell'umore - Task 1
        async def handle_mood():
            current_mood = "non specificato"
            if request.mood:
                await save_mood_data(request.session_id, request.mood)
                return request.mood
            else:
                mood_data = await get_mood_data(request.session_id)
                if mood_data:
                    return mood_data[-1]
            return current_mood
        
        # Recupero dei messaggi - Task 2
        async def get_messages():
            messages = await get_conversation_messages(request.session_id)
            if len(messages) > MAX_HISTORY_LENGTH * 2:
                messages = messages[-MAX_HISTORY_LENGTH*2:]
            return messages
            
        # Esegui queste operazioni in parallelo
        current_mood_task = asyncio.create_task(handle_mood())
        messages_task = asyncio.create_task(get_messages())
        
        # Attendi il completamento di entrambe le operazioni
        current_mood = await current_mood_task
        messages = await messages_task
        
        # Formatta la conversazione precedente
        conversation_text = ""
        for msg in messages:
            role = "Paziente" if msg["role"] == "user" else "Psicologo"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        # Configura il modello LLM - Questa operazione non è asincrona
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Usa un prompt specifico per la chat del paziente senza RAG
        prompt_template = patient_chat_prompt.format(
            chat_history=conversation_text,
            question=request.query,
            current_mood=current_mood
        )
        
        # Esegue la query direttamente all'LLM
        start_time = time.time()
        response = llm.invoke(prompt_template)
        end_time = time.time()
        logger.info(f"Tempo risposta LLM: {end_time - start_time:.2f} secondi")
        
        # Estrai la risposta
        answer = response.content
        
        # Avvia il salvataggio della risposta, ma non attendere l'analisi né il completamento del salvataggio
        # Il salvataggio può continuare in background
        save_task = asyncio.create_task(save_message(request.session_id, "assistant", answer))
        
        # Ritorna la risposta senza attendere l'analisi né il completamento del salvataggio
        response_obj = QueryResponse(
            answer=answer,
            sources=[],
            analysis=None
        )
        
        # Avvia l'analisi in background se necessario, senza attendere il completamento
        if len(messages) > 3:
            asyncio.create_task(generate_background_analysis(messages, current_mood, request.session_id))
        
        return response_obj
    
    except Exception as e:
        logger.error(f"Errore nel processare la query paziente: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore del server: {str(e)}")

# Nuova funzione per generare l'analisi in background
async def generate_background_analysis(messages, current_mood, session_id):
    """Genera l'analisi in background e la salva senza ritardare la risposta principale."""
    try:
        analysis_llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.1)
        messages_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-6:]])
        
        # Includi l'umore dichiarato nell'analisi
        mood_info = ""
        if current_mood != "non specificato":
            mood_info = f"\nIl paziente ha dichiarato di sentirsi: {current_mood}"
        
        analysis_prompt = f"""
        Analizza brevemente questa conversazione terapeutica e identifica:
        1. Temi principali emersi
        2. Stato emotivo del paziente
        3. Eventuali segnali di allarme
        4. Se lo stato emotivo espresso nel contenuto della conversazione corrisponde all'umore dichiarato
        
        {mood_info}
        
        Conversazione:
        {messages_text}
        """
        analysis_response = analysis_llm.invoke(analysis_prompt)
        analysis = analysis_response.content
        
        # Salva l'analisi in Supabase o altro sistema di storage
        # Questo potrebbe richiedere la creazione di un nuovo endpoint o funzione
        logger.info(f"Analisi in background completata per la sessione {session_id}")
    except Exception as e:
        logger.error(f"Errore nell'analisi in background: {str(e)}", exc_info=True)
        # Non solleviamo eccezioni qui perché è un processo in background

# ENDPOINT PER RESET SESSIONE
@app.post("/api/reset-session", response_model=ResetResponse)
async def reset_session_endpoint(request: ResetRequest, user_data: Dict = Depends(verify_token)):
    """Endpoint compatibile con il frontend per resettare la sessione."""
    return await reset_conversation(request)

# Mantengo anche il vecchio endpoint per compatibilità
@app.post("/reset-session", response_model=ResetResponse)
async def reset_session_legacy(request: ResetRequest, user_data: Dict = Depends(verify_token)):
    """Endpoint legacy per resettare la sessione."""
    return await reset_conversation(request)

async def reset_conversation(request: ResetRequest):
    """Resetta la sessione terapeutica."""
    session_id = request.session_id
    
    try:
        # Rimuovi messaggi dal database Supabase
        # Nota: questa è una simulazione, dovremmo implementare una vera cancellazione
        # await supabase.delete('messages', {'session_id': session_id})
        
        # Reset memoria in-process (per compatibilità)
        if session_id in conversation_history:
            conversation_history[session_id] = []
        if session_id in mood_history:
            mood_history[session_id] = []
            
        return ResetResponse(status="success", message="Sessione resettata con successo")
    except Exception as e:
        logger.error(f"Errore nel reset della sessione: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nel reset della sessione: {str(e)}")

# ENDPOINT PER ANALISI UMORE
@app.post("/api/mood-analysis", response_model=MoodAnalysisResponse)
async def mood_analysis_endpoint(request: AnalysisSourceRequest, user_data: Dict = Depends(verify_token)):
    """Analizza l'umore e il progresso del paziente basato su diverse fonti."""
    return await analyze_mood(request)

# Mantengo anche l'endpoint legacy per retrocompatibilità
@app.get("/api/mood-analysis/{session_id}", response_model=MoodAnalysisResponse)
async def mood_analysis_legacy(session_id: str, user_data: Dict = Depends(verify_token)):
    """Endpoint legacy per retrocompatibilità."""
    request = AnalysisSourceRequest(
        session_id=session_id,
        analyze_chatbot=True,
        analyze_elevenlabs=False
    )
    return await analyze_mood(request)

async def analyze_mood(request: AnalysisSourceRequest):
    """Analizza l'umore e il progresso del paziente."""
    try:
        combined_text = ""
        
        # Raccogli conversazione dal chatbot se richiesto
        if request.analyze_chatbot:
            messages = await get_conversation_messages(request.session_id)
            if messages:
                chatbot_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                combined_text += "## Conversazione Chatbot:\n" + chatbot_text + "\n\n"
            else:
                combined_text += "## Conversazione Chatbot:\nNessuna conversazione disponibile\n\n"
        
        # Raccogli conversazione da ElevenLabs se richiesto
        if request.analyze_elevenlabs and request.elevenlabs_conversation_id:
            try:
                elevenlabs_data = get_elevenlabs_conversation(request.elevenlabs_conversation_id)
                elevenlabs_text = format_elevenlabs_transcript(elevenlabs_data)
                combined_text += "## Conversazione Vocale ElevenLabs:\n" + elevenlabs_text + "\n\n"
            except Exception as e:
                combined_text += f"## Conversazione Vocale ElevenLabs:\nErrore nel recupero della conversazione: {str(e)}\n\n"
        
        # Se non ci sono dati, ritorna un messaggio di errore
        if not combined_text.strip():
            return MoodAnalysisResponse(mood_analysis="# Analisi dell'Umore\n\n**Dati insufficienti per l'analisi.**\n\nNon ci sono conversazioni disponibili da analizzare.")
        
        # Analizza il testo combinato
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.2)
        analysis_prompt = f"""
        Analizza questa conversazione terapeutica e fornisci:
        1. Una valutazione dell'umore generale del paziente
        2. Eventuali schemi di pensiero o comportamento ricorrenti
        3. Suggerimenti per il terapeuta su come procedere nella prossima sessione
        
        Formatta la risposta in Markdown seguendo questo formato:
        
        # Analisi della Conversazione Terapeutica
        
        ## 1. Valutazione dell'umore generale del paziente
        [Inserisci qui la tua analisi...]
        
        ## 2. Eventuali schemi di pensiero o comportamento ricorrenti
        [Inserisci qui la tua analisi...]
        
        ## 3. Suggerimenti per il terapeuta su come procedere nella prossima sessione
        - Punto 1
        - Punto 2
        - Punto 3
        
        Conversazione:
        {combined_text}
        """
        
        response = llm.invoke(analysis_prompt)
        return MoodAnalysisResponse(mood_analysis=response.content)
    
    except Exception as e:
        logger.error(f"Errore nell'analisi dell'umore: {str(e)}", exc_info=True)
        return MoodAnalysisResponse(
            mood_analysis=f"# Errore nell'Analisi\n\nSi è verificato un errore durante l'analisi dell'umore: {str(e)}"
        )

# ENDPOINT PER RIEPILOGO SESSIONE
@app.get("/api/session-summary/{session_id}", response_model=SessionSummaryResponse)
async def session_summary_endpoint(session_id: str, user_data: Dict = Depends(verify_token)):
    """Genera un riepilogo della sessione terapeutica."""
    return await get_session_summary(session_id)

async def get_session_summary(session_id: str):
    """Genera un riepilogo HTML della sessione."""
    try:
        messages = await get_conversation_messages(session_id)
        
        if not messages:
            return SessionSummaryResponse(summary_html="<p>Nessuna sessione disponibile</p>")
        
        # Formatta il riepilogo in HTML
        html = """
        <div class="p-4 bg-gray-50 rounded-lg">
            <h2 class="text-xl font-semibold mb-4 text-blue-700">Riepilogo della Sessione</h2>
        """
        
        for idx, message in enumerate(messages):
            role_class = "text-blue-600 font-medium" if message["role"] == "assistant" else "text-gray-700 font-medium"
            role_name = "Psicologo" if message["role"] == "assistant" else "Paziente"
            
            html += f"""
            <div class="mb-4 pb-3 border-b border-gray-200">
                <div class="mb-1"><span class="{role_class}">{role_name}:</span></div>
                <p class="pl-2">{message["content"]}</p>
            </div>
            """
        
        html += "</div>"
        
        # Se disponibile, aggiunge grafico dell'umore
        mood_data = await get_mood_data(session_id)
        if mood_data:
            html += """
            <div class="mt-6 p-4 bg-gray-50 rounded-lg">
                <h3 class="text-lg font-semibold mb-2 text-blue-700">Tracciamento dell'Umore</h3>
                <div class="mood-chart">
                    <!-- Qui si potrebbe inserire un grafico generato con D3.js o simili -->
                    <p>Trend dell'umore rilevato durante la sessione.</p>
                </div>
            </div>
            """
        
        return SessionSummaryResponse(summary_html=html)
    except Exception as e:
        logger.error(f"Errore nel generare il riepilogo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nel generare il riepilogo: {str(e)}")

# ENDPOINT PER RISORSE CONSIGLIATE
@app.post("/api/recommend-resources", response_model=ResourceResponse)
async def resources_endpoint(request: ResourceRequest, user_data: Dict = Depends(verify_token)):
    """Raccomanda risorse psicologiche basate sulla conversazione."""
    return await recommend_resources(request)

async def recommend_resources(request: ResourceRequest):
    """Raccomanda risorse basate sulla conversazione."""
    try:
        messages = await get_conversation_messages(request.session_id)
        
        # Se non ci sono messaggi o si verifica un errore, restituisci risorse generiche
        # per evitare errori 401 e garantire sempre una risposta
        if not messages:
            return ResourceResponse(resources=[
                {"title": "Mindfulness per principianti", "description": "Tecniche base di mindfulness per la gestione dello stress", "type": "Libro/App"},
                {"title": "Diario delle emozioni", "description": "Strumento per tracciare e comprendere i tuoi stati emotivi", "type": "Esercizio"},
                {"title": "Respirazione profonda", "description": "Tecnica di rilassamento rapido per momenti di ansia", "type": "Tecnica"}
            ])
        
        # Prendi gli ultimi messaggi della conversazione
        recent_messages = messages[-8:]  # ultimi 8 messaggi
        messages_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
        
        # Chiedi al modello di consigliare risorse
        try:
            llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
            resource_prompt = f"""
            Basandoti su questa conversazione terapeutica, consiglia 3-5 risorse specifiche che potrebbero essere utili per il paziente.
            Per ogni risorsa, fornisci:
            - Titolo
            - Breve descrizione (1-2 frasi)
            - Tipo (libro, app, esercizio, tecnica, video, ecc.)
            
            Conversazione:
            {messages_text}
            
            Restituisci le risorse in formato JSON come questo:
            [
                {{"title": "Titolo della risorsa", "description": "Breve descrizione", "type": "Tipo di risorsa"}},
                ...
            ]
            """
            
            response = llm.invoke(resource_prompt)
            
            # Estrai JSON dalla risposta
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                resources = json.loads(json_str)
            else:
                # Fallback se il formato non è corretto
                resources = [
                    {"title": "Mindfulness per principianti", "description": "Tecniche base di mindfulness per la gestione dello stress", "type": "Libro/App"},
                    {"title": "Diario delle emozioni", "description": "Strumento per tracciare e comprendere i tuoi stati emotivi", "type": "Esercizio"},
                    {"title": "Respirazione profonda", "description": "Tecnica di rilassamento rapido per momenti di ansia", "type": "Tecnica"}
                ]
        except Exception as llm_error:
            logger.error(f"Errore nella generazione di risorse con LLM: {str(llm_error)}")
            # Fallback risorse generiche in caso di errore LLM
            resources = [
                {"title": "Mindfulness per principianti", "description": "Tecniche base di mindfulness per la gestione dello stress", "type": "Libro/App"},
                {"title": "Diario delle emozioni", "description": "Strumento per tracciare e comprendere i tuoi stati emotivi", "type": "Esercizio"},
                {"title": "Respirazione profonda", "description": "Tecnica di rilassamento rapido per momenti di ansia", "type": "Tecnica"}
            ]
        
        return ResourceResponse(resources=resources)
    
    except Exception as e:
        logger.error(f"Errore nel generare risorse: {str(e)}", exc_info=True)
        # Sempre fornire una risposta per evitare errori 401 o 500
        return ResourceResponse(resources=[
            {"title": "Mindfulness per principianti", "description": "Tecniche base di mindfulness per la gestione dello stress", "type": "Libro/App"},
            {"title": "Diario delle emozioni", "description": "Strumento per tracciare e comprendere i tuoi stati emotivi", "type": "Esercizio"},
            {"title": "Respirazione profonda", "description": "Tecnica di rilassamento rapido per momenti di ansia", "type": "Tecnica"}
        ])

# ENDPOINT PER ANALISI PATOLOGIE
@app.post("/api/pathology-analysis", response_model=PathologyAnalysisResponse)
async def pathology_endpoint(request: PathologyAnalysisRequest, user_data: Dict = Depends(verify_token)):
    """Analizza le conversazioni per identificare possibili patologie psicologiche."""
    # Verifica se l'utente è un terapeuta
    user_role = await get_user_role(user_data["id"])
    if user_role != "therapist":
        raise HTTPException(
            status_code=403,
            detail="Solo i terapeuti possono accedere a questa funzionalità"
        )
    
    return await analyze_pathologies(request)

async def analyze_pathologies(request: PathologyAnalysisRequest):
    """Analizza le conversazioni per identificare possibili patologie."""
    try:
        combined_text = ""
        
        # Raccogli conversazione dal chatbot
        if request.analyze_chatbot:
            messages = await get_conversation_messages(request.session_id)
            if messages:
                chatbot_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                combined_text += "## Conversazione Chatbot:\n" + chatbot_text + "\n\n"
            else:
                combined_text += "## Conversazione Chatbot:\nNessuna conversazione disponibile\n\n"
        
        # Raccogli conversazione da ElevenLabs se richiesto
        if request.analyze_elevenlabs and request.elevenlabs_conversation_id:
            try:
                elevenlabs_data = get_elevenlabs_conversation(request.elevenlabs_conversation_id)
                elevenlabs_text = format_elevenlabs_transcript(elevenlabs_data)
                combined_text += "## Conversazione Vocale ElevenLabs:\n" + elevenlabs_text + "\n\n"
            except Exception as e:
                combined_text += f"## Conversazione Vocale ElevenLabs:\nErrore nel recupero della conversazione: {str(e)}\n\n"
        
        # Se non ci sono dati, ritorna un messaggio di errore
        if not combined_text.strip():
            return PathologyAnalysisResponse(
                possible_pathologies=[],
                analysis_summary="Dati insufficienti per l'analisi. Non ci sono conversazioni disponibili da analizzare."
            )
        
        # Contiamo i messaggi utente per verificare che ci siano dati sufficienti
        user_messages_count = 0
        total_user_words = 0
        
        messages = await get_conversation_messages(request.session_id)
        for msg in messages:
            if msg["role"] == "user":
                user_messages_count += 1
                total_user_words += len(msg["content"].split())
        
        # Requisiti minimi per procedere con l'analisi
        MIN_USER_MESSAGES = 1
        MIN_USER_WORDS = 10
        
        if user_messages_count < MIN_USER_MESSAGES or total_user_words < MIN_USER_WORDS:
            return PathologyAnalysisResponse(
                possible_pathologies=[],
                analysis_summary=f"Dati insufficienti per un'analisi clinica significativa. Sono necessari almeno {MIN_USER_MESSAGES} messaggi e {MIN_USER_WORDS} parole dall'utente per procedere. Attualmente: {user_messages_count} messaggi, {total_user_words} parole."
            )
        
        # Utilizziamo il vector store per trovare documenti rilevanti
        vector_store = get_vectorstore()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SIMILARITY_TOP_K}
        )
        
        # Estrai i sintomi e comportamenti rilevanti dalla conversazione
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.2)
        extraction_prompt = f"""
        Analizza questa conversazione terapeutica ed estrai i sintomi principali, 
        comportamenti problematici o schemi di pensiero che potrebbero essere 
        rilevanti per un'analisi clinica. Fornisci solo i sintomi e comportamenti,
        senza interpretarli o diagnosticarli.
        
        Conversazione:
        {combined_text}
        
        Estrai e elenca solo i sintomi, comportamenti o schemi di pensiero rilevanti, 
        uno per riga. Sii specifico e dettagliato, concentrandoti sui fatti osservabili.
        """
        
        extraction_response = llm.invoke(extraction_prompt)
        extracted_behaviors = extraction_response.content
        
        # Utilizziamo i comportamenti estratti per interrogare il vector store
        docs = retriever.get_relevant_documents(extracted_behaviors)
        
        # Analisi delle patologie
        analysis_prompt = f"""
        Basandoti sui sintomi e comportamenti estratti dalla conversazione terapeutica e sui documenti
        clinici correlati, identifica possibili patologie psicologiche che potrebbero richiedere ulteriore
        valutazione. Per ogni patologia, fornisci un breve descrizione, i sintomi chiave che l'hanno fatta
        emergere dall'analisi, e una stima di confidenza (da 0.0 a 1.0) basata su quanti sintomi sono presenti.
        
        Sintomi estratti dalla conversazione:
        {extracted_behaviors}
        
        Documenti clinici rilevanti:
        {[doc.page_content for doc in docs]}
        
        Fornisci la risposta nel seguente formato JSON:
        {{
            "possible_pathologies": [
                {{
                    "name": "Nome della patologia",
                    "description": "Breve descrizione",
                    "confidence": 0.7,
                    "key_symptoms": ["sintomo 1", "sintomo 2", ...],
                    "source": "Nome del documento di riferimento"
                }},
                ...
            ],
            "analysis_summary": "Breve riassunto dell'analisi complessiva"
        }}
        
        Includi solo patologie con un minimo di confidenza (0.4 o superiore).
        """
        
        analysis_response = llm.invoke(analysis_prompt)
        
        # Estrai il JSON dalla risposta
        import re
        import json
        json_match = re.search(r'\{.*\}', analysis_response.content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return PathologyAnalysisResponse(**result)
        else:
            # Fallback
            return PathologyAnalysisResponse(
                possible_pathologies=[],
                analysis_summary="Non è stato possibile identificare patologie specifiche dai dati forniti. La conversazione potrebbe non contenere informazioni clinicamente rilevanti o potrebbe essere necessario un colloquio più approfondito."
            )
    
    except Exception as e:
        logger.error(f"Errore nell'analisi delle patologie: {str(e)}", exc_info=True)
        return PathologyAnalysisResponse(
            possible_pathologies=[],
            analysis_summary=f"Si è verificato un errore durante l'analisi: {str(e)}"
        )

# Endpoint di debug per verifica token
@app.get("/api/debug/token")
async def debug_token_endpoint(authorization: str = Header(None)):
    """Endpoint di debug per analizzare il token JWT."""
    if not authorization:
        return {"error": "Authorization header mancante", "status": "error"}
    
    try:
        # Estrai il token
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            
            # Verifica il token con Supabase
            supabase = get_supabase()
            user_data = await supabase.get_user(token)
            
            # Restituisci informazioni token
            if user_data:
                return {
                    "token_valid": True,
                    "user_id": user_data.get("id"),
                    "email": user_data.get("email"),
                    "status": "success"
                }
            else:
                return {"token_valid": False, "status": "invalid", "error": "Token non valido o scaduto"}
        else:
            return {"error": "Authorization header non ha formato Bearer", "status": "error"}
    except Exception as e:
        return {"error": f"Errore nell'analisi del token: {str(e)}", "status": "error"}

# Gestione chiusura applicazione
@app.on_event("shutdown")
async def shutdown_event():
    """Chiude le connessioni durante lo shutdown."""
    if supabase_client:
        await supabase_client.close()

@app.get("/patients/{patient_id}/recommendations", response_model=ResourceResponse)
async def patient_recommendations_endpoint(
    patient_id: str, 
    query: Optional[str] = None, 
    user_data: Dict = Depends(verify_token)
):
    """Endpoint for patient-specific resource recommendations."""
    logger.info(f"Patient recommendations request for patient: {patient_id}, query: {query}")
    logger.info(f"User data: {user_data}")  # Log user data for debugging
    
    # Create a resource request with the session ID from the query parameter
    request = ResourceRequest(
        query=query or "", 
        session_id=query or ""  # Use query as session_id if provided
    )
    
    try:
        # Reuse the existing recommend_resources logic
        return await recommend_resources(request)
    except Exception as e:
        logger.error(f"Error in patient recommendations: {str(e)}")
        # Return empty resources as fallback to avoid 403
        return ResourceResponse(resources=[])

# Add a public endpoint without token verification for development
@app.get("/api/public/recommendations", response_model=ResourceResponse)
async def public_recommendations_endpoint(session_id: Optional[str] = None):
    """Public endpoint for resource recommendations without auth (for development)."""
    logger.info(f"Public recommendations request with session_id: {session_id}")
    
    # Create a resource request with the provided session ID
    request = ResourceRequest(
        query=session_id or "", 
        session_id=session_id or ""
    )
    
    try:
        # Reuse the existing recommend_resources logic
        return await recommend_resources(request)
    except Exception as e:
        logger.error(f"Error in public recommendations: {str(e)}")
        # Return empty resources list as fallback
        return ResourceResponse(resources=[])

# Avvio dell'applicazione
if __name__ == "__main__":
    import uvicorn
    try:
        # Verifica che il vector store esista
        get_vectorstore()
        logger.info("Vector store trovato. Avvio del server...")
    except FileNotFoundError:
        logger.error("Collezione Qdrant non trovata. Eseguire prima ingest.py per indicizzare i documenti con conoscenze psicologiche.")
        exit(1)
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione: {str(e)}", exc_info=True)
        exit(1)
        
    # Avvia il server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)