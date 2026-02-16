import functools
import logging
import os
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import chromadb
import streamlit as st

warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# ==================== PAGE CONFIG (MUST BE FIRST) ====================
st.set_page_config(
    page_title="DocuMind-Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DocuMind-Agent - Intelligent Document Q&A System"},
)

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== PATH SETUP ====================
PROJECT_ROOT = Path(__file__).parent.parent
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# ==================== BACKEND IMPORTS ====================
load_dotenv()
_BACKEND_READY = False
_IMPORT_ERROR = None
_chroma_client: Optional[chromadb.Client] = None
_embedding_available = False
_embedding = None

try:
    from src.agent.agent_builder import create_agent
    from src.document_processor.pdf_loader import PDFLoader
    from src.document_processor.text_splitter import TextSplitter
    from src.vector_store.chroma_db import ChromaDBVectorStore
    from src.vector_store.embeddings_manager import EmbeddingManager

    _BACKEND_READY = True
    logger.info("✅ Backend modules loaded successfully")
except ImportError as e:
    _IMPORT_ERROR = str(e)
    logger.error(f"❌ Backend Import Error: {e}")

# ==================== CONSTANTS ====================
VECTOR_STORE_PATH = "data/vector_store/chroma"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".pdf"]

AGENT_PRESETS = {
    " Fast": {
        "model": "phi3:mini",
        "temp": 0.2,
        "iter": 1,
        "desc": "Quick responses, single iteration",
    },
    " Balanced": {
        "model": "phi3:mini",
        "temp": 0.1,
        "iter": 2,
        "desc": "Optimal speed/accuracy balance",
    },
    " Deep": {
        "model": "mistral:7b",
        "temp": 0.0,
        "iter": 4,
        "desc": "Thorough analysis, multiple iterations",
    },
}


# ==================== MODERN CSS STYLING ====================
def get_modern_style(theme_mode: str) -> str:
    """Generate modern CSS based on theme mode."""

    if theme_mode == "dark":
        colors = {
            "bg_main": "#0e1117",
            "bg_sidebar": "#1a1d24",
            "bg_card": "#262730",
            "text_primary": "#fafafa",
            "text_secondary": "#a0a0a0",
            "accent": "#4dabf7",
            "accent_hover": "#339af0",
            "border": "#3a3d47",
            "user_msg": "#1e3a8a",
            "bot_msg": "#2b2d3e",
            "success": "#51cf66",
            "warning": "#ffd43b",
            "error": "#ff6b6b",
            "shadow": "rgba(0, 0, 0, 0.3)",
        }
    else:
        colors = {
            "bg_main": "#ffffff",
            "bg_sidebar": "#f8f9fa",
            "bg_card": "#ffffff",
            "text_primary": "#1a1a1a",
            "text_secondary": "#6c757d",
            "accent": "#0068c9",
            "accent_hover": "#0056b3",
            "border": "#dee2e6",
            "user_msg": "#e3f2fd",
            "bot_msg": "#f5f5f5",
            "success": "#40c057",
            "warning": "#ffc107",
            "error": "#fa5252",
            "shadow": "rgba(0, 0, 0, 0.1)",
        }

    return f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            /* ===== GLOBAL STYLES ===== */
            * {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            .stApp {{
                background-color: {colors['bg_main']};
                color: {colors['text_primary']};
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {colors['bg_sidebar']} 0%, {colors['bg_main']} 100%);
                border-right: 1px solid {colors['border']};
            }}

            /* Hide default elements */
            header {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            .stDeployButton {{display: none;}}
            #MainMenu {{visibility: hidden;}}

            /* ===== TYPOGRAPHY ===== */
            h1, h2, h3, h4, h5, h6 {{
                font-weight: 600;
                color: {colors['text_primary']} !important;
                letter-spacing: -0.02em;
            }}

            h1 {{ font-size: 2.5rem; }}
            h2 {{ font-size: 1.75rem; }}
            h3 {{ font-size: 1.25rem; }}

            p, span, div {{
                color: {colors['text_primary']};
            }}

            /* ===== SIDEBAR SECTIONS ===== */
            .sidebar-label {{
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: {colors['text_secondary']};
                margin-bottom: 0.75rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid {colors['accent']};
            }}

            /* ===== CHAT INTERFACE ===== */
            .stChatMessage {{
                background-color: transparent;
                padding: 1rem 0;
                animation: fadeIn 0.3s ease-in;
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            [data-testid="stChatMessageContent"] {{
                border-radius: 16px;
                padding: 1.2rem 1.5rem;
                box-shadow: 0 4px 12px {colors['shadow']};
                border: 1px solid {colors['border']};
                backdrop-filter: blur(10px);
            }}

            /* Bot messages */
            div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"])
            [data-testid="stChatMessageContent"] {{
                background: linear-gradient(135deg, {colors['bot_msg']} 0%, {colors['bg_card']} 100%);
                border-left: 4px solid {colors['accent']};
            }}

            /* User messages */
            div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"])
            [data-testid="stChatMessageContent"] {{
                background: linear-gradient(135deg, {colors['user_msg']} 0%, {colors['accent']} 100%);
                color: white;
            }}

            div[data-testid="chatAvatarIcon-assistant"] {{
                background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_hover']} 100%) !important;
            }}

            /* ===== INPUT FIELD ===== */
            [data-testid="stChatInput"] {{
                border-radius: 24px;
                border: 2px solid {colors['border']};
                background-color: {colors['bg_card']};
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px {colors['shadow']};
            }}

            [data-testid="stChatInput"]:focus-within {{
                border-color: {colors['accent']};
                box-shadow: 0 4px 16px {colors['shadow']};
            }}

            /* ===== BUTTONS ===== */
            .stButton > button {{
                border-radius: 12px;
                border: none;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 2px 8px {colors['shadow']};
                background: linear-gradient(135deg, {colors['accent']} 0%, {colors['accent_hover']} 100%);
                color: white;
            }}

            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 16px {colors['shadow']};
            }}

            .stButton > button:active {{
                transform: translateY(0);
            }}

            /* ===== SELECT BOXES & INPUTS ===== */
            .stSelectbox > div > div,
            .stTextInput > div > div > input {{
                background-color: {colors['bg_card']};
                border-radius: 12px;
                border: 2px solid {colors['border']};
                color: {colors['text_primary']};
                transition: all 0.3s ease;
            }}

            .stSelectbox > div > div:hover,
            .stTextInput > div > div > input:hover {{
                border-color: {colors['accent']};
            }}

            /* ===== EXPANDER ===== */
            .streamlit-expanderHeader {{
                background-color: {colors['bg_card']};
                border-radius: 12px;
                border: 1px solid {colors['border']};
                font-weight: 600;
                transition: all 0.3s ease;
            }}

            .streamlit-expanderHeader:hover {{
                background-color: {colors['border']};
                border-color: {colors['accent']};
            }}

            /* ===== RADIO BUTTONS ===== */
            .stRadio > div {{
                background-color: {colors['bg_card']};
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid {colors['border']};
            }}

            /* ===== FILE UPLOADER ===== */
            [data-testid="stFileUploader"] {{
                border-radius: 12px;
                border: 2px dashed {colors['border']};
                background-color: {colors['bg_card']};
                transition: all 0.3s ease;
            }}

            [data-testid="stFileUploader"]:hover {{
                border-color: {colors['accent']};
                background-color: {colors['border']};
            }}

            /* ===== STATS CARDS ===== */
            .stat-card {{
                background: linear-gradient(135deg, {colors['bg_card']} 0%, {colors['bg_sidebar']} 100%);
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid {colors['border']};
                margin: 0.5rem 0;
                box-shadow: 0 2px 8px {colors['shadow']};
                transition: all 0.3s ease;
            }}

            .stat-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 16px {colors['shadow']};
            }}

            .stat-value {{
                font-size: 1.75rem;
                font-weight: 700;
                color: {colors['accent']};
            }}

            .stat-label {{
                font-size: 0.875rem;
                color: {colors['text_secondary']};
                text-transform: uppercase;
                letter-spacing: 1px;
            }}

            /* ===== FILE LIST ===== */
            .file-item {{
                background-color: {colors['bg_card']};
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin: 0.5rem 0;
                border: 1px solid {colors['border']};
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: all 0.3s ease;
            }}

            .file-item:hover {{
                background-color: {colors['border']};
                transform: translateX(4px);
            }}

            .file-name {{
                font-weight: 500;
                color: {colors['text_primary']};
            }}

            .file-meta {{
                font-size: 0.75rem;
                color: {colors['text_secondary']};
            }}

            /* ===== SCROLLBAR ===== */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}

            ::-webkit-scrollbar-track {{
                background: {colors['bg_sidebar']};
            }}

            ::-webkit-scrollbar-thumb {{
                background: {colors['accent']};
                border-radius: 4px;
            }}

            ::-webkit-scrollbar-thumb:hover {{
                background: {colors['accent_hover']};
            }}
        </style>
    """


# ==================== UTILITY FUNCTIONS ====================


@contextmanager
def temporary_file(suffix=".pdf"):
    """Context manager for temporary files."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield tf
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def validate_uploaded_file(file) -> tuple[bool, str]:
    """Validate uploaded file."""
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"

    if not any(file.name.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    return True, "OK"


# ==================== CACHE FUNCTIONS ====================


@functools.lru_cache(maxsize=1)
def get_vector_store_path() -> Path:
    """Get vector store path (cached)."""
    path = Path(VECTOR_STORE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_embedding_manager():
    global _embedding_available, _embedding

    if _embedding is None:
        if _BACKEND_READY:
            manager = EmbeddingManager()
            _embedding = manager.chroma_embedding_function()
            _embedding_available = True
    return _embedding


def get_chroma_client():
    global _chroma_client

    if _chroma_client is None:
        path = get_vector_store_path()
        _chroma_client = chromadb.PersistentClient(path=str(path))

    return _chroma_client


def get_vector_store(collection_name: str):
    return ChromaDBVectorStore(
        collection_name=collection_name,
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=get_embedding_manager(),
        reset_on_start=False,
        client=get_chroma_client(),
    )


def get_available_collections() -> List[str]:
    """Get list of available collections."""
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]

@functools.lru_cache(maxsize=32)
def get_collection_stats(collection_name: str) -> Dict[str, Any]:
    """Get statistics for a collection."""
    if not _BACKEND_READY:
        return {"error": "Backend not ready"}

    try:
        vector_store = ChromaDBVectorStore(
            collection_name=collection_name,
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=get_embedding_manager(),
            reset_on_start=False,
            client=get_chroma_client(),
        )

        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {"error": str(e)}


def get_files_in_collection(collection_name: str) -> List[Dict[str, Any]]:
    """Get list of files in a collection with metadata."""
    if not _BACKEND_READY:
        return []

    try:
        vector_store = ChromaDBVectorStore(
            collection_name=collection_name,
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=get_embedding_manager(),
            reset_on_start=False,
            client=get_chroma_client(),
        )

        all_docs = vector_store.collection.get()

        if not all_docs or not all_docs.get("metadatas"):
            return []

        files_dict = {}
        for metadata in all_docs["metadatas"]:
            if metadata and "filename" in metadata:
                filename = metadata["filename"]
                if filename not in files_dict:
                    files_dict[filename] = {
                        "name": filename,
                        "chunks": 0,
                        "total_chars": 0,
                        "source": metadata.get("source", "unknown"),
                        "timestamp": metadata.get("timestamp", "unknown"),
                    }
                files_dict[filename]["chunks"] += 1
                files_dict[filename]["total_chars"] += metadata.get("text_length", 0)

        return list(files_dict.values())

    except Exception as e:
        logger.error(f"Error getting files in collection: {e}")
        return []


# ==================== SESSION STATE ====================


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "messages": [],
        "agents": {},  # Dict to store agents by mode
        "collection_name": "documents",
        "agent_mode": " Balanced",
        "theme": "dark",
        "show_collection_creator": False,
        "upload_success": False,
        "last_error": None,
        "refresh_collections": False,  # Flag to trigger collections refresh
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================== BACKEND OPERATIONS ====================


def process_uploaded_files(files: List, collection_name: str) -> tuple[bool, str]:
    """Process uploaded PDF files."""
    if not _BACKEND_READY:
        return False, "Backend not ready"

    if not files:
        return False, "No files provided"

    try:
        vector_store = ChromaDBVectorStore(
            collection_name=collection_name,
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=get_embedding_manager(),
            reset_on_start=False,
            client=get_chroma_client(),
        )
        loader = PDFLoader(loader_type="auto")
        splitter = TextSplitter(chunk_size=300, chunk_overlap=50, strategy="token")

        all_chunks = []
        processed_files = []

        for file in files:
            valid, msg = validate_uploaded_file(file)
            if not valid:
                logger.warning(f"File validation failed: {file.name} - {msg}")
                continue

            try:
                with temporary_file() as tmp:
                    file.seek(0)
                    tmp.write(file.read())
                    tmp.flush()
                    tmp.close()

                    raw_docs = loader.load_pdf(tmp.name)
                    text = "\n\n".join([d.get("text", "") for d in raw_docs])

                    chunks = splitter.split_text(
                        text,
                        metadata={
                            "filename": file.name,
                            "source": collection_name,
                            "upload_date": datetime.now().isoformat(),
                            "file_size": file.size,
                        },
                    )

                    all_chunks.extend(chunks)
                    processed_files.append(file.name)
                    logger.info(f"Processed {file.name}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                continue

        if not all_chunks:
            return False, "No chunks extracted from files"

        docs_to_add = [
            {"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
            for chunk in all_chunks
        ]

        added_count = vector_store.add_documents(docs_to_add, batch_size=100)

        return True, f"✅ Processed {len(processed_files)} file(s), added {added_count} chunks"

    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg


def get_agent_for_mode(mode: str) -> Optional[Any]:
    """Get or create agent for specific mode (lazy loading with caching)."""
    if not _BACKEND_READY:
        return None

    # Check if agent exists for this mode
    if mode in st.session_state.agents:
        logger.info(f"Using cached agent for mode: {mode}")
        return st.session_state.agents[mode]

    # Create new agent
    preset = AGENT_PRESETS[mode]

    try:
        logger.info(f"Creating new agent for mode: {mode}")
        agent = create_agent(
            vector_store_config={
                "collection_name": st.session_state.collection_name,
                "persist_directory": VECTOR_STORE_PATH,
                "client": get_chroma_client(),
            },
            grader_config={
                "grader_type": "robust",
                "confidence_threshold": 0.6,
                "model_name": "phi3:mini",
            },
            generator_config={"model_name": preset["model"], "temperature": preset["temp"]},
            max_iterations=preset["iter"],
            use_tools=True,
        )

        # Cache the agent
        st.session_state.agents[mode] = agent
        logger.info(f"Agent created and cached for mode: {mode}")
        return agent

    except Exception as e:
        error_msg = f"Agent initialization failed: {str(e)}"
        logger.error(error_msg)
        st.session_state.last_error = error_msg
        return None


# ==================== UI COMPONENTS ====================


def render_sidebar():
    """Render sidebar with all controls."""
    with st.sidebar:
        # Header
        st.markdown("# DocuMind-Agent")
        st.markdown("*Intelligent Document Q&A*")
        st.markdown("---")

        # ===== SECTION 1: KNOWLEDGE BASE =====
        st.markdown("<div class='sidebar-label'>📚 KNOWLEDGE BASE</div>", unsafe_allow_html=True)

        # Collection selector
        collections = get_available_collections()

        # Ensure current collection is in the list (in case it was just created)
        if st.session_state.collection_name not in collections:
            collections.append(st.session_state.collection_name)
            collections.sort()

        # Add "Create New" option button
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox(
                "Active Collection",
                collections,
                index=(
                    collections.index(st.session_state.collection_name)
                    if st.session_state.collection_name in collections
                    else 0
                ),
                label_visibility="collapsed",
                help="Select a knowledge base",
                key="collection_selector",
            )
        with col2:
            if st.button("➕", help="Create new collection"):
                st.session_state.show_collection_creator = True

        # Handle collection change
        if selected != st.session_state.collection_name:
            st.session_state.collection_name = selected
            # Clear agents cache when collection changes
            st.session_state.agents = {}
            # Clear chat when collection changes
            st.session_state.messages = []
            logger.info(f"Changed collection to: {selected}")

        # Collection creator modal
        if st.session_state.show_collection_creator:
            with st.container():
                st.markdown("**Create New Collection**")
                new_name = st.text_input(
                    "Collection Name",
                    placeholder="e.g., Documents",
                    help="Alphanumeric and underscores only",
                    key="new_collection_name",
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "Create", type="primary", use_container_width=True, key="create_btn"
                    ):
                        if new_name and new_name.replace("_", "").isalnum():
                            # Actually create the collection in ChromaDB
                            try:
                                if _BACKEND_READY:
                                    get_vector_store(collection_name=new_name)
                                    logger.info(f"Created collection: {new_name}")

                                st.session_state.collection_name = new_name
                                st.session_state.agents = {}
                                st.session_state.show_collection_creator = False
                                st.session_state.refresh_collections = True  # Trigger refresh
                                st.success(f"✅ Created: {new_name}")
                                time.sleep(0.5)
                            except Exception as e:
                                logger.error(f"Failed to create collection: {e}")
                                st.error(f"Failed: {str(e)}")
                        else:
                            st.error("Invalid name. Use letters, numbers, and underscores only.")

                with col2:
                    if st.button("Cancel", use_container_width=True, key="cancel_btn"):
                        st.session_state.show_collection_creator = False

        # Collection stats
        stats = get_collection_stats(st.session_state.collection_name)

        if "error" not in stats:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-value'>{stats.get('total_documents', 0)}</div>"
                    f"<div class='stat-label'>Docs</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                files = get_files_in_collection(st.session_state.collection_name)
                st.markdown(
                    f"<div class='stat-card'>"
                    f"<div class='stat-value'>{len(files)}</div>"
                    f"<div class='stat-label'>Files</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ===== SECTION 2: FILE MANAGEMENT =====
        with st.expander(f"📂 Files in '{st.session_state.collection_name}'", expanded=False):
            # File upload
            uploaded = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help=f"Max {MAX_FILE_SIZE_MB}MB per file",
                key="file_uploader",
            )

            if uploaded:
                if st.button(
                    "📤 Process Files", use_container_width=True, type="primary", key="process_btn"
                ):
                    with st.spinner("Processing..."):
                        success, message = process_uploaded_files(
                            uploaded, st.session_state.collection_name
                        )

                        if success:
                            st.success(message)
                            st.session_state.upload_success = True
                            # Don't rerun - just show success
                        else:
                            st.error(message)

            st.markdown("---")

            # List files
            files = get_files_in_collection(st.session_state.collection_name)

            if files:
                st.caption(f"**{len(files)} file(s)**")

                for file in files:
                    st.markdown(
                        f"<div class='file-item'>"
                        f"<div>"
                        f"<div class='file-name'>📄 {file['name']}</div>"
                        f"<div class='file-meta'>{file['chunks']} chunks • "
                        f"{format_file_size(file['total_chars'])}</div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("📭 No files yet")

        st.markdown("---")

        # ===== SECTION 3: AGENT MODE SELECTION =====
        st.markdown("<div class='sidebar-label'>🤖 AGENT MODE</div>", unsafe_allow_html=True)

        mode_options = list(AGENT_PRESETS.keys())
        selected_mode = st.radio(
            "Agent Mode",
            mode_options,
            index=mode_options.index(st.session_state.agent_mode),
            format_func=lambda x: f"{x}",
            label_visibility="collapsed",
            help="Agent will initialize on first question",
            key="agent_mode_radio",
        )

        # Update mode (no rerun, no agent creation here)
        if selected_mode != st.session_state.agent_mode:
            st.session_state.agent_mode = selected_mode
            logger.info(f"Agent mode changed to: {selected_mode}")

        preset = AGENT_PRESETS[st.session_state.agent_mode]
        st.caption(f"*{preset['desc']}*")

        # Show if agent is initialized
        if st.session_state.agent_mode in st.session_state.agents:
            st.caption("✅ Agent ready")
        else:
            st.caption("⏳ Will initialize on first question")

        st.markdown("---")

        # ===== SECTION 4: SYSTEM CONTROLS =====
        st.markdown("<div class='sidebar-label'>🛠️ CONTROLS</div>", unsafe_allow_html=True)

        # Theme toggle
        col1, col2 = st.columns([1, 3])
        with col1:
            theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
            if st.button(theme_icon, help="Toggle theme", key="theme_btn"):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        with col2:
            st.caption(f"**{st.session_state.theme.title()} Mode**")

        # Clear chat
        if st.button("️ Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.messages = []
            st.success("Chat cleared!")

        # Exit application
        if st.button(" Exit Application", use_container_width=True, key="exit_btn"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Stop execution
            st.write("👋 Goodbye!")
            st.stop()

        # Backend status
        st.markdown("---")
        if _BACKEND_READY:
            st.success("✅ Backend Online")
        else:
            st.error("❌ Backend Error")
            if _IMPORT_ERROR:
                with st.expander("Error Details"):
                    st.code(_IMPORT_ERROR)


def render_main_chat():
    """Render main chat interface."""
    # Header
    st.title(" Chat with Your Documents")

    # Info bar
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.caption(f" **Collection:** {st.session_state.collection_name}")
    with col2:
        st.caption(f" **Mode:** {st.session_state.agent_mode}")
    with col3:
        st.caption(f" **Messages:** {len(st.session_state.messages)}")

    st.markdown("---")

    # Chat container
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.info(
                "👋 **Welcome to DocuMind-Agent!**\n\n"
                "• Upload documents in the sidebar\n"
                "• Ask questions about your documents\n"
                "• Get AI-powered answers with sources"
            )

        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Show metadata if available
                if "metadata" in msg and msg["role"] == "assistant":
                    with st.expander("📊 Details", expanded=False):
                        metadata = msg["metadata"]

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{metadata.get('confidence', 0):.0%}")
                        with col2:
                            st.metric("Iterations", metadata.get("iterations_used", 0))
                        with col3:
                            st.metric("Documents", metadata.get("relevant_documents", 0))

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        if not _BACKEND_READY:
            st.error("❌ Backend not ready")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Get agent (lazy init)
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                try:
                    # Initialize agent if needed
                    with st.spinner("🔄 Initializing agent..."):
                        agent = get_agent_for_mode(st.session_state.agent_mode)

                    if not agent:
                        error_msg = st.session_state.last_error or "Failed to initialize agent"
                        message_placeholder.error(f"❌ {error_msg}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"❌ **Error:** {error_msg}"}
                        )
                        return

                    # Generate response
                    with st.spinner("🤔 Thinking..."):
                        response = agent.invoke(prompt)

                    answer = response.get("answer", "No answer generated.")
                    confidence = response.get("confidence", 0.0)
                    iterations = response.get("iterations_used", 0)
                    relevant_docs = len(response.get("relevant_documents", []))

                    # Display answer
                    message_placeholder.markdown(answer)

                    # Store message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "metadata": {
                                "confidence": confidence,
                                "iterations_used": iterations,
                                "relevant_documents": relevant_docs,
                                "timestamp": datetime.now().isoformat(),
                            },
                        }
                    )

                    # Show details
                    with st.expander("📊 Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{confidence:.0%}")
                        with col2:
                            st.metric("Iterations", iterations)
                        with col3:
                            st.metric("Documents", relevant_docs)

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())

                    message_placeholder.error(f"❌ {error_msg}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ {error_msg}"}
                    )


# ==================== MAIN APP ====================


def main():
    """Main application entry point."""
    # Initialize
    init_session_state()

    # Apply CSS
    st.markdown(get_modern_style(st.session_state.theme), unsafe_allow_html=True)

    # Render UI
    render_sidebar()
    render_main_chat()


if __name__ == "__main__":
    main()
