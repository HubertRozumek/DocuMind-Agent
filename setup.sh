#!/bin/bash
# DocuMind-Agent Setup Script
# This script sets up the proper directory structure for the project

set -e  # Exit on error

echo "🧠 DocuMind-Agent Setup Script"
echo "========================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "ℹ $1"
}

# Check if running from project root
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the project root."
    exit 1
fi

print_info "Setting up DocuMind-Agent project structure..."
echo ""

# Step 1: Create directory structure
print_info "Step 1: Creating directory structure..."
mkdir -p src/agent/nodes
mkdir -p src/vector_store
mkdir -p src/document_processor
mkdir -p src/tools
mkdir -p data/vector_store/chroma
mkdir -p data/raw_documents
mkdir -p logs
mkdir -p tests
mkdir -p sample_documents
print_success "Directories created"

# Step 2: Move files to proper locations (if they exist)
print_info "Step 2: Organizing source files..."

# Move agent files
if [ -f "agent_builder.py" ]; then
    mv agent_builder.py src/agent/ 2>/dev/null || true
fi
if [ -f "edges.py" ]; then
    mv edges.py src/agent/ 2>/dev/null || true
fi
if [ -f "graph_state.py" ]; then
    mv graph_state.py src/agent/ 2>/dev/null || true
fi

# Move node files
for file in generator_node.py grader_node.py query_rewriter.py retriever_node.py robust_grader.py grader_model.py; do
    if [ -f "$file" ]; then
        mv "$file" src/agent/nodes/ 2>/dev/null || true
    fi
done

# Move vector store files
for file in chroma_db.py embeddings_manager.py; do
    if [ -f "$file" ]; then
        mv "$file" src/vector_store/ 2>/dev/null || true
    fi
done

# Move document processor files
for file in pdf_loader.py text_splitter.py; do
    if [ -f "$file" ]; then
        mv "$file" src/document_processor/ 2>/dev/null || true
    fi
done

# Move tool files
for file in document_tool.py ticket_checker.py; do
    if [ -f "$file" ]; then
        mv "$file" src/tools/ 2>/dev/null || true
    fi
done

print_success "Source files organized"

# Step 3: Create __init__.py files
print_info "Step 3: Creating package initialization files..."
touch src/__init__.py
touch src/agent/__init__.py
touch src/agent/nodes/__init__.py
touch src/vector_store/__init__.py
touch src/document_processor/__init__.py
touch src/tools/__init__.py
print_success "Package files created"

# Step 4: Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Step 4: Creating .env configuration file..."
    cat > .env << 'EOF'
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_GRADER_MODEL=phi3:mini
OLLAMA_GENERATOR_MODEL=llama3.1:8b
OLLAMA_REWRITER_MODEL=mistral:7b

# Vector Store
CHROMA_PERSIST_DIR=data/vector_store/chroma
CHROMA_DEFAULT_COLLECTION=documents
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Embedding Model
EMBEDDING_MODEL_TYPE=MPNET
EMBEDDING_CACHE_DIR=models/cache
EMBEDDING_DEVICE=cpu

# Agent Configuration
MAX_ITERATIONS=3
SEARCH_THRESHOLD=0.7
GRADER_CONFIDENCE_THRESHOLD=0.6
RETRIEVAL_TOP_K=5
ENABLE_TOOLS=true

# Document Processing
CHUNK_SIZE=400
CHUNK_OVERLAP=50
CHUNKING_STRATEGY=recursive
PDF_LOADER_TYPE=auto
MAX_FILE_SIZE_MB=50

# Streamlit Configuration
APP_TITLE=DocuMind-Agent
APP_THEME=light
APP_LAYOUT=wide

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Performance
ENABLE_QUERY_CACHE=true
CACHE_SIZE=100
BATCH_SIZE=100

# Security
ENABLE_AUTH=false
ALLOWED_ORIGINS=http://localhost:8501

# Testing
TEST_COLLECTION_NAME=test_collection
TEST_PERSIST_DIR=tests/data/vector_store/chroma

# Debug
DEBUG=false
EOF
    print_success ".env file created"
else
    print_warning ".env file already exists, skipping creation"
fi

# Step 5: Check Python version
print_info "Step 5: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_success "Python $python_version detected (>= 3.10 required)"
else
    print_error "Python $python_version detected, but >= 3.10 required"
    exit 1
fi

# Step 6: Create virtual environment
if [ ! -d "venv" ]; then
    print_info "Step 6: Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists, skipping creation"
fi

# Step 7: Install dependencies
print_info "Step 7: Installing dependencies..."
print_warning "This may take several minutes..."

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
if pip install -r requirements.txt --quiet; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Step 8: Check Ollama
print_info "Step 8: Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    print_success "Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        print_success "Ollama server is running"

        # Check for required models
        print_info "Checking for required models..."
        models=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || echo "")

        check_model() {
            if echo "$models" | grep -q "^$1"; then
                print_success "Model $1 is available"
                return 0
            else
                print_warning "Model $1 is not downloaded"
                return 1
            fi
        }

        missing_models=0
        check_model "phi3:mini" || ((missing_models++))
        check_model "llama3.1:8b" || ((missing_models++))
        check_model "mistral:7b" || ((missing_models++))

        if [ $missing_models -gt 0 ]; then
            echo ""
            print_warning "Some models are missing. Download them with:"
            echo "  ollama pull phi3:mini"
            echo "  ollama pull llama3.1:8b"
            echo "  ollama pull mistral:7b"
        fi
    else
        print_warning "Ollama server is not running. Start it with: ollama serve"
    fi
else
    print_error "Ollama is not installed"
    echo ""
    echo "Install Ollama from: https://ollama.ai"
    echo "After installation, download the required models:"
    echo "  ollama pull phi3:mini"
    echo "  ollama pull llama3.1:8b"
    echo "  ollama pull mistral:7b"
fi

# Step 9: Verify installation
print_info "Step 9: Verifying installation..."
if python3 -c "import streamlit; import langchain; import chromadb" 2>/dev/null; then
    print_success "Core dependencies verified"
else
    print_error "Some dependencies are missing"
    exit 1
fi

# Summary
echo ""
echo "════════════════════════════════════════"
print_success "Setup complete!"
echo "════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Download Ollama models (if not already done):"
echo "     ollama pull phi3:mini"
echo "     ollama pull llama3.1:8b"
echo "     ollama pull mistral:7b"
echo ""
echo "  3. Start the application:"
echo "     streamlit run app.py"
echo ""
echo "  4. Or run tests:"
echo "     pytest tests/ -v"
echo ""
echo "📚 For more information, see README.md"
echo ""
