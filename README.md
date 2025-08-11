# RAG Chatbot with DeepSeek R1 Distill Qwen 1.5B

A complete Retrieval-Augmented Generation (RAG) chatbot prototype using the distilled DeepSeek R1 model. This system handles multiple users with continuous chat sessions, demonstrating robust NLP capabilities and effective RAG implementation with complete local execution.

## Features

- **Local Execution**: All components run locally (LLM inference, embeddings, vector database, chat history)
- **Multi-User Support**: Handle multiple users with unique user IDs and session IDs
- **Continuous Sessions**: Maintain conversation context across sessions using local chat history storage
- **Clean Responses**: Output only final answers, excluding reasoning or intermediate thoughts
- **Modular Design**: Separate modules for RAG retrieval, LLM inference, and chat history management

## Architecture

### Core Components

1. **LLM Component**: Uses DeepSeek-R1-Distill-Qwen-1.5B model with Hugging Face Transformers
2. **Knowledge Base & RAG**: FAISS vector database with local embedding model (all-MiniLM-L6-v2)
3. **Chat History Management**: JSON-based local storage organized by user ID and session ID

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG-Chatbot-for-Multi-User
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Setup

The system will automatically:
- Download DeepSeek-R1-Distill-Qwen-1.5B model (~3GB) on first run
- Download all-MiniLM-L6-v2 embedding model (~90MB)
- Create necessary directories (`models/`, `data/vector_store/`, `data/chat_history/`)
- Process knowledge base documents and build vector indices

### How to Run the Chatbot

#### Interactive Mode (Recommended)
```bash
python main.py interactive
```

#### Demo Mode (Sample queries)
```bash
python main.py
```

### System Architecture & Design Choices

#### NLP Components
- **Language Model**: DeepSeek-R1-Distill-Qwen-1.5B with chat template optimization
- **Embeddings**: SentenceTransformers all-MiniLM-L6-v2 (384-dimensional vectors)
- **Vector Search**: FAISS IndexFlatL2 for exact similarity matching
- **Text Processing**: Automatic tokenization and context window management

#### RAG Pipeline Design
1. **Document Processing**: Text files are loaded and embedded automatically
2. **Query Processing**: User queries are embedded using the same model
3. **Retrieval**: Top-k similar documents retrieved via cosine similarity
4. **Context Assembly**: Retrieved content combined with conversation history
5. **Response Generation**: LLM generates contextually aware responses

## Usage

### Interactive Mode

Run the chatbot in interactive mode for real-time conversations:

```bash
python main.py interactive
```

### Quick Start Example

Here's how to start chatting:

```bash
$ python main.py interactive

# Step 1: Set your user ID
> /user john

# Step 2: Start a session
> /session

# Step 3: Ask questions and get responses
> What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It involves algorithms that can identify patterns in data and make predictions based on that data.

> How do neural networks work?
Assistant: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions...

# Use /quit to exit
> /quit
Goodbye!
```

### Available Commands
- `/user <user_id>` - Set user ID
- `/session [session_id]` - Start new session or specify session ID
- `/history` - Show conversation history
- `/sessions` - List user sessions
- `/info` - Show current session info
- `/quit` - Exit

### Demo Mode

Run with sample queries:

```bash
python main.py
```

## File Structure

```
├── app/                    # Main application modules
│   ├── model_handler.py    # LLM inference with DeepSeek R1 Distill Qwen 1.5B
│   ├── retriever.py        # RAG retrieval system
│   ├── vector_store.py     # FAISS vector database
│   ├── embedding_handler.py # Local embeddings
│   └── chat_manager.py     # Multi-user chat management
├── data/
│   ├── knowledge_base/     # Documents for RAG (6 sample documents included)
│   └── chat_history/       # User chat sessions
├── models/                 # Local model storage (created automatically)
├── main.py                 # Main chatbot application
└── requirements.txt        # Python dependencies
```

## Loading Sample Knowledge Base

### Included Documents
The system includes 6 sample documents in `data/knowledge_base/`:

1. **machine_learning.txt** - ML fundamentals, supervised/unsupervised/reinforcement learning
2. **neural_networks.txt** - Network architectures, deep learning concepts
3. **nlp_basics.txt** - NLP tasks, techniques, transformers, applications
4. **computer_vision.txt** - CV tasks, CNNs, applications, challenges
5. **transformers.txt** - Transformer architecture, attention mechanisms, LLMs
6. **rag_systems.txt** - RAG concepts, components, benefits, applications

### Loading Process
The knowledge base is automatically loaded when you run the chatbot:

1. **Automatic Loading**: Run `python main.py` - the system automatically:
   - Scans `data/knowledge_base/` for `.txt` files
   - Generates embeddings for each document
   - Builds FAISS vector index
   - Saves index to `data/vector_store/`

2. **Adding New Documents**: Simply add `.txt` files to `data/knowledge_base/` and restart
3. **Verification**: Check successful loading with `/info` command in interactive mode

### Vector Database Details
- **Index Type**: FAISS IndexFlatL2 (exact L2 distance search)
- **Embedding Dimension**: 384 (from all-MiniLM-L6-v2)
- **Storage**: Local files in `data/vector_store/` (faiss_index + metadata.json)
- **Persistence**: Automatically saves/loads between sessions

## System Requirements

### Hardware
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: Optional but recommended (NVIDIA GPU with 4GB+ VRAM for faster inference)
- **Storage**: 10GB+ free space for models and data

### Software
- Python 3.8+
- CUDA-compatible GPU drivers (optional, for GPU acceleration)

## Key Capabilities

- **Diverse Query Handling**: Factual, conversational, and ambiguous queries
- **Intent Recognition**: Robust understanding of user intentions
- **Response Generation**: Context-aware responses using RAG
- **Session Management**: Persistent conversations across multiple sessions

## Testing Multi-User, Multi-Session Functionality

### Multi-User Testing
Test the system with different users:

```bash
python main.py interactive

# User 1
> /user alice
> /session
> What is machine learning?
> /quit

# User 2
> /user hasib
> /session
> How do neural networks work?
> /sessions  # Shows only bob's sessions
> /quit
```

### Multi-Session Testing
Test session management for a single user:

```bash
# Start first session
> /user alice
> /session session_work
> What is deep learning?

# Start second session  
> /session session_research
> What is computer vision?

# Switch back to first session
> /session session_work
> /history  # Shows only session_work history

# List all sessions
> /sessions
```

### Session Isolation Verification
- Each user's chat history is stored separately in `data/chat_history/user_<userid>.json`
- Sessions within a user are isolated - `/history` shows only current session
- Users cannot access other users' sessions or history

### Conversational Context Tests

```bash
# Test conversation memory
> What is machine learning?
> What are its main types?  # Should reference previous ML context
> Give me examples of the supervised type  # Should maintain ML + supervised context
```

## Example Interactions

```
User: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It involves algorithms that can identify patterns in data and make predictions based on that data.

```

## Acknowledgments

- DeepSeek AI for the distilled R1 model
- Hugging Face for the Transformers library
- Facebook Research for FAISS vector search
- Sentence Transformers for embedding models