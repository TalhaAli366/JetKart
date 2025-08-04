import streamlit as st
import requests
import json
import time
import os
import subprocess
import threading
import sys
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="JetKart - Hybrid RAG System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .search-result {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .text-content {
        color: #2c3e50 !important;
        background-color: #f8f9fa !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    .info-content {
        background-color: #e3f2fd !important;
        color: #1565c0 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bbdefb;
    }
    
    .success-content {
        background-color: #e8f5e8 !important;
        color: #2e7d32 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c8e6c9;
    }
    
    .warning-content {
        background-color: #fff3e0 !important;
        color: #ef6c00 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffcc02;
    }
    
    .error-content {
        background-color: #ffebee !important;
        color: #c62828 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffcdd2;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Global variable to track server process
server_process = None

def start_api_server():
    """Start the FastAPI server in a separate thread."""
    global server_process
    
    def run_server():
        try:
            # Change to the project directory
            project_dir = Path(__file__).parent
            os.chdir(project_dir)
            
            # Start the API server
            server_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "src.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload",
                "--loop", "asyncio"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for the process to complete (it won't unless stopped)
            server_process.wait()
            
        except Exception as e:
            st.error(f"Error starting API server: {e}")
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a bit for the server to start
    time.sleep(3)
    
    return server_process

def stop_api_server():
    """Stop the FastAPI server if it's running."""
    global server_process
    if server_process and server_process.poll() is None:
        server_process.terminate()
        server_process.wait()
        server_process = None

def check_api_connection():
    """Check if the FastAPI server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_collection(collection_name: str) -> Dict[str, Any]:
    """Create a new vector collection."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/create-collection",
            json={"collection_name": collection_name},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def ingest_data(filename: str, file_type: str, collection_name: str) -> Dict[str, Any]:
    """Ingest data into the vector store."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={
                "filename": filename,
                "file_type": file_type,
                "collection_name": collection_name
            },
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_with_langgraph(query: str, collection_name: str) -> Dict[str, Any]:
    """Search using LangGraph agent."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "collection_name": collection_name
            },
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_available_files():
    """Get list of available files in the data directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    files = []
    for file_path in data_dir.rglob("*"):
        if file_path.is_file():
            # Determine file type
            if file_path.suffix.lower() == '.json':
                file_type = 'json'
            elif file_path.suffix.lower() in ['.md', '.markdown']:
                file_type = 'markdown'
            elif file_path.suffix.lower() in ['.txt', '.text']:
                file_type = 'text'
            else:
                continue
            
            files.append({
                'path': str(file_path),
                'name': file_path.name,
                'type': file_type,
                'size': file_path.stat().st_size
            })
    return files

def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ JetKart Hybrid RAG System</h1>', unsafe_allow_html=True)
    
    # Check API connection
    if not check_api_connection():
        st.warning("⚠️ Cannot connect to JetKart API server. Starting server automatically...")
        
        with st.spinner("Starting API server..."):
            start_api_server()
        
        # Wait and check again
        time.sleep(5)
        if not check_api_connection():
            st.markdown('<div class="error-content">❌ Failed to start API server automatically</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-content">💡 Please start the server manually with: `python src/main.py`</div>', unsafe_allow_html=True)
            return
        else:
            st.markdown('<div class="success-content">✅ API server started successfully!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-content">✅ Connected to JetKart API server</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Server status indicator
    if check_api_connection():
        st.sidebar.success("🟢 API Server: Running")
    else:
        st.sidebar.error("🔴 API Server: Stopped")
    
    # Server control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Restart Server"):
            stop_api_server()
            time.sleep(2)
            with st.spinner("Restarting server..."):
                start_api_server()
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Server"):
            stop_api_server()
            st.rerun()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🗄️ Vector Store", "📁 Data Ingestion", "🔍 Search", "📊 Analytics"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🗄️ Vector Store":
        show_vector_store()
    elif page == "📁 Data Ingestion":
        show_data_ingestion()
    elif page == "🔍 Search":
        show_search()
    elif page == "📊 Analytics":
        show_analytics()

def show_dashboard():
    """Show the main dashboard."""
    st.markdown('<h2 class="sub-header">🚀 Welcome to JetKart</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="text-content">
            <h3 style="color: #2c3e50; text-align: center;">🎯 Hybrid RAG</h3>
            <p style="color: #2c3e50; text-align: center;">Advanced retrieval with query classification, dynamic filtering, and LLM reranking</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="text-content">
            <h3 style="color: #2c3e50; text-align: center;">🧠 LangGraph</h3>
            <p style="color: #2c3e50; text-align: center;">Sophisticated workflow orchestration with intelligent routing and processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="text-content">
            <h3 style="color: #2c3e50; text-align: center;">🔍 Vector Search</h3>
            <p style="color: #2c3e50; text-align: center;">High-performance vector storage with Qdrant and Gemini embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown('<h3>🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-content">
    1. **Create Vector Store**: Go to the Vector Store page to create a new collection
    2. **Ingest Data**: Use the Data Ingestion page to upload and process your files
    3. **Search**: Use the Search page to query your data with the LangGraph agent
    4. **Monitor**: Check Analytics to see system performance and usage
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    st.markdown('<h3>📊 System Status</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="success-content">✅ API Server: Running</div>', unsafe_allow_html=True)
        st.markdown('<div class="success-content">✅ Vector Store: Available</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-content">ℹ️ LangGraph: Ready</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-content">ℹ️ Embeddings: Gemini text-embedding-004</div>', unsafe_allow_html=True)

def show_vector_store():
    """Show vector store management."""
    st.markdown('<h2 class="sub-header">🗄️ Vector Store Management</h2>', unsafe_allow_html=True)
    
    # Create new collection
    st.markdown('<h3>➕ Create New Collection</h3>', unsafe_allow_html=True)
    
    with st.form("create_collection_form"):
        collection_name = st.text_input(
            "Collection Name",
            placeholder="e.g., flights_data, travel_policies",
            help="Enter a unique name for your vector collection"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            vector_size = st.number_input("Vector Size", value=768, disabled=True)
        with col2:
            embedding_model = st.text_input("Embedding Model", value="Gemini text-embedding-004", disabled=True)
        
        submitted = st.form_submit_button("Create Collection", type="primary")
        
        if submitted:
            if collection_name:
                with st.spinner("Creating collection..."):
                    result = create_collection(collection_name)
                    
                if result.get("success"):
                    st.markdown(f'<div class="success-content">✅ Collection \'{collection_name}\' created successfully!</div>', unsafe_allow_html=True)
                    st.json(result)
                else:
                    st.markdown(f'<div class="error-content">❌ Failed to create collection: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-content">Please enter a collection name</div>', unsafe_allow_html=True)

def show_data_ingestion():
    """Show data ingestion interface."""
    st.markdown('<h2 class="sub-header">📁 Data Ingestion</h2>', unsafe_allow_html=True)
    
    # File selection
    st.markdown('<h3>📂 Select File to Ingest</h3>', unsafe_allow_html=True)
    
    available_files = get_available_files()
    
    if not available_files:
        st.markdown('<div class="warning-content">⚠️ No files found in the data directory. Please add files to the \'data\' folder.</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-content">Supported formats: JSON, Markdown (.md), Text (.txt)</div>', unsafe_allow_html=True)
        return
    
    # File selection interface
    selected_file = st.selectbox(
        "Choose a file:",
        options=available_files,
        format_func=lambda x: f"{x['name']} ({x['type'].upper()}, {x['size']} bytes)"
    )
    
    if selected_file:
        st.markdown(f'<div class="info-content">Selected: {selected_file["name"]} ({selected_file["type"]})</div>', unsafe_allow_html=True)
        
        # Collection selection
        collection_name = st.text_input(
            "Target Collection Name",
            value="default_collection",
            help="Enter the collection name where you want to store this data"
        )
        
        # Preview file content
        if st.checkbox("Preview file content"):
            try:
                with open(selected_file['path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 1000:
                        st.text_area("File Preview (first 1000 chars):", content[:1000] + "...")
                    else:
                        st.text_area("File Preview:", content)
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Ingest button
        if st.button("🚀 Ingest Data", type="primary"):
            if collection_name:
                with st.spinner("Ingesting data..."):
                    result = ingest_data(
                        filename=selected_file['path'],
                        file_type=selected_file['type'],
                        collection_name=collection_name
                    )
                
                if result.get("success"):
                    st.markdown(f'<div class="success-content">✅ Successfully ingested {result.get("documents_processed", 0)} documents!</div>', unsafe_allow_html=True)
                    st.json(result)
                else:
                    st.markdown(f'<div class="error-content">❌ Ingestion failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-content">Please enter a collection name</div>', unsafe_allow_html=True)

def show_search():
    """Show search interface."""
    st.markdown('<h2 class="sub-header">🔍 LangGraph Search</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-content">
        <strong>💡 Search Capabilities:</strong><br>
        • <strong>Flight Queries:</strong> "Emirates business class flights to Dubai under $2000"<br>
        • <strong>Information Queries:</strong> "What are the refund policies for cancelled flights?"<br>
        • <strong>Mixed Queries:</strong> "Flights to Japan and visa requirements for US citizens"
    </div>
    """, unsafe_allow_html=True)
    
    # Search interface
    collection_name = st.text_input(
        "Collection Name",
        value="default_collection",
        help="Enter the collection name to search in"
    )
    
    query = st.text_area(
        "Search Query",
        placeholder="Enter your search query here...",
        height=100,
        help="Ask questions about flights, travel policies, or any travel-related information"
    )
    
    # Search options
    col1, col2 = st.columns(2)
    with col1:
        show_filters = st.checkbox("Show Applied Filters", value=True)
    with col2:
        show_metrics = st.checkbox("Show Processing Metrics", value=True)
    
    # Search button
    if st.button("🔍 Search with LangGraph", type="primary"):
        if query and collection_name:
            with st.spinner("Searching with LangGraph agent..."):
                start_time = time.time()
                result = search_with_langgraph(query, collection_name)
                search_time = time.time() - start_time
            
            if result.get("success"):
                st.markdown('<div class="success-content">✅ Search completed successfully!</div>', unsafe_allow_html=True)
                
                # Display results
                st.markdown('<div class="search-result">', unsafe_allow_html=True)
                st.markdown(f"**🤖 Generated Answer:**")
                st.markdown(result.get("answer", "No answer generated"))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show additional information
                if show_filters or show_metrics:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if show_filters and result.get("filters_applied"):
                            st.markdown("**🔧 Applied Filters:**")
                            st.json(result.get("filters_applied"))
                        elif show_filters:
                            st.markdown('<div class="info-content">No filters were applied to this query</div>', unsafe_allow_html=True)
                    
                    with col2:
                        if show_metrics:
                            st.markdown("**📊 Processing Metrics:**")
                            metrics_data = {
                                "Query Type": result.get("query_type", "unknown"),
                                "Documents Used": result.get("documents_used", 0),
                                "Processing Time": f"{result.get('processing_time', 0):.2f}s",
                                "Total Time": f"{search_time:.2f}s"
                            }
                            st.json(metrics_data)
                
                # Show full response for debugging
                with st.expander("🔍 View Full Response"):
                    st.json(result)
            else:
                st.markdown(f'<div class="error-content">❌ Search failed: {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-content">Please enter both a query and collection name</div>', unsafe_allow_html=True)

def show_analytics():
    """Show analytics and system information."""
    st.markdown('<h2 class="sub-header">📊 Analytics & System Info</h2>', unsafe_allow_html=True)
    
    # System information
    st.markdown('<h3>⚙️ System Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="text-content">
            <h4 style="color: #2c3e50; text-align: center;">🔧 API Status</h4>
            <p style="color: #2c3e50; text-align: center;">✅ Running</p>
            <p style="color: #2c3e50; text-align: center;">Port: 8000</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="text-content">
            <h4 style="color: #2c3e50; text-align: center;">🧠 LangGraph</h4>
            <p style="color: #2c3e50; text-align: center;">✅ Active</p>
            <p style="color: #2c3e50; text-align: center;">Workflow: Hybrid RAG</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="text-content">
            <h4 style="color: #2c3e50; text-align: center;">🗄️ Vector Store</h4>
            <p style="color: #2c3e50; text-align: center;">✅ Qdrant</p>
            <p style="color: #2c3e50; text-align: center;">Embeddings: Gemini</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="text-content">
            <h4 style="color: #2c3e50; text-align: center;">🔍 Search Engine</h4>
            <p style="color: #2c3e50; text-align: center;">✅ Hybrid</p>
            <p style="color: #2c3e50; text-align: center;">Dense + Sparse</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Architecture diagram
    st.markdown('<h3>🏗️ System Architecture</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    ```mermaid
    graph TD
        A[User Query] --> B[Query Classification]
        B --> C{Query Type}
        C -->|Flight| D[Generate Filters]
        C -->|Info| E[Hybrid Retrieval]
        C -->|Both| F[Both Paths]
        D --> G[Apply Hard Filters]
        G --> H[LLM Reranker]
        E --> I[Merge Documents]
        F --> I
        H --> I
        I --> J[Generate Answer]
        J --> K[Response]
    ```
    """)
    
    # Features overview
    st.markdown('<h3>✨ Key Features</h3>', unsafe_allow_html=True)
    
    features = [
        "🎯 **Intelligent Query Classification**: Automatically determines query type (flight/info/both)",
        "🔧 **Dynamic Filter Generation**: LLM-powered filter creation for precise results",
        "🎛️ **Hard Filtering**: Metadata-based document filtering",
        "🧠 **LLM Reranking**: GPT-4o-mini powered document reranking",
        "🔍 **Hybrid Retrieval**: Combines dense and sparse vector search",
        "📄 **Document Merging**: Intelligent combination of results from different paths",
        "⚡ **Async Processing**: Full async/await support for better performance",
        "🛡️ **Robust Fallbacks**: Multiple fallback strategies ensure reliability"
    ]
    
    st.markdown('<div class="text-content">', unsafe_allow_html=True)
    for feature in features:
        st.markdown(f"• {feature}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup: stop the server when the app is closed
        stop_api_server() 