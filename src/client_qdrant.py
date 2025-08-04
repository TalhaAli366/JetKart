import os
import asyncio
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance
from langchain_qdrant import FastEmbedSparse, RetrievalMode, QdrantVectorStore
from typing import Optional

logger = logging.getLogger(__name__)

def get_qdrant_client(timeout: int = 30):
    qdrant_url = os.getenv("QDRANT_CLOUD")
    return QdrantClient(
        url=qdrant_url,
        api_key=os.getenv("QDRANT_CLOUD_KEY"),
        port=None,
        prefer_grpc=False,
        timeout=timeout
    )

async def initialize_vector_store(
    client: QdrantClient,
    collection_name: str,
    embedding_model,
    sparse_model: str = "Qdrant/bm25"
) -> Optional[QdrantVectorStore]:
    """
    Initialize the vector store with the given parameters.
    
    Args:
        client: Initialized Qdrant client
        collection_name: Name of the collection to use
        embedding_model: Gemini embedding model instance
        sparse_model: Name of the sparse embedding model
        
    Returns:
        QdrantVectorStore or None: Initialized vector store if successful, None otherwise
    """
    try:
        vector_store = await asyncio.to_thread(
            QdrantVectorStore,
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
            sparse_embedding=FastEmbedSparse(model_name=sparse_model),
            sparse_vector_name = "default",
            retrieval_mode=RetrievalMode.HYBRID
        )
        logger.info(f"Successfully initialized vector store for collection: {collection_name}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return None

async def create_qdrant_collection(
    collection_name: str,
    client: QdrantClient,
    vector_size: int,
) -> None:
    """
    Create a new collection in Qdrant. If collection exists, it will be deleted and recreated.
    
    Args:
        collection_name: Name of the collection to create
        client: Initialized Qdrant client
        vector_size: Size of the vectors
    """
    try:
        collections = await asyncio.to_thread(client.get_collections)
        logger.info(f"Existing collections: {collections}")
        
        if any(collection.name == collection_name for collection in collections.collections):
            await asyncio.to_thread(client.delete_collection, collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        
        await asyncio.to_thread(
            client.create_collection,
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            ),
            sparse_vectors_config={
                "default": SparseVectorParams()
            }
        )
        logger.info(f"Successfully created collection: {collection_name}")
        
        # Create payload indexes for filtering
        await create_filter_indexes(client, collection_name)
        
    except Exception as e:
        logger.error(f"Error in collection creation: {str(e)}")
        raise


async def create_filter_indexes(client: QdrantClient, collection_name: str) -> None:
    """
    Create payload indexes for fields that will be used in filtering.
    
    Args:
        client: Initialized Qdrant client
        collection_name: Name of the collection
    """
    try:
        # Define the fields we want to index for filtering
        filter_fields = [
            ("document_type", "keyword"),
            ("airline", "keyword"),
            ("alliance", "keyword"),
            ("from_country", "keyword"),
            ("to_country", "keyword"),
            ("travel_class", "keyword"),
            ("price_usd", "integer"),
            ("refundable", "bool"),
            ("baggage_included", "bool"),
            ("wifi_available", "bool"),
            ("meal_service", "keyword"),
            ("aircraft_type", "keyword"),
        ]
        
        for field_name, field_type in filter_fields:
            try:
                await asyncio.to_thread(
                    client.create_payload_index,
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created index for field: {field_name} ({field_type})")
            except Exception as e:
                logger.warning(f"Failed to create index for field {field_name}: {str(e)}")
                # Continue with other fields even if one fails
        
        logger.info(f"Successfully created filter indexes for collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Error creating filter indexes: {str(e)}")
        raise

async def ensure_filter_indexes(client: QdrantClient, collection_name: str) -> None:
    """
    Ensure that payload indexes exist for fields that will be used in filtering.
    This function can be called for existing collections that may not have the necessary indexes.
    
    Args:
        client: Initialized Qdrant client
        collection_name: Name of the collection
    """
    try:
        # Check if collection exists
        collections = await asyncio.to_thread(client.get_collections)
        collection_exists = any(collection.name == collection_name for collection in collections.collections)
        
        if not collection_exists:
            logger.warning(f"Collection {collection_name} does not exist, cannot create indexes")
            return
        
        # Create indexes for the collection
        await create_filter_indexes(client, collection_name)
        
    except Exception as e:
        logger.error(f"Error ensuring filter indexes: {str(e)}")
        raise
