from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_qdrant.qdrant import QdrantVectorStore
from qdrant_client.http import models
from qdrant_client.http.models import Distance
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed



class VectorDatabaseManager:
    """
    embedding_model: Path of the pre-trained Huggingface embedding model used for generating dense embeddings;
    sparse_model: Path of the model used to produce sparse embeddings. 
                  The default value is specified as 'Qdrant/bm25', which is a common choice for sparse representations;
    metric: The distance metric employed for comparing vectors. 
            The default metric is 'cosine', the user can choose among four metrics ['cosine', 'euclid', 'dot', 'manhattan'];
    device: The computing device used for model inference. 
            It can be set to ['cpu', 'cuda', 'mps'];
    location: The URL endpoint for the Qdrant service.
    threads: The number of threads to utilize for concurrent processing of requests. 
             The default is 4;
    """

    def __init__(self, 
                 embedding_model: str,
                 sparse_model: str = 'Qdrant/bm25',
                 metric: str = 'cosine', 
                 device: str = 'cpu', 
                 location: str = 'http://localhost:6333',
                 threads: int = 4):
        
        print("Loading Embedding Models ......")
        self.dense_embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.sparse_embeddings = FastEmbedSparse(model_name=sparse_model)
        print("Embedding Models Loaded ......")
    
        distance_mapping = {
            'cosine': Distance.COSINE,
            'euclid': Distance.EUCLID,
            'dot': Distance.DOT,
            'manhattan': Distance.MANHATTAN,
        }

        self.distance = distance_mapping.get(metric)
        if self.distance is None:
            raise ValueError("Unknown metric, please chose: cosine, euclid, dot or manhattan!: {}".format(metric))
        

        self.client = QdrantClient(url=location, prefer_grpc=False)
        self.locaton = location
        self.metric = metric
        self.threads = threads


    def create_document(self, row: dict) -> Document:
            metadata_without_content = {key: value for key, value in row.items() if key != 'content'}
            return Document(page_content=row['content'], metadata=metadata_without_content)

    def create_vector_database(self, 
                               data: List[dict], 
                               collection_name: str): 
        '''
        data: List of dict data to be collected in the database.
              The 'content' field is the context to be embedded in the vector.
              The other fields are metadata;
        collection_name: Name of the collection in the vector database where vectors will be stored 
        '''
        
        print("Creating vectors...")
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            documents = list(executor.map(self.create_document, data))
        print("Vectors are created...")


        print('Qdrant creating ........')
        QdrantVectorStore.from_documents(
            documents,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            location=self.locaton,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            distance=self.distance
        )

        print('Qdrant is created ........')


    def vector_database_search(self, 
                               query: str, 
                               collection_name: str,
                               retrval: str, 
                               k: int = 4) -> List[dict]:
        '''
        query: The query you are sending to the vector database;
        collection_name: The name of the collection you want to search;
        retrval: The way you want to search.
                 The default is 'hybrid', the user can choose between ['hybrid', 'dense', 'sparse'];
        k: Number of best vectors to retrieve from the database
        '''

        retrival_mapping = {
            'dense': RetrievalMode.DENSE,
            'sparse': RetrievalMode.SPARSE,
            'hybrid': RetrievalMode.HYBRID,
        }

        retival = retrival_mapping.get(retrval)
        if retival is None:
            raise ValueError("Unknown metric, please chose: dense, sparse or hybrid!: {}".format(retrval))
            

        db = QdrantVectorStore(
            client = self.client,
            embedding = self.dense_embeddings,
            sparse_embedding = self.sparse_embeddings,
            collection_name = collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            distance = self.distance
        )
        docs = db.similarity_search_with_score(query=query, k=k)
            
        best_matches = []
        for item in docs:
            doc, score = item
            best_matches.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            })

        return best_matches


    def process_title_and_id(self, query, collection_name, retrval, k):
            best_match = self.vector_database_search(query, collection_name, retrval, k)
            return best_match, query
    

    def vector_database_search_list(self, 
                                    list_of_query: List, 
                                    collection_name: str,
                                    retrval: str, 
                                    k: int = 4) -> List:
        '''
        query: The query you are sending to the vector database;
        collection_name: The name of the collection you want to search;
        retrval: The way you want to search.
                 The default is 'hybrid', the user can choose between ['hybrid', 'dense', 'sparse'];
        k: Number of best vectors to retrieve from the database
        '''

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(self.process_title_and_id, query, collection_name, retrval, k): query for query in list_of_query}

            # Collect results as they are completed
            reults = []
            for i, future in enumerate(as_completed(futures)):
                try:
                    best_match, query = future.result()
                    reults.append({'query':query, 'retrival':best_match})
                    if i % 10 == 0:
                        print(f"Processed {i} queries.")
                except Exception as e:
                    print(f"Error processing item: {futures[future]}, error: {e}")
        return reults