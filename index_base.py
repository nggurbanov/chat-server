from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel
from llama_index.llms.deepinfra import DeepInfraLLM
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
import logging

chromadb_logger = logging.getLogger("chromadb")
chromadb_logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='\033[93m%(message)s\033[0m')

default_prompt = \
"""
Context information is below.
---------------------
{context_str}
---------------------
By using that information and that information only answer to the question: 
{query_str}
Your answer should be short, concise and helpful.
Do not use any knowledge outside of the context provided.
Response should not be lengthy unless specifically instructed so.
"""

class RAG():

    def __init__(self, token, llm=None, prompt=None):
        self.token = token
        self.EMBED_MODEL = DeepInfraEmbeddingModel(
            model_id="intfloat/e5-large-v2",
            api_token=self.token,
            normalize=True,
            text_prefix="text: ",
            query_prefix="query: ")
        
        self.llm = llm if llm else DeepInfraLLM(
            model="microsoft/WizardLM-2-8x22B",
            api_key=self.token,
            temperature=0.5,
            max_tokens=1024,
            additional_kwargs={"top_p": 0.9},)
        
        if prompt:
            self.prompt = PromptTemplate(prompt)
        else: 
            self.prompt = PromptTemplate(default_prompt)
    
    @staticmethod
    async def load_docs(path):
        logging.info("Documents are loaded")
        return SimpleDirectoryReader(path).load_data()

    @staticmethod
    def set_vector_store(path):
        db = chromadb.PersistentClient(path=path)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
    
    async def load_index_db(self, path):
        logging.info("Embedding loading started")
        vector_store = RAG.set_vector_store(path)
        index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=self.EMBED_MODEL,)
        logging.info("Embedding loading is complete!")
        return index
        
    async def create_index_db(self, path, documents):
        vector_store = RAG.set_vector_store(path)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logging.info("Embedding process started")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.EMBED_MODEL)
        logging.info("Embedding process is complete\n")
        return index
        
    async def basic_query(self, index, question):
        query_engine = index.as_query_engine(llm=self.llm)
        response = query_engine.query(question)
        return response.response
    
    async def retrieve(self, index, question, top_k=1):
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

        retrieved_docs = retriever.retrieve(question)
        retrieved_text = "\n".join([doc.node.text for doc in retrieved_docs])        
        
        return retrieved_text


    