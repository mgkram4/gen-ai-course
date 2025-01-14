import CodeBlock from '../components/CodeBlock';
import Layout from '../layout';

export default function Unit3() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 3: Advanced LLM Concepts</h1>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. Retrieval Augmented Generation (RAG)</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">
              RAG combines the power of retrieval systems with generative AI to produce more accurate and contextual responses.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div className="bg-white p-4 rounded-lg shadow">
                <h4 className="font-bold mb-2">Core Components</h4>
                <ul className="list-disc list-inside text-gray-700">
                  <li>Document Processing</li>
                  <li>Vector Embeddings</li>
                  <li>Similarity Search</li>
                  <li>Context Integration</li>
                </ul>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <h4 className="font-bold mb-2">Key Benefits</h4>
                <ul className="list-disc list-inside text-gray-700">
                  <li>Up-to-date Information</li>
                  <li>Reduced Hallucination</li>
                  <li>Domain Adaptation</li>
                  <li>Source Attribution</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Document loading and processing</li>
                <li>Vector embedding generation</li>
                <li>Similarity search implementation</li>
                <li>Query processing and response generation</li>
              </ul>
            </div>
            <h4 className="text-xl mb-2 text-blue-300">RAG Implementation</h4>
            <CodeBlock code={`
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class RAGSystem:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
    
    def load_documents(self, file_path: str):
        """Load and process documents."""
        loader = TextLoader(file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
    
    def query(self, question: str, k: int = 3) -> list:
        """Retrieve relevant documents for a query."""
        if not self.vector_store:
            raise ValueError("No documents loaded")
        
        return self.vector_store.similarity_search(question, k=k)

# Example usage
rag = RAGSystem('your-api-key')
rag.load_documents('knowledge_base.txt')
relevant_docs = rag.query("What are the benefits of RAG?")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. Vector Databases and Embeddings</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Vector Database Types</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>FAISS (Facebook AI Similarity Search)</li>
                <li>Pinecone</li>
                <li>Weaviate</li>
                <li>Milvus</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Key Features</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>Efficient similarity search</li>
                <li>Scalable vector storage</li>
                <li>Real-time updates</li>
                <li>Multi-modal support</li>
              </ul>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Understanding Vector Embeddings</h4>
            <p className="mb-4">
              Vector embeddings are numerical representations of text that capture semantic meaning:
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>High-dimensional vectors (typically 768-1536 dimensions)</li>
              <li>Similar meanings have similar vector representations</li>
              <li>Enable efficient semantic search</li>
              <li>Can be used for clustering and classification</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Working with Embeddings</h3>
            <CodeBlock code={`
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import numpy as np

class EmbeddingSystem:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vector_store = None
    
    def create_embeddings(self, texts: list[str]):
        """Generate embeddings for a list of texts."""
        return self.embeddings.embed_documents(texts)
    
    def build_vector_store(self, texts: list[str]):
        """Create a FAISS vector store from texts."""
        self.vector_store = FAISS.from_texts(
            texts,
            self.embeddings
        )
    
    def similarity_search(self, query: str, k: int = 5):
        """Find similar texts using vector similarity."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)

# Example usage
system = EmbeddingSystem('your-api-key')
texts = [
    "Vector databases store high-dimensional vectors",
    "Embeddings capture semantic meaning",
    "FAISS is optimized for similarity search"
]
system.build_vector_store(texts)
results = system.similarity_search("What are vector databases?")
            `} language="python" />
          </div>

          
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. Fine-tuning Strategies</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Fine-tuning Methods</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>Full Fine-tuning</li>
                <li>Parameter-Efficient Fine-tuning (PEFT)</li>
                <li>LoRA (Low-Rank Adaptation)</li>
                <li>Prompt Tuning</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Key Considerations</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>Dataset quality and size</li>
                <li>Computational resources</li>
                <li>Model architecture</li>
                <li>Evaluation metrics</li>
              </ul>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Fine-tuning Best Practices</h4>
            <p className="mb-4">
              Effective fine-tuning requires careful consideration of several factors:
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Data preparation and cleaning</li>
              <li>Hyperparameter optimization</li>
              <li>Validation strategies</li>
              <li>Preventing catastrophic forgetting</li>
            </ul>
            
            <div className="mt-4">
              <h4 className="font-bold mb-2">Common Applications:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Domain adaptation</li>
                <li>Task-specific optimization</li>
                <li>Style transfer</li>
                <li>Bias mitigation</li>
              </ul>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Implementation Example</h3>
            <CodeBlock code={`
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch

class ModelFinetuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def prepare_dataset(self, texts, labels):
        """Prepare dataset for fine-tuning."""
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })
    
    def train(self, train_dataset, eval_dataset=None):
        """Fine-tune the model."""
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            learning_rate=5e-5,
            logging_dir="./logs",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        return trainer.train()
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Advanced Memory Management</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Memory Types</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>Conversation Buffer</li>
                <li>Summary Buffer</li>
                <li>Vector Store</li>
                <li>Entity Memory</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Management Strategies</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>Token-based pruning</li>
                <li>Importance sampling</li>
                <li>Hierarchical summarization</li>
                <li>Selective retention</li>
              </ul>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Memory Management Techniques</h4>
            <p className="mb-4">
              Effective memory management is crucial for maintaining context in long conversations:
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Dynamic context window adjustment</li>
              <li>Intelligent message summarization</li>
              <li>Priority-based memory retention</li>
              <li>Entity tracking and relationship mapping</li>
            </ul>
            
            <div className="mt-4">
              <h4 className="font-bold mb-2">Implementation Considerations:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Token limit management</li>
                <li>Memory persistence strategies</li>
                <li>Context relevance scoring</li>
                <li>Memory compression techniques</li>
              </ul>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Memory Management Implementation</h3>
            <CodeBlock code={`
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from typing import List, Dict

class AdvancedMemoryManager:
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.memory = ConversationBufferMemory()
        self.summary_buffer = []
        
    def add_interaction(self, user_input: str, ai_response: str):
        """Add new interaction to memory with token management."""
        self.memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
        
        # Manage memory size
        if self._get_token_count() > self.max_tokens:
            self._summarize_and_prune()
    
    def _get_token_count(self) -> int:
        """Estimate token count in current memory."""
        # Implementation for token counting
        pass
    
    def _summarize_and_prune(self):
        """Summarize old memories and remove them from active memory."""
        # Implementation for summarization
        pass
    
    def get_relevant_context(self, query: str) -> str:
        """Retrieve relevant context based on query."""
        return self.memory.load_memory_variables({})["history"]
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">5. Context Window Optimization</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Optimization Techniques</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>Sliding window approach</li>
                <li>Chunk overlap strategies</li>
                <li>Dynamic resizing</li>
                <li>Content prioritization</li>
              </ul>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <h4 className="font-bold mb-2">Performance Metrics</h4>
              <ul className="list-disc list-inside text-gray-700">
                <li>Response latency</li>
                <li>Context relevance</li>
                <li>Memory efficiency</li>
                <li>Token utilization</li>
              </ul>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Window Optimization Strategies</h4>
            <p className="mb-4">
              Optimizing context windows requires balancing several factors:
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Content relevance scoring</li>
              <li>Adaptive chunk sizing</li>
              <li>Intelligent content selection</li>
              <li>Memory-performance tradeoffs</li>
            </ul>
            
            <div className="mt-4">
              <h4 className="font-bold mb-2">Best Practices:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Regular context pruning</li>
                <li>Dynamic window resizing</li>
                <li>Content prioritization</li>
                <li>Performance monitoring</li>
              </ul>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Implementation Example</h3>
            <CodeBlock code={`
class ContextWindowOptimizer:
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.tokenizer = None  # Initialize with your tokenizer
        
    def optimize_context(self, content: str, query: str = None) -> str:
        """Optimize content to fit within context window."""
        tokens = self._count_tokens(content)
        
        if tokens <= self.max_tokens:
            return content
            
        return self._apply_optimization_strategy(content, query)
    
    def _apply_optimization_strategy(self, content: str, query: str = None) -> str:
        """Apply various optimization strategies based on content and query."""
        if query:
            # Use query-focused compression
            return self._query_focused_compression(content, query)
        else:
            # Use general compression
            return self._sliding_window_compression(content)
    
    def _query_focused_compression(self, content: str, query: str) -> str:
        """Compress content while maintaining query-relevant parts."""
        # Implementation for query-focused compression
        pass
    
    def _sliding_window_compression(self, content: str) -> str:
        """Apply sliding window compression strategy."""
        # Implementation for sliding window compression
        pass
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Conclusion and Next Steps</h2>
          <div className="mb-6">
            <p className="mb-4 text-gray-700">
              We&apos;ve covered advanced concepts in working with Language Models:
            </p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Understanding and implementing RAG systems</li>
              <li>Working with vector databases and embeddings</li>
              <li>Applying fine-tuning strategies</li>
              <li>Managing memory effectively</li>
              <li>Optimizing context windows</li>
            </ul>

            <p className="mb-4 text-gray-700">
              Moving forward, consider exploring:
            </p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Advanced RAG architectures</li>
              <li>Custom embedding models</li>
              <li>Hybrid search systems</li>
              <li>Production deployment strategies</li>
              <li>Performance optimization techniques</li>
            </ul>

            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-bold mb-2">Practical Exercises</h4>
              <ol className="list-decimal list-inside space-y-2">
                <li>Build a RAG system with custom document processing</li>
                <li>Implement a vector database from scratch</li>
                <li>Create a fine-tuning pipeline</li>
                <li>Develop an advanced memory management system</li>
                <li>Optimize context windows for specific use cases</li>
              </ol>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}