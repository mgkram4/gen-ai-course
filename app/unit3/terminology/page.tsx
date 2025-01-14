export default function Unit3Terminology() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 3: Advanced LLM Concepts Terminology</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">RAG Concepts</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Retrieval Augmented Generation (RAG)</h3>
            <p className="text-gray-700">A technique that combines information retrieval with language generation to produce more accurate and contextual responses by referencing external knowledge.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Document Chunking</h3>
            <p className="text-gray-700">The process of breaking down large documents into smaller, manageable pieces for efficient processing and retrieval in RAG systems.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Context Integration</h3>
            <p className="text-gray-700">The process of combining retrieved information with the LLM prompt to provide relevant context for generating responses.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Similarity Search</h3>
            <p className="text-gray-700">A method of finding relevant documents or text chunks based on their semantic similarity to a query.</p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">RAG Architecture Overview</h3>
          <svg viewBox="0 0 800 300" className="w-full max-w-3xl mx-auto">
            {/* Document Processing */}
            <rect x="50" y="50" width="200" height="80" rx="5" fill="#60A5FA" />
            <text x="150" y="85" textAnchor="middle" fill="white" className="font-bold">Document Processing</text>
            <rect x="70" y="95" width="160" height="25" rx="3" fill="#3B82F6" />
            <text x="150" y="112" textAnchor="middle" fill="white" className="text-xs">Document Chunking</text>

            {/* Vector Store */}
            <rect x="300" y="50" width="200" height="200" rx="5" fill="#34D399" />
            <text x="400" y="85" textAnchor="middle" fill="white" className="font-bold">Vector Store</text>
            <rect x="320" y="95" width="160" height="25" rx="3" fill="#2FB344" />
            <text x="400" y="112" textAnchor="middle" fill="white" className="text-xs">Embeddings</text>
            <rect x="320" y="130" width="160" height="25" rx="3" fill="#2FB344" />
            <text x="400" y="147" textAnchor="middle" fill="white" className="text-xs">Similarity Search</text>

            {/* LLM Integration */}
            <rect x="550" y="50" width="200" height="200" rx="5" fill="#F87171" />
            <text x="650" y="85" textAnchor="middle" fill="white" className="font-bold">LLM Integration</text>
            <rect x="570" y="95" width="160" height="25" rx="3" fill="#EF4444" />
            <text x="650" y="112" textAnchor="middle" fill="white" className="text-xs">Context Integration</text>
            <rect x="570" y="130" width="160" height="25" rx="3" fill="#EF4444" />
            <text x="650" y="147" textAnchor="middle" fill="white" className="text-xs">Response Generation</text>

            {/* Arrows */}
            <path d="M250 90 L300 90" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M500 90 L550 90" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />

            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#94A3B8" />
              </marker>
            </defs>
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 1:</strong> RAG system architecture showing document processing, vector storage, and LLM integration.
          </p>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Vector Databases and Embeddings</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Vector Embedding</h3>
            <p className="text-gray-700">A numerical representation of text in high-dimensional space that captures semantic meaning and enables similarity comparisons.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Vector Database</h3>
            <p className="text-gray-700">A specialized database designed to store and efficiently query high-dimensional vectors, optimized for similarity search operations.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Dimensionality</h3>
            <p className="text-gray-700">The number of values in a vector embedding, typically ranging from 768 to 1536 dimensions for modern language models.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Cosine Similarity</h3>
            <p className="text-gray-700">A metric used to measure the similarity between two vectors by calculating the cosine of the angle between them.</p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">Vector Database Architecture</h3>
          <svg viewBox="0 0 800 300" className="w-full max-w-3xl mx-auto">
            {/* Text Input */}
            <rect x="50" y="50" width="150" height="60" rx="5" fill="#60A5FA" />
            <text x="125" y="85" textAnchor="middle" fill="white">Text Input</text>

            {/* Embedding Process */}
            <rect x="250" y="30" width="300" height="100" rx="5" fill="#34D399" />
            <text x="400" y="60" textAnchor="middle" fill="white" className="font-bold">Embedding Process</text>
            <rect x="270" y="70" width="80" height="40" rx="3" fill="#2FB344" />
            <text x="310" y="95" textAnchor="middle" fill="white" className="text-xs">Tokenization</text>
            <rect x="360" y="70" width="80" height="40" rx="3" fill="#2FB344" />
            <text x="400" y="95" textAnchor="middle" fill="white" className="text-xs">Encoding</text>
            <rect x="450" y="70" width="80" height="40" rx="3" fill="#2FB344" />
            <text x="490" y="95" textAnchor="middle" fill="white" className="text-xs">Vector</text>

            {/* Vector Space */}
            <rect x="600" y="30" width="150" height="200" rx="5" fill="#F87171" />
            <text x="675" y="60" textAnchor="middle" fill="white">Vector Space</text>
            <circle cx="650" cy="120" r="5" fill="white" />
            <circle cx="680" cy="150" r="5" fill="white" />
            <circle cx="700" cy="180" r="5" fill="white" />
            <circle cx="670" cy="100" r="5" fill="white" />

            {/* Arrows */}
            <path d="M200 80 L250 80" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M550 80 L600 80" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 2:</strong> Vector database architecture showing embedding process and vector space representation.
          </p>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Fine-tuning Concepts</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Fine-tuning</h3>
            <p className="text-gray-700">The process of adapting a pre-trained model to specific tasks or domains by updating its parameters with new training data.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">LoRA (Low-Rank Adaptation)</h3>
            <p className="text-gray-700">A parameter-efficient fine-tuning technique that adds trainable rank decomposition matrices to existing weights.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Catastrophic Forgetting</h3>
            <p className="text-gray-700">A phenomenon where a model loses previously learned information when fine-tuned on new data.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Prompt Tuning</h3>
            <p className="text-gray-700">A technique that learns continuous prompt embeddings while keeping the model parameters frozen.</p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">Fine-tuning Process Overview</h3>
          <svg viewBox="0 0 800 250" className="w-full max-w-3xl mx-auto">
            {/* Pre-trained Model */}
            <rect x="50" y="50" width="200" height="150" rx="5" fill="#60A5FA" />
            <text x="150" y="85" textAnchor="middle" fill="white" className="font-bold">Pre-trained Model</text>
            <rect x="70" y="100" width="160" height="30" rx="3" fill="#3B82F6" />
            <text x="150" y="120" textAnchor="middle" fill="white" className="text-xs">Base Knowledge</text>
            <rect x="70" y="140" width="160" height="30" rx="3" fill="#3B82F6" />
            <text x="150" y="160" textAnchor="middle" fill="white" className="text-xs">Original Weights</text>

            {/* Fine-tuning Process */}
            <rect x="300" y="50" width="200" height="150" rx="5" fill="#34D399" />
            <text x="400" y="85" textAnchor="middle" fill="white" className="font-bold">Fine-tuning</text>
            <rect x="320" y="100" width="160" height="30" rx="3" fill="#2FB344" />
            <text x="400" y="120" textAnchor="middle" fill="white" className="text-xs">Training Data</text>
            <rect x="320" y="140" width="160" height="30" rx="3" fill="#2FB344" />
            <text x="400" y="160" textAnchor="middle" fill="white" className="text-xs">Parameter Updates</text>

            {/* Specialized Model */}
            <rect x="550" y="50" width="200" height="150" rx="5" fill="#F87171" />
            <text x="650" y="85" textAnchor="middle" fill="white" className="font-bold">Specialized Model</text>
            <rect x="570" y="100" width="160" height="30" rx="3" fill="#EF4444" />
            <text x="650" y="120" textAnchor="middle" fill="white" className="text-xs">Task-Specific</text>
            <rect x="570" y="140" width="160" height="30" rx="3" fill="#EF4444" />
            <text x="650" y="160" textAnchor="middle" fill="white" className="text-xs">Updated Weights</text>

            {/* Arrows */}
            <path d="M250 125 L300 125" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M500 125 L550 125" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 3:</strong> Fine-tuning process showing the progression from pre-trained model to specialized model.
          </p>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Memory Management</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Conversation Buffer</h3>
            <p className="text-gray-700">A storage mechanism for maintaining chat history and context in conversational AI systems.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Summary Buffer</h3>
            <p className="text-gray-700">A compressed representation of previous conversation history using summarization techniques.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Token Management</h3>
            <p className="text-gray-700">The process of tracking and optimizing token usage to stay within model context limits while maintaining relevant information.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Entity Memory</h3>
            <p className="text-gray-700">A system for tracking and maintaining information about specific entities mentioned in conversations.</p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Context Window Optimization</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Context Window</h3>
            <p className="text-gray-700">The maximum number of tokens a model can process in a single inference, limiting the amount of text it can consider at once.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Sliding Window</h3>
            <p className="text-gray-700">A technique for processing long documents by moving a fixed-size window through the text, maintaining overlap between segments.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Content Prioritization</h3>
            <p className="text-gray-700">The strategy of selecting the most relevant content to include within the context window based on importance or relevance scores.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Token Utilization</h3>
            <p className="text-gray-700">The efficiency with which available tokens in the context window are used to maintain relevant information.</p>
          </div>
        </div>
      </section>
    </div>
  );
}