'use client';

import CodeBlock from '@/app/components/CodeBlock';
import React, { useState } from 'react';

interface MultipleChoiceQuestionProps {
  question: string;
  options: string[];
  correctAnswer: number;
}

const MultipleChoiceQuestion: React.FC<MultipleChoiceQuestionProps> = ({ question, options, correctAnswer }) => {
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState<boolean>(false);

  const handleSubmit = () => {
    setShowResult(true);
  };

  return (
    <div className="mb-8 w-full max-w-md mx-auto">
      <h3 className="text-xl font-semibold mb-2">{question}</h3>
      <div className="space-y-2">
        {options.map((option, index) => (
          <div key={index} className="flex items-center">
            <input
              type="radio"
              id={`q${question.replace(/\s/g, '')}-${index}`}
              name={`answer-${question.replace(/\s/g, '')}`}
              value={index}
              checked={selectedAnswer === index}
              onChange={() => setSelectedAnswer(index)}
              className="mr-2"
            />
            <label htmlFor={`q${question.replace(/\s/g, '')}-${index}`}>{option}</label>
          </div>
        ))}
      </div>
      <button
        onClick={handleSubmit}
        className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 w-full"
      >
        Submit
      </button>
      {showResult && (
        <p className={`mt-2 ${selectedAnswer === correctAnswer ? 'text-green-600' : 'text-red-600'}`}>
          {selectedAnswer === correctAnswer ? 'Correct!' : 'Incorrect. Try again!'}
        </p>
      )}
    </div>
  );
};

interface GraphicsExerciseProps {
  question: string;
  inputPlaceholder: string;
  checkAnswer: (input: string) => boolean;
}

const GraphicsExercise: React.FC<GraphicsExerciseProps> = ({ question, inputPlaceholder, checkAnswer }) => {
  const [input, setInput] = useState<string>('');
  const [result, setResult] = useState<boolean | null>(null);

  const handleSubmit = () => {
    setResult(checkAnswer(input));
  };

  return (
    <div className="mb-8 w-full max-w-md mx-auto">
      <h3 className="text-xl font-semibold mb-2">{question}</h3>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={inputPlaceholder}
        className="border p-2 mr-2 w-full mb-2"
      />
      <button
        onClick={handleSubmit}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 w-full"
      >
        Submit
      </button>
      {result !== null && (
        <p className={`mt-2 ${result ? 'text-green-600' : 'text-red-600'}`}>
          {result ? 'Correct!' : 'Incorrect. Try again!'}
        </p>
      )}
    </div>
  );
};

const Unit3Exercises: React.FC = () => {
  return (
    <div className="container mx-auto px-4">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 3: Computer Graphics Exercises</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Multiple Choice Questions</h2>
        
        <MultipleChoiceQuestion
          question="What is the primary purpose of RAG (Retrieval Augmented Generation)?"
          options={[
            "To generate random text",
            "To combine information retrieval with language generation",
            "To create vector embeddings",
            "To optimize memory usage"
          ]}
          correctAnswer={1}
        />

        <MultipleChoiceQuestion
          question="Which technique is used for parameter-efficient fine-tuning of language models?"
          options={[
            "Vector Embedding",
            "Document Chunking",
            "LoRA (Low-Rank Adaptation)",
            "Cosine Similarity"
          ]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="What is the purpose of a Vector Database in RAG systems?"
          options={[
            "To store raw text documents",
            "To process images",
            "To store and query high-dimensional vectors",
            "To generate embeddings"
          ]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="What is Catastrophic Forgetting in the context of LLMs?"
          options={[
            "When a model crashes during training",
            "When a model loses previously learned information during fine-tuning",
            "When a model's memory is full",
            "When a model fails to generate responses"
          ]}
          correctAnswer={1}
        />

        <MultipleChoiceQuestion
          question="Which metric is commonly used to measure similarity between vectors in embedding space?"
          options={[
            "Euclidean distance",
            "Manhattan distance",
            "Cosine similarity",
            "Hamming distance"
          ]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="What is the purpose of Document Chunking in RAG systems?"
          options={[
            "To compress documents",
            "To break documents into smaller pieces for efficient processing",
            
            "To encrypt documents",
            "To format documents"
          ]}
          correctAnswer={1}
        />
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Technical Calculations</h2>

        <GraphicsExercise
          question="If a vector has dimensions of 1536, and you want to store 1 million vectors, how many GB of storage would you need? (assuming 4 bytes per number, round to 2 decimal places)"
          inputPlaceholder="Enter size in GB"
          checkAnswer={(input: string) => parseFloat(input) === 5.73}
        />

        <GraphicsExercise
          question="If your context window is 8K tokens and each chunk is 512 tokens, how many chunks can fit in the context window while leaving 1K tokens for the response?"
          inputPlaceholder="Enter number of chunks"
          checkAnswer={(input: string) => parseInt(input) === 13}
        />

        <GraphicsExercise
          question="If your embedding model processes 100 tokens per second, how long (in seconds) would it take to process a document with 25,000 tokens?"
          inputPlaceholder="Enter time in seconds"
          checkAnswer={(input: string) => parseInt(input) === 250}
        />

        <GraphicsExercise
          question="Calculate the cosine similarity between vectors [1,1] and [0,1] (round to 2 decimal places):"
          inputPlaceholder="Enter similarity value"
          checkAnswer={(input: string) => parseFloat(input) === 0.71}
        />
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Code Exercises</h2>
        
        <div className="mb-8">
          <h3 className="text-xl mb-4">1. Document Chunking</h3>
          <p className="mb-4">Implement a function to chunk a document based on token count:</p>
          <CodeBlock code={`
def chunk_document(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split a document into chunks with specified size and overlap
    text: input document
    chunk_size: maximum tokens per chunk
    overlap: number of overlapping tokens between chunks
    Returns: list of chunks
    """
    # Your implementation here
    pass

# Example usage:
# chunks = chunk_document("Your long document here...", 512, 50)
          `} language="python" />
        </div>

        <div className="mb-8">
          <h3 className="text-xl mb-4">2. Vector Similarity</h3>
          <p className="mb-4">Implement cosine similarity calculation:</p>
          <CodeBlock code={`
def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors
    Returns: similarity score between -1 and 1
    """
    # Your implementation here
    pass

# Test cases:
print(cosine_similarity([1,0], [1,0]))     # Should output 1.0
print(cosine_similarity([1,0], [0,1]))     # Should output 0.0
          `} language="python" />
        </div>

        <div className="mb-8">
          <h3 className="text-xl mb-4">3. RAG Pipeline</h3>
          <p className="mb-4">Implement a basic RAG retrieval function:</p>
          <CodeBlock code={`
async function retrieveRelevantDocuments(
  query: string,
  vectorStore: VectorStore,
  numResults: number = 3
): Promise<Document[]> {
    """
    Retrieve relevant documents for a query using vector similarity
    Returns: Array of most relevant documents
    """
    // Your implementation here
    pass
}

// Example usage:
// const docs = await retrieveRelevantDocuments("How does RAG work?", vectorStore);
          `} language="typescript" />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Challenge Problems</h2>
        
        <div className="mb-8">
          <h3 className="text-xl mb-4">Hybrid Search Implementation</h3>
          <p className="mb-4">Implement a hybrid search combining vector and keyword search:</p>
          <CodeBlock code={`
class HybridSearch:
    def __init__(self, vector_store, keyword_index):
        self.vector_store = vector_store
        self.keyword_index = keyword_index

    def search(
        self,
        query: str,
        alpha: float = 0.5,
        top_k: int = 5
    ) -> list:
        """
        Implement hybrid search combining vector similarity and keyword matching
        alpha: weight between vector (alpha) and keyword (1-alpha) scores
        Returns: list of ranked documents
        """
        # Your implementation here
        pass

# Example usage:
# searcher = HybridSearch(vector_store, keyword_index)
# results = searcher.search("What is RAG?", alpha=0.7)
          `} language="python" />
        </div>
      </section>
    </div>
  );
};

export default Unit3Exercises;