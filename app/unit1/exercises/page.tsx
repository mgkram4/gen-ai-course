"use client"

import CodeBlock from '@/app/components/CodeBlock';
import { useState } from 'react';

interface MultipleChoiceQuestionProps {
  question: string;
  options: string[];
  correctAnswer: number;
}

interface PromptingExerciseProps {
  scenario: string;
  objective: string;
  placeholderText: string;
}

const MultipleChoiceQuestion = ({ question, options, correctAnswer }: MultipleChoiceQuestionProps) => {
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);

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

const PromptingExercise = ({ scenario, objective, placeholderText }: PromptingExerciseProps) => {
  const [input, setInput] = useState('');
  const [feedback, setFeedback] = useState('');

  const handleSubmit = () => {
    if (input.length < 10) {
      setFeedback('Please provide a more detailed prompt');
      return;
    }
    setFeedback('Good attempt! Compare your prompt with the sample solution below.');
  };

  return (
    <div className="mb-8 w-full max-w-md mx-auto">
      <div className="bg-gray-50 p-4 rounded-lg mb-4">
        <h3 className="font-bold mb-2">Scenario:</h3>
        <p className="text-gray-700 mb-2">{scenario}</p>
        <h3 className="font-bold mb-2">Objective:</h3>
        <p className="text-gray-700">{objective}</p>
      </div>
      <textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={placeholderText}
        className="border p-2 w-full mb-2 h-32 rounded"
      />
      <button
        onClick={handleSubmit}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 w-full"
      >
        Submit
      </button>
      {feedback && (
        <p className="mt-2 text-blue-600">{feedback}</p>
      )}
    </div>
  );
};

const Unit1Exercises = () => {
  return (
    <div className="container mx-auto px-4">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 1: Foundations Exercises</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Understanding Transformers</h2>
        
        <MultipleChoiceQuestion
          question="What is the primary purpose of self-attention in transformer models?"
          options={[
            "To reduce training time",
            "To compress the input sequence",
            "To capture relationships between different parts of the input",
            "To convert text to embeddings"
          ]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="Why is positional encoding necessary in transformer architectures?"
          options={[
            "To increase model capacity",
            "To provide sequence order information",
            "To reduce memory usage",
            "To enable parallel processing"
          ]}
          correctAnswer={1}
        />

        <MultipleChoiceQuestion
          question="What is the typical purpose of multi-head attention?"
          options={[
            "To reduce computational complexity",
            "To enable training on multiple GPUs",
            "To capture different types of relationships in parallel",
            "To increase model size"
          ]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="Which component normalizes the outputs in a transformer layer?"
          options={[
            "Feed Forward Network",
            "Layer Normalization",
            "Positional Encoding",
            "Attention Heads"
          ]}
          correctAnswer={1}
        />
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Prompt Engineering</h2>

        <PromptingExercise
          scenario="You need to create a customer service chatbot that helps users track their packages."
          objective="Write a system prompt that defines the bot's personality and capabilities."
          placeholderText="Write your system prompt here..."
        />

        <PromptingExercise
          scenario="You want the model to generate a technical tutorial about React hooks."
          objective="Create a few-shot prompt with examples to ensure consistent formatting."
          placeholderText="Write your few-shot prompt here..."
        />

        <PromptingExercise
          scenario="You need to help the model solve a complex math problem."
          objective="Write a chain-of-thought prompt that guides the model through the solution steps."
          placeholderText="Write your chain-of-thought prompt here..."
        />
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Code Implementation</h2>
        <p className="mb-4">Implement a simple self-attention mechanism:</p>
        <CodeBlock code={`
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # TODO: Initialize Q, K, V projection matrices
        
    def forward(self, x):
        # TODO: Implement the self-attention mechanism
        # 1. Project input to Q, K, V
        # 2. Compute attention scores
        # 3. Apply softmax
        # 4. Compute weighted sum
        pass

# Test your implementation
embed_dim = 64
batch_size = 2
seq_length = 10
x = torch.randn(batch_size, seq_length, embed_dim)
attention = SelfAttention(embed_dim)
output = attention(x)
        `} language="python" />
        <p className="mt-2">Implement the attention mechanism and test it with different inputs.</p>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Advanced Concepts</h2>
        
        <MultipleChoiceQuestion
          question="What is the purpose of temperature in sampling from a language model?"
          options={[
            "To control GPU temperature",
            "To adjust training speed",
            "To control output randomness",
            "To compress the model"
          ]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="Which sampling method helps prevent repetitive text generation?"
          options={[
            "Top-k sampling",
            "Random sampling",
            "Greedy sampling",
            "Uniform sampling"
          ]}
          correctAnswer={0}
        />

        <MultipleChoiceQuestion
          question="What is the main advantage of using embeddings over one-hot encoding?"
          options={[
            "Faster processing speed",
            "Lower memory usage",
            "Capture semantic relationships",
            "Simpler implementation"
          ]}
          correctAnswer={2}
        />
      </section>
    </div>
  );
};

export default Unit1Exercises;