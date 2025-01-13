'use client';

import CodeBlock from '@/app/components/CodeBlock';
import React, { useState } from 'react';

interface SolutionPanelProps {
  children: React.ReactNode;
}

const SolutionPanel: React.FC<SolutionPanelProps> = ({ children }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="mt-4">
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="bg-gray-100 text-blue-600 px-4 py-2 rounded hover:bg-gray-200 transition-colors w-full text-left flex justify-between items-center"
      >
        <span>{isVisible ? 'Hide Solution' : 'Show Solution'}</span>
        <span className="transform transition-transform duration-200" style={{ transform: isVisible ? 'rotate(180deg)' : '' }}>▼</span>
      </button>
      {isVisible && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
          {children}
        </div>
      )}
    </div>
  );
};

interface MultipleChoiceQuestionProps {
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const MultipleChoiceQuestion: React.FC<MultipleChoiceQuestionProps> = ({ question, options, correctAnswer, explanation }) => {
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
      <SolutionPanel>
        <p className="font-semibold">Explanation:</p>
        <p>{explanation}</p>
      </SolutionPanel>
    </div>
  );
};

const Unit2Exercises: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 2: Language Models and APIs - Exercises</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">LLM Integration</h2>
        
        <MultipleChoiceQuestion
          question="Which approach allows an LLM to perform tasks without any examples in the prompt?"
          options={[
            "Few-shot learning",
            "Zero-shot learning",
            "One-shot learning",
            "Transfer learning"
          ]}
          correctAnswer={1}
          explanation="Zero-shot learning allows models to handle tasks using only clear instructions without examples, relying on their pre-trained knowledge to understand and execute the task."
        />

        <MultipleChoiceQuestion
          question="What is the primary purpose of an API wrapper when working with LLMs?"
          options={[
            "To increase the model's accuracy",
            "To reduce API costs",
            "To simplify integration and handle common tasks",
            "To improve response speed"
          ]}
          correctAnswer={2}
          explanation="API wrappers simplify integration by handling authentication, request formatting, error handling, and response processing, making it easier to work with LLM APIs."
        />

        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-2">Implement Basic API Integration</h3>
          <CodeBlock code={`
import requests
from typing import Dict, Any

class LLMWrapper:
    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the LLM API wrapper
        Handle authentication and base configuration
        """
        # Your implementation here
        pass
    
    def generate_completion(self, prompt: str) -> Dict[str, Any]:
        """
        Send a completion request to the LLM API
        Handle errors and return the response
        """
        # Your implementation here
        pass

    def handle_rate_limiting(self) -> None:
        """
        Implement rate limiting logic to stay within API quotas
        """
        # Your implementation here
        pass

# Example usage:
wrapper = LLMWrapper("your-api-key", "https://api.example.com")
response = wrapper.generate_completion("Translate 'hello' to French")
          `} language="python" />
          <SolutionPanel>
            <CodeBlock code={`
import requests
import time
from typing import Dict, Any

class LLMWrapper:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second

    def handle_rate_limiting(self) -> None:
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def generate_completion(self, prompt: str) -> Dict[str, Any]:
        self.handle_rate_limiting()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                headers=self.headers,
                json={
                    "prompt": prompt,
                    "max_tokens": 100
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during API call: {e}")
            return {"error": str(e)}

# Example usage
api = LLMWrapper("your-api-key", "https://api.example.com")
result = api.generate_completion("Translate 'hello' to French")
print(result)
            `} language="python" />
            <p className="mt-4">
              This implementation includes:
              1. Basic authentication and headers setup
              2. Rate limiting implementation
              3. Error handling for API calls
              4. Clean response processing
            </p>
          </SolutionPanel>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Prompt Engineering</h2>

        <MultipleChoiceQuestion
          question="What is chain-of-thought prompting used for?"
          options={[
            "Reducing API costs",
            "Improving response speed",
            "Breaking down complex reasoning steps",
            "Handling multiple languages"
          ]}
          correctAnswer={2}
          explanation="Chain-of-thought prompting helps models break down complex problems into explicit reasoning steps, improving their ability to solve complex tasks by showing their work."
        />

        <MultipleChoiceQuestion
          question="When is few-shot learning most appropriate?"
          options={[
            "When you need the fastest possible response",
            "When demonstrating specific patterns or formats",
            "When working with simple yes/no questions",
            "When trying to reduce token usage"
          ]}
          correctAnswer={1}
          explanation="Few-shot learning is most appropriate when you need to demonstrate specific patterns or formats through examples, helping the model understand the exact output structure you want."
        />

        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-2">Implement Chain of Thought Prompting</h3>
          <CodeBlock code={`
def create_chain_of_thought_prompt(question: str) -> str:
    """
    Create a prompt that encourages step-by-step reasoning
    Example: For math problems, break down the solution steps
    """
    # Your implementation here
    pass

# Test cases:
math_question = "If a train travels 120 km in 2 hours, what is its average speed?"
reasoning_prompt = create_chain_of_thought_prompt(math_question)
          `} language="python" />
          <SolutionPanel>
            <CodeBlock code={`
def create_chain_of_thought_prompt(question: str) -> str:
    template = f"""Please solve this step by step:
Question: {question}
Let's approach this systematically:

1) First, let's identify the key information
2) Then, we'll determine what formula to use
3) Finally, we'll calculate the answer

Please show your reasoning for each step."""

    return template

# Example usage for a math problem
math_question = "If a train travels 120 km in 2 hours, what is its average speed?"
prompt = create_chain_of_thought_prompt(math_question)

# Expected LLM response would include steps like:
# 1) Key information: Distance = 120 km, Time = 2 hours
# 2) Formula needed: Average Speed = Distance ÷ Time
# 3) Calculation: 120 km ÷ 2 hours = 60 km/h
            `} language="python" />
            <p className="mt-4">
              This implementation:
              1. Creates a structured prompt template
              2. Encourages explicit step-by-step reasoning
              3. Helps the model break down complex problems
              4. Makes the solution process transparent
            </p>
          </SolutionPanel>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Advanced Integration Patterns</h2>

        <MultipleChoiceQuestion
          question="What is the purpose of function calling in LLM APIs?"
          options={[
            "To run Python functions faster",
            "To allow models to request specific actions",
            "To reduce API costs",
            "To improve response timing"
          ]}
          correctAnswer={1}
          explanation="Function calling allows models to request specific actions or external data by specifying structured function calls, enabling them to interact with external systems and tools."
        />

        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-2">Implement Function Calling</h3>
          <CodeBlock code={`
from typing import List, Dict, Any

def define_functions() -> List[Dict[str, Any]]:
    """
    Define available functions for the LLM to call
    Include function descriptions and parameters
    """
    return [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    ]

def handle_function_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle function calls from the LLM
    Implement the actual functionality
    """
    # Your implementation here
    pass
          `} language="python" />
          <SolutionPanel>
            <CodeBlock code={`
import json
from typing import List, Dict, Any
import requests

def define_functions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    ]

def handle_function_call(call: Dict[str, Any]) -> Dict[str, Any]:
    function_name = call.get("name")
    arguments = json.loads(call.get("arguments", "{}"))
    
    if function_name == "get_weather":
        location = arguments.get("location")
        # In a real implementation, call a weather API
        return {
            "temperature": 20,
            "conditions": "sunny",
            "location": location
        }
    
    return {"error": "Function not found"}

class LLMWithFunctions:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.functions = define_functions()
    
    def generate_with_functions(self, prompt: str) -> Dict[str, Any]:
        # Send prompt and function definitions to LLM API
        response = {
            "function_call": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "London"})
            }
        }
        
        if "function_call" in response:
            function_response = handle_function_call(response["function_call"])
            # Send function response back to LLM for final answer
            return self.generate_completion(prompt, function_response)
        
        return response

# Example usage
llm = LLMWithFunctions("your-api-key")
response = llm.generate_with_functions("What's the weather like in London?")
            `} language="python" />
            <p className="mt-4">
              This implementation demonstrates:
              1. Function definition with JSON Schema
              2. Function call handling
              3. Integration with LLM API
              4. Response processing and follow-up
            </p>
          </SolutionPanel>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Challenge Questions</h2>

        <MultipleChoiceQuestion
          question="Which state management approach is most appropriate for long conversations with LLMs?"
          options={[
            "Storing only the last message",
            "Using a sliding window of messages",
            "Storing all messages without compression",
            "Ignoring previous messages"
          ]}
          correctAnswer={1}
          explanation="A sliding window of messages balances context retention with token limits, allowing for coherent conversations while managing costs and context length."
        />

        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-2">Implement Message Window Management</h3>
          <CodeBlock code={`
class ConversationManager:
    def __init__(self, window_size: int = 10):
        self.messages = []
        self.window_size = window_size
    
    def add_message(self, message: str) -> None:
        self.messages.append(message)
        if len(self.messages) > self.window_size:
            self.messages.pop(0)
    
    def get_context(self) -> str:
        return "\\n".join(self.messages)
          `} language="python" />
          <SolutionPanel>
            <p className="mt-4">
              Key considerations for conversation management:
              1. Balance context retention with token limits
              2. Implement efficient message storage
              3. Handle context summarization
              4. Manage conversation state
            </p>
          </SolutionPanel>
        </div>
      </section>
    </div>
  );
};

export default Unit2Exercises;