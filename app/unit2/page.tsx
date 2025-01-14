import CodeBlock from '../components/CodeBlock';
import Layout from '../layout';

export default function Unit2() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 2: Working with Language Models</h1>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. Understanding Large Language Models</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Major LLM Providers</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div className="bg-white p-4 rounded-lg shadow">
                <h4 className="font-bold mb-2">OpenAI</h4>
                <ul className="list-disc list-inside text-gray-700">
                  <li>GPT-4 and GPT-3.5</li>
                  <li>Specialized models (text-davinci-003)</li>
                  <li>Function calling capabilities</li>
                  <li>Pay-per-token pricing</li>
                </ul>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <h4 className="font-bold mb-2">Anthropic</h4>
                <ul className="list-disc list-inside text-gray-700">
                  <li>Claude and Claude 2</li>
                  <li>Longer context windows</li>
                  <li>Advanced reasoning</li>
                  <li>Constitutional AI focus</li>
                </ul>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <h4 className="font-bold mb-2">Open Source</h4>
                <ul className="list-disc list-inside text-gray-700">
                  <li>LLaMA and derivatives</li>
                  <li>Local deployment options</li>
                  <li>Community-driven development</li>
                  <li>Customization flexibility</li>
                </ul>
              </div>
              <div className="bg-white p-4 rounded-lg shadow">
                <h4 className="font-bold mb-2">Cloud Providers</h4>
                <ul className="list-disc list-inside text-gray-700">
                  <li>Google (PaLM, Gemini)</li>
                  <li>Azure OpenAI Service</li>
                  <li>AWS Bedrock</li>
                  <li>Enterprise integration focus</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Comparison Matrix</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full bg-white">
                <thead>
                  <tr>
                    <th className="px-4 py-2 border">Feature</th>
                    <th className="px-4 py-2 border">GPT-4</th>
                    <th className="px-4 py-2 border">Claude 2</th>
                    <th className="px-4 py-2 border">LLaMA 2</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="px-4 py-2 border">Context Window</td>
                    <td className="px-4 py-2 border">8K-32K tokens</td>
                    <td className="px-4 py-2 border">100K tokens</td>
                    <td className="px-4 py-2 border">4K tokens</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 border">Deployment</td>
                    <td className="px-4 py-2 border">Cloud API</td>
                    <td className="px-4 py-2 border">Cloud API</td>
                    <td className="px-4 py-2 border">Self-hosted</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2 border">Cost Model</td>
                    <td className="px-4 py-2 border">Pay per token</td>
                    <td className="px-4 py-2 border">Pay per token</td>
                    <td className="px-4 py-2 border">Infrastructure</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. API Integration Patterns</h2>
          
          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Understanding API Integration</h4>
            <p className="mb-4">
              API integration allows your application to communicate with Language Models. Each provider has their own API structure,
              but they follow similar patterns. The examples below show how to create reusable clients for different providers.
            </p>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">OpenAI Integration</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Create a reusable wrapper around OpenAIs API</li>
                <li>Handle authentication and provide a simplified interface</li>
                <li>Support both basic chat completions and function calling</li>
                <li>Enable AI assistants to perform actions through function definitions</li>
              </ul>
              <div className="mt-4">
                <h4 className="font-bold mb-2">Real-world Applications:</h4>
                <ul className="list-disc list-inside space-y-2">
                  <li>Building chatbots</li>
                  <li>Creating AI assistants that can perform actions</li>
                  <li>Integration with external services</li>
                </ul>
              </div>
            </div>
            <CodeBlock code={`
from openai import OpenAI
from typing import Optional, Dict, Any

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete_chat(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        functions: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI API.
        
        Args:
            messages: List of message dictionaries
            temperature: Randomness of output (0-1)
            max_tokens: Maximum tokens to generate
            functions: List of function definitions
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                functions=functions
            )
            return {
                'content': response.choices[0].message.content,
                'usage': response.usage.total_tokens,
                'finish_reason': response.choices[0].finish_reason
            }
        except Exception as e:
            return {'error': str(e)}

# Example usage
client = OpenAIClient('your-api-key')

# Basic chat completion
response = client.complete_chat([
    {"role": "user", "content": "Summarize the benefits of LLMs"}
])

# Function calling
functions = [{
    "name": "get_weather",
    "description": "Get weather for a location",
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
}]

response = client.complete_chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    functions=functions
)
            `} language="python" />
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Anthropic Integration</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Similar to OpenAI but for Anthropics Claude model</li>
                <li>Simpler interface focused on single-message completions</li>
                <li>Different providers have different API structures</li>
                <li>Demonstrates proper error handling</li>
              </ul>
            </div>
            <CodeBlock code={`
from anthropic import Anthropic

class AnthropicClient:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def complete_message(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Send a completion request to Anthropic's Claude.
        """
        try:
            message = self.client.messages.create(
                model="claude-2",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Example usage
client = AnthropicClient('your-api-key')
response = client.complete_message(
    "Analyze the implications of large context windows in LLMs."
)
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. Basic Prompt Engineering</h2>
          
          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Understanding Prompt Engineering</h4>
            <p className="mb-4">
              Prompt engineering is the art of crafting effective inputs for language models. Different techniques can be used
              depending on your needs:
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Few-shot learning: Teaching by example</li>
              <li>Chain of thought: Breaking down complex reasoning</li>
              <li>Zero-shot learning: Direct instructions without examples</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Few-Shot Learning Examples</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Help models understand patterns through examples</li>
                <li>Show input/output pairs instead of explaining rules</li>
                <li>Model learns from examples and applies patterns</li>
              </ul>
              <div className="mt-4">
                <h4 className="font-bold mb-2">Use Cases:</h4>
                <ul className="list-disc list-inside space-y-2">
                  <li>Teaching specific output formats</li>
                  <li>Training for consistent response patterns</li>
                  <li>Handling domain-specific tasks</li>
                </ul>
              </div>
            </div>
            <CodeBlock code={`
def create_few_shot_prompt(examples: list, query: str) -> str:
    """Create a few-shot prompt with examples."""
    prompt = "Transform the input according to these examples:\\n\\n"
    
    for input_text, output_text in examples:
        prompt += f"Input: {input_text}\\n"
        prompt += f"Output: {output_text}\\n\\n"
    
    prompt += f"Input: {query}\\nOutput:"
    return prompt

# Example: Teaching addition format
examples = [
    ("2 + 3", "Let's solve step by step:\\n1) 2 + 3 = 5\\nAnswer: 5"),
    ("7 + 4", "Let's solve step by step:\\n1) 7 + 4 = 11\\nAnswer: 11")
]

query = "9 + 6"
prompt = create_few_shot_prompt(examples, query)
            `} language="python" />

            <h3 className="text-2xl font-medium mb-2 mt-6 text-blue-400">Chain of Thought</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Break down complex problems into steps</li>
                <li>Improve reasoning through step-by-step thinking</li>
                <li>Reduce errors by making thought process explicit</li>
                <li>Perfect for math, logic, and complex reasoning</li>
              </ul>
            </div>
            <CodeBlock code={`
def create_cot_prompt(question: str) -> str:
    """Create a chain-of-thought prompt."""
    return f"""Question: {question}
Please solve this step by step:
1) First, let's understand what we're asked
2) Break down the components
3) Process each part
4) Combine the results
5) Verify the answer

For each step, explain your reasoning clearly."""

# Example usage
question = "If a train travels 120 km in 2 hours, then stops for 30 minutes, and finally travels 90 km in 1.5 hours, what is its average speed for the entire journey?"
prompt = create_cot_prompt(question)
            `} language="python" />

            <h3 className="text-2xl font-medium mb-2 mt-6 text-blue-400">Zero-Shot Learning</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Handle tasks without providing examples</li>
                <li>Rely on clear instructions and format specifications</li>
                <li>Best for simpler, well-defined tasks</li>
                <li>Useful when examples arent available</li>
              </ul>
            </div>
            <CodeBlock code={`
def create_zero_shot_prompt(task: str, format_instructions: str) -> str:
    """Create a zero-shot prompt with format instructions."""
    return f"""Task: {task}

Instructions:
- Follow the exact format specified
- Be concise and accurate
- Show only the requested information

Format Specification:
{format_instructions}

Please complete the task:"""

# Example usage
task = "Classify the sentiment of this tweet: 'Just had the best day ever at the beach!'"
format_instructions = """
Output should be exactly one of:
- POSITIVE
- NEGATIVE
- NEUTRAL
"""

prompt = create_zero_shot_prompt(task, format_instructions)
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Building Simple Applications</h2>
          
          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Understanding Chat Applications</h4>
            <p className="mb-4">
              Building a chat application with LLMs involves several key components working together:
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Message history management</li>
              <li>User interface for interaction</li>
              <li>State management between requests</li>
              <li>Error handling and response processing</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Basic Chat Application</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Components:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>ChatApp class: Manages the core chat functionality</li>
                <li>Message History: Tracks conversation context</li>
                <li>Streamlit Interface: Provides user interaction</li>
                <li>Session State: Maintains conversation across requests</li>
              </ul>
              <div className="mt-4">
                <h4 className="font-bold mb-2">Implementation Features:</h4>
                <ul className="list-disc list-inside space-y-2">
                  <li>Real-time message updates</li>
                  <li>Persistent conversation history</li>
                  <li>Clean separation of concerns</li>
                  <li>Error handling and recovery</li>
                </ul>
              </div>
            </div>
            <CodeBlock code={`
import streamlit as st
from typing import List, Dict

class ChatApp:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""
        self.messages.append({"role": role, "content": content})
    
    def get_response(self) -> str:
        """Get model response for the current conversation."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.title("Simple Chat Application")
    
    # Initialize session state
    if "chat_app" not in st.session_state:
        st.session_state.chat_app = ChatApp(st.secrets["OPENAI_API_KEY"])
    
    # Chat interface
    user_input = st.text_input("Your message:")
    if st.button("Send"):
        # Add user message
        st.session_state.chat_app.add_message("user", user_input)
        
        # Get and add assistant response
        response = st.session_state.chat_app.get_response()
        st.session_state.chat_app.add_message("assistant", response)
    
    # Display chat history
    for message in st.session_state.chat_app.messages:
        role = "You" if message["role"] == "user" else "Assistant"
        st.write(f"{role}: {message['content']}")

if __name__ == "__main__":
    main()
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">5. Introduction to LangChain</h2>
          
          <div className="bg-blue-50 p-4 rounded-lg mb-6">
            <h4 className="font-bold mb-2">Understanding LangChain</h4>
            <p className="mb-4">
              LangChain is a powerful framework that simplifies building LLM applications by providing:
            </p>
            <ul className="list-disc list-inside space-y-2">
              <li>Abstracted model interactions</li>
              <li>Chainable operations</li>
              <li>Built-in prompt management</li>
              <li>Memory and state handling</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Basic LangChain Usage</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>LLMChain: Basic building block for LLM operations</li>
                <li>PromptTemplates: Structured prompt creation</li>
                <li>Chain Types: Summarization and Q&A patterns</li>
                <li>Error Handling: Robust application design</li>
              </ul>
              <div className="mt-4">
                <h4 className="font-bold mb-2">Common Use Cases:</h4>
                <ul className="list-disc list-inside space-y-2">
                  <li>Text summarization</li>
                  <li>Question answering systems</li>
                  <li>Document analysis</li>
                  <li>Conversational agents</li>
                </ul>
              </div>
            </div>
            <CodeBlock code={`
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LangChainExample:
    def __init__(self, api_key: str):
        self.llm = OpenAI(api_key=api_key)
    
    def create_summary_chain(self) -> LLMChain:
        """Create a chain for text summarization."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Please summarize the following text:\\n\\n{text}"
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def create_qa_chain(self) -> LLMChain:
        """Create a chain for question answering."""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
        )
        return LLMChain(llm=self.llm, prompt=prompt)

    def run_chain(self, chain: LLMChain, **kwargs) -> str:
        """Run a LangChain chain with given inputs."""
        try:
            return chain.run(**kwargs)
        except Exception as e:
            return f"Error: {str(e)}"

# Example usage
example = LangChainExample('your-api-key')

# Text summarization
summary_chain = example.create_summary_chain()
summary = example.run_chain(
    summary_chain,
    text="LangChain is a framework for developing applications powered by language models. It provides tools and abstractions for working with LLMs, including prompt management, model interaction, and chain creation."
)

# Question answering
qa_chain = example.create_qa_chain()
answer = example.run_chain(
    qa_chain,
    context="The Python programming language was created by Guido van Rossum and was first released in 1991.",
    question="When was Python first released?"
)
            `} language="python" />

            <h3 className="text-2xl font-medium mb-2 mt-6 text-blue-400">Sequential Chains</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h4 className="font-bold mb-2">Key Concepts:</h4>
              <ul className="list-disc list-inside space-y-2">
                <li>Chain Composition: Combining multiple operations</li>
                <li>Data Flow: Passing information between chains</li>
                <li>Error Propagation: Managing failures across chains</li>
                <li>Debugging: Verbose mode for chain inspection</li>
              </ul>
              <div className="mt-4">
                <h4 className="font-bold mb-2">Common Applications:</h4>
                <ul className="list-disc list-inside space-y-2">
                  <li>Multi-step text processing</li>
                  <li>Complex analysis pipelines</li>
                  <li>Document transformation workflows</li>
                  <li>Advanced reasoning systems</li>
                </ul>
              </div>
            </div>
            <CodeBlock code={`
from langchain.chains import SimpleSequentialChain

def create_analysis_chain(api_key: str) -> SimpleSequentialChain:
    """Create a sequential chain for text analysis."""
    llm = OpenAI(api_key=api_key)
    
    # First chain: Summarization
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the main points:\\n{text}"
    )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    
    # Second chain: Analysis
    analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template="Analyze the implications of:\\n{text}"
    )
    analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
    
    # Combine chains
    return SimpleSequentialChain(
        chains=[summary_chain, analysis_chain],
        verbose=True
    )

# Example usage
chain = create_analysis_chain('your-api-key')
result = chain.run(
    "Recent advances in LLMs have enabled new applications in areas like code generation, creative writing, and automated analysis."
)
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Conclusion and Next Steps</h2>
          <div className="mb-6">
            <p className="mb-4 text-gray-700">In this unit, weve covered practical aspects of working with Language Models:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Understanding different LLM providers and their offerings</li>
              <li>Implementing API integrations with major providers</li>
              <li>Applying basic prompt engineering techniques</li>
              <li>Building simple applications</li>
              <li>Getting started with LangChain</li>
            </ul>

            <p className="mb-4 text-gray-700">Moving forward to Unit 3, well explore advanced concepts including:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Retrieval Augmented Generation (RAG)</li>
              <li>Vector databases and embeddings</li>
              <li>Fine-tuning strategies</li>
              <li>Advanced memory management</li>
              <li>Context window optimization</li>
            </ul>

            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-bold mb-2">Practical Exercises</h4>
              <ol className="list-decimal list-inside space-y-2">
                <li>Implement a simple chatbot using the OpenAI API</li>
                <li>Create a few-shot learning system for specific tasks</li>
                <li>Build a basic LangChain application</li>
                <li>Compare responses from different LLM providers</li>
                <li>Practice prompt engineering with real-world examples</li>
              </ol>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}