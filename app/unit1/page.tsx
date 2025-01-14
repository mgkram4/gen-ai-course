import CodeBlock from '../components/CodeBlock';
import Layout from '../layout';

export default function Unit1() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 1: Foundations of Generative AI</h1>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. Introduction to Generative AI</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Generative AI refers to artificial intelligence systems that can create new content. Unlike traditional AI that only analyzes data, generative AI learns patterns and creates new outputs that maintain the characteristics of its training data.</p>
            
            <p className="mb-4 text-gray-700">Core concepts in generative AI include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Training data and distributions - The foundation of learning patterns</li>
              <li>Latent spaces - The AI&apos;s &quot;imagination space&quot; where concepts are represented</li>
              <li>Sampling strategies - Methods for selecting specific outputs</li>
              <li>Quality metrics - Measuring the effectiveness of generations</li>
              <li>Control methods - Techniques for guiding the generation process</li>
            </ul>
            
            <p className="mb-4 text-gray-700">Notable achievements in the field:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>GPT series - Advanced language understanding and generation</li>
              <li>DALL-E and Stable Diffusion - Text-to-image generation</li>
              <li>Whisper - State-of-the-art speech recognition</li>
              <li>Claude - Advanced reasoning and analysis</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Basic OpenAI API Integration</h4>
            <CodeBlock code={`
import openai

class SimpleGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key
        
    def generate_text(self, prompt, max_tokens=100):
        """Generate text using GPT model."""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Example usage
generator = SimpleGenerator('your-api-key')
prompt = "Write a haiku about artificial intelligence"
print(generator.generate_text(prompt))
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. Neural Networks Review</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Neural networks are computational systems inspired by biological brains. They process information through interconnected layers of artificial neurons, each performing specific mathematical operations.</p>
            
            <p className="mb-4 text-gray-700">Essential components:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Neurons - Basic processing units that transform inputs</li>
              <li>Activation functions - Non-linear transformations enabling complex learning</li>
              <li>Backpropagation - The learning process of adjusting network weights</li>
              <li>Optimization - Methods for improving network performance</li>
            </ul>

            <p className="mb-4 text-gray-700">Key architectures for generative AI:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Transformers - Specialized for understanding context in sequences</li>
              <li>CNNs - Optimized for processing visual information</li>
              <li>Autoencoders - Learning compressed data representations</li>
              <li>Attention mechanisms - Focusing on relevant information</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Simple Neural Network with PyTorch</h4>
            <CodeBlock code={`
import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Create and test the model
latent_dim = 100
output_dim = 784  # 28x28 image
generator = SimpleGenerator(latent_dim, output_dim)
z = torch.randn(1, latent_dim)
fake_image = generator(z)
print(f"Generated image shape: {fake_image.shape}")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. Transformers & Attention</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Transformers represent a breakthrough in AI architecture, enabling unprecedented understanding of context and relationships in data. They process entire sequences simultaneously rather than sequentially.</p>
            
            <p className="mb-4 text-gray-700">Core mechanisms:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Self-attention - Weighing relationships between all elements</li>
              <li>Multi-head attention - Multiple parallel attention processes</li>
              <li>Positional encoding - Maintaining sequence order information</li>
              <li>Feed-forward networks - Processing individual positions</li>
              <li>Layer normalization - Stabilizing learning process</li>
            </ul>

            <p className="mb-4 text-gray-700">Architecture variations:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Encoder-decoder - Complete sequence transformation</li>
              <li>Decoder-only - Focused on generation (like GPT)</li>
              <li>Encoder-only - Specialized for understanding (like BERT)</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Self-Attention Implementation</h4>
            <CodeBlock code={`
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create Q, K, V projection matrices
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.scale = embed_dim ** 0.5
    
    def forward(self, x):
        # Project input to Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        return output

# Test the attention mechanism
batch_size, seq_len, embed_dim = 1, 10, 512
x = torch.randn(batch_size, seq_len, embed_dim)
attention = SelfAttention(embed_dim)
output = attention(x)
print(f"Output shape: {output.shape}")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Prompt Engineering</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Prompt engineering is the art and science of effectively communicating with AI models to achieve desired outcomes. It combines understanding of model behavior with structured input design.</p>
            
            <p className="mb-4 text-gray-700">Key techniques and their applications:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Zero-shot prompting - Direct instructions without examples</li>
              <li>Few-shot learning - Providing demonstration examples</li>
              <li>Chain-of-thought - Guiding logical reasoning steps</li>
              <li>System prompts - Setting model behavior and context</li>
              <li>Output formatting - Controlling response structure</li>
            </ul>

            <p className="mb-4 text-gray-700">Best practices:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Be specific and clear in instructions</li>
              <li>Provide context and constraints</li>
              <li>Use consistent formatting</li>
              <li>Test and iterate prompt designs</li>
              <li>Consider edge cases and limitations</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Advanced Prompting Patterns</h4>
            <CodeBlock code={`
class PromptTemplates:
    @staticmethod
    def few_shot_prompt(task, examples, query):
        """Create a few-shot prompt with examples."""
        prompt = f"{task}\\n\\nExamples:\\n"
        
        for input_text, output_text in examples:
            prompt += f"Input: {input_text}\\n"
            prompt += f"Output: {output_text}\\n\\n"
            
        prompt += f"Input: {query}\\nOutput:"
        return prompt
    
    @staticmethod
    def chain_of_thought(question):
        """Create a chain-of-thought prompt."""
        return f"""Question: {question}
Let's solve this step by step:
1) First, let's understand what we're asked
2) Then, let's break down the problem
3) Finally, let's solve each part
Please show your reasoning for each step."""

# Example usage
examples = [
    ("Convert 'hello' to uppercase", "HELLO"),
    ("Convert 'Python' to uppercase", "PYTHON")
]
prompt = PromptTemplates.few_shot_prompt(
    "Convert the given text to uppercase.",
    examples,
    "javascript"
)
print(prompt)
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Conclusion</h2>
          <p className="mb-4 text-gray-700">This unit has covered the essential foundations of generative AI, including:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>Core concepts and terminology</li>
            <li>Neural network fundamentals</li>
            <li>Transformer architectures</li>
            <li>Prompt engineering techniques</li>
            <li>Basic implementation patterns</li>
          </ul>
          
          <p className="mb-4 text-gray-700">Next steps for deeper understanding:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>Experiment with different model architectures</li>
            <li>Practice prompt engineering techniques</li>
            <li>Study attention mechanisms in detail</li>
            <li>Explore model evaluation metrics</li>
            <li>Learn about ethical considerations</li>
          </ul>
          
          <p className="mb-4 text-gray-700">These foundational concepts will be essential as we explore more advanced topics in subsequent units, including working with specific types of generative models and deploying them in production environments.</p>
        </section>
      </div>
    </Layout>
  );
}
