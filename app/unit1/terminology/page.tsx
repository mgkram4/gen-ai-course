export default function Unit1Terminology() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 1: Foundations of Generative AI - Key Terms</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Core Concepts</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Generative AI</h3>
            <p className="text-gray-700">AI systems that can create new content including text, images, code, and more. These models learn patterns from training data to generate novel outputs that maintain similar characteristics.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Latent Space</h3>
            <p className="text-gray-700">A compressed, continuous representation space where similar data points are mapped close together. It captures the essential features and patterns of the training data.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Training Distribution</h3>
            <p className="text-gray-700">The statistical patterns and characteristics of the data used to train the model. The model learns to generate new samples that follow this distribution.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Sampling</h3>
            <p className="text-gray-700">The process of generating new outputs from a trained model by selecting points from the learned distribution, often with parameters like temperature to control randomness.</p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Key Concept Diagrams</h2>
        
        {/* Latent Space Visualization */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
          <h3 className="font-bold mb-4">Latent Space Representation</h3>
          <svg viewBox="0 0 400 200" className="w-full max-w-2xl mx-auto">
            {/* Grid Background */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
              </pattern>
            </defs>
            <rect width="400" height="200" fill="url(#grid)" />
            
            {/* Data Points */}
            {[...Array(20)].map((_, i) => (
              <circle
                key={`point-${i}`}
                cx={100 + Math.random() * 200}
                cy={50 + Math.random() * 100}
                r="4"
                fill={`hsl(${Math.random() * 360}, 70%, 50%)`}
                opacity="0.6"
              />
            ))}
            
            {/* Clusters */}
            <circle cx="150" cy="100" r="30" fill="none" stroke="#60A5FA" strokeWidth="2" strokeDasharray="5,5"/>
            <circle cx="250" cy="100" r="35" fill="none" stroke="#F87171" strokeWidth="2" strokeDasharray="5,5"/>
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 1 - Latent Space Representation:</strong> Visualization of how data points are clustered in latent space. 
            The colored dots represent encoded data points, while the dashed circles show distinct clusters of similar items. 
            This demonstrates how the model learns to organize and represent data in a meaningful way.
          </p>
        </div>

        {/* Sampling Process */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
          <h3 className="font-bold mb-4">Sampling Process</h3>
          <svg viewBox="0 0 400 150" className="w-full max-w-2xl mx-auto">
            {/* Distribution Curve */}
            <path
              d={`M 50 120 C 100 120 100 30 200 30 C 300 30 300 120 350 120`}
              fill="none"
              stroke="#60A5FA"
              strokeWidth="2"
            />
            
            {/* Sample Points */}
            {[...Array(8)].map((_, i) => (
              <circle
                key={`sample-${i}`}
                cx={75 + i * 35}
                cy={75 + Math.sin(i) * 30}
                r="4"
                fill="#F87171"
              />
            ))}
            
            {/* Temperature Slider */}
            <line x1="50" y1="140" x2="350" y2="140" stroke="#94A3B8" strokeWidth="2"/>
            <circle cx="200" cy="140" r="6" fill="#34D399"/>
            <text x="175" y="130" fontSize="12">Temperature</text>
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 2 - Sampling Process:</strong> Illustration of how new outputs are generated through sampling. 
            The blue curve represents the learned probability distribution, red dots show sample points, 
            and the temperature slider controls randomness in the sampling process.
          </p>
        </div>

        {/* Context Window */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
          <h3 className="font-bold mb-4">Context Window</h3>
          <svg viewBox="0 0 400 100" className="w-full max-w-2xl mx-auto">
            {/* Token Boxes */}
            {[...Array(10)].map((_, i) => (
              <g key={`token-${i}`}>
                <rect
                  x={40 * i + 10}
                  y="20"
                  width="35"
                  height="35"
                  rx="5"
                  fill={i < 7 ? "#60A5FA" : "#94A3B8"}
                  opacity={i < 7 ? "1" : "0.5"}
                />
                <text
                  x={40 * i + 27.5}
                  y="40"
                  textAnchor="middle"
                  fill="white"
                  fontSize="12"
                >
                  T{i + 1}
                </text>
              </g>
            ))}
            
            {/* Window Frame */}
            <rect
              x="5"
              y="15"
              width="285"
              height="45"
              fill="none"
              stroke="#34D399"
              strokeWidth="2"
              strokeDasharray="5,5"
            />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 3 - Context Window:</strong> Demonstration of the models context window, showing how tokens (T1-T10) 
            are processed sequentially. The green dashed border indicates the current active context window, 
            with dimmed tokens representing those outside the window.
          </p>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Transformer Architecture</h2>
        
        {/* Basic Transformer Architecture */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
          <p className="text-gray-700 mb-4">Basic Transformer Architecture:</p>
          <svg 
            viewBox="0 0 400 300" 
            className="w-full max-w-2xl mx-auto"
          >
            {/* Input */}
            <rect x="150" y="20" width="100" height="40" rx="5" fill="#60A5FA" />
            <text x="200" y="45" textAnchor="middle" fill="white">Input</text>
            
            {/* Embedding */}
            <rect x="150" y="80" width="100" height="40" rx="5" fill="#34D399" />
            <text x="200" y="105" textAnchor="middle" fill="white">Embedding</text>
            
            {/* Positional Encoding */}
            <rect x="150" y="140" width="100" height="40" rx="5" fill="#F87171" />
            <text x="200" y="165" textAnchor="middle" fill="white">Pos Encoding</text>
            
            {/* Self Attention */}
            <rect x="150" y="200" width="100" height="40" rx="5" fill="#818CF8" />
            <text x="200" y="225" textAnchor="middle" fill="white">Self Attention</text>
            
            {/* Feed Forward */}
            <rect x="150" y="260" width="100" height="40" rx="5" fill="#FCD34D" />
            <text x="200" y="285" textAnchor="middle" fill="white">Feed Forward</text>
            
            {/* Connecting Lines */}
            <path d="M200 60 L200 80" stroke="#94A3B8" strokeWidth="2" />
            <path d="M200 120 L200 140" stroke="#94A3B8" strokeWidth="2" />
            <path d="M200 180 L200 200" stroke="#94A3B8" strokeWidth="2" />
            <path d="M200 240 L200 260" stroke="#94A3B8" strokeWidth="2" />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 4 - Basic Transformer Architecture:</strong> Simplified view of the transformers core components, 
            showing the flow from input through embedding, positional encoding, self-attention, and feed-forward layers.
          </p>
        </div>

        {/* Multi-Head Attention Mechanism */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
          <p className="text-gray-700 mb-4">Multi-Head Attention Mechanism:</p>
          <svg 
            viewBox="0 0 400 200" 
            className="w-full max-w-2xl mx-auto"
          >
            {/* Attention Heads */}
            <g transform="translate(50,20)">
              {[0, 1, 2].map((i) => (
                <>
                  <rect x={i * 100} y="0" width="80" height="30" rx="5" fill="#60A5FA" />
                  <text x={i * 100 + 40} y="20" textAnchor="middle" fill="white">
                    Head {i + 1}
                  </text>
                </>
              ))}
              
              {/* Concat Box */}
              <rect x="100" y="80" width="120" height="30" rx="5" fill="#34D399" />
              <text x="160" y="100" textAnchor="middle" fill="white">Concat</text>
              
              {/* Linear Layer */}
              <rect x="100" y="140" width="120" height="30" rx="5" fill="#F87171" />
              <text x="160" y="160" textAnchor="middle" fill="white">Linear</text>
              
              {/* Connecting Lines */}
              {[0, 1, 2].map((i) => (
                <path 
                  key={`path-${i}`}
                  d={`M${i * 100 + 40} 30 L160 80`} 
                  stroke="#94A3B8" 
                  strokeWidth="2" 
                />
              ))}
              <path d="M160 110 L160 140" stroke="#94A3B8" strokeWidth="2" />
            </g>
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 5 - Multi-Head Attention:</strong> Visualization of parallel attention heads that capture different 
            aspects of relationships in the input, combining their outputs through concatenation and linear transformation.
          </p>
        </div>

        {/* Positional Encoding Pattern */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
          <p className="text-gray-700 mb-4">Positional Encoding Pattern:</p>
          <svg 
            viewBox="0 0 400 100" 
            className="w-full max-w-2xl mx-auto"
          >
            {/* Generate sine wave pattern */}
            <path
              d={`M 0 50 ${Array.from({ length: 40 }, (_, i) => 
                `L ${i * 10} ${50 + Math.sin(i * 0.5) * 30}`
              ).join(' ')}`}
              fill="none"
              stroke="#60A5FA"
              strokeWidth="2"
            />
            <path
              d={`M 0 50 ${Array.from({ length: 40 }, (_, i) => 
                `L ${i * 10} ${50 + Math.cos(i * 0.5) * 20}`
              ).join(' ')}`}
              fill="none"
              stroke="#F87171"
              strokeWidth="2"
            />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 6 - Positional Encoding:</strong> Sinusoidal patterns used to encode position information, 
            showing how different frequencies help the model distinguish between positions in the input sequence.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Self-Attention</h3>
            <p className="text-gray-700">A mechanism that allows the model to weigh the importance of different parts of the input sequence when processing each element, enabling understanding of context and relationships.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Multi-Head Attention</h3>
            <p className="text-gray-700">Multiple parallel attention mechanisms that allow the model to focus on different aspects of the input simultaneously, capturing various types of relationships.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Positional Encoding</h3>
            <p className="text-gray-700">Information added to input embeddings to provide the model with awareness of sequence order, since attention mechanisms themselves are order-agnostic.</p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Neural Network Components</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Embeddings</h3>
            <p className="text-gray-700">Dense vector representations of discrete inputs (like words or tokens), mapping them to continuous space where similar items are closer together.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Layer Normalization</h3>
            <p className="text-gray-700">A technique to stabilize neural network training by normalizing the activations across features, helping models train faster and more reliably.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Feed-Forward Network</h3>
            <p className="text-gray-700">Neural network layers that process each position independently, typically consisting of two linear transformations with a non-linear activation in between.</p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Prompt Engineering Concepts</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Zero-Shot Learning</h3>
            <p className="text-gray-700">The ability of a model to perform tasks without specific examples, relying on instructions in natural language and general knowledge from training.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Few-Shot Learning</h3>
            <p className="text-gray-700">Providing a small number of examples in the prompt to help the model understand the desired format or approach for a task.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Chain-of-Thought</h3>
            <p className="text-gray-700">A prompting technique that encourages the model to break down complex problems into steps and show its reasoning process.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">System Prompt</h3>
            <p className="text-gray-700">Initial instructions that set the context, personality, or behavioral parameters for the models responses throughout a conversation.</p>
          </div>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Model Parameters & Training</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Temperature</h3>
            <p className="text-gray-700">A parameter controlling the randomness of model outputs. Higher values lead to more diverse but potentially less focused responses.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Top-p Sampling</h3>
            <p className="text-gray-700">A sampling method that selects from the smallest set of tokens whose cumulative probability exceeds p, helping balance diversity and quality.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Context Window</h3>
            <p className="text-gray-700">The maximum number of tokens the model can process at once, including both input prompt and generated output.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Token</h3>
            <p className="text-gray-700">The basic unit of text processing in language models, which can be words, subwords, or characters, used to convert text into a format the model can process.</p>
          </div>
        </div>
      </section>

   {/* Neural Network Architecture */}
<div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6">
  <h3 className="font-bold mb-4">Neural Network Architecture</h3>
  <svg 
    viewBox="0 0 400 300" 
    className="w-full max-w-2xl mx-auto"
  >
    {/* Input Layer */}
    <g>
      {[0, 1, 2].map((i) => (
        <circle 
          key={`input-${i}`}
          cx="100" 
          cy={80 + i * 60} 
          r="20" 
          fill="#60A5FA"
        />
      ))}
      <text x="40" y="160" textAnchor="middle" fill="#4B5563">Input Layer</text>
    </g>

    {/* Hidden Layer */}
    <g>
      {[0, 1, 2, 3].map((i) => (
        <circle 
          key={`hidden-${i}`}
          cx="200" 
          cy={60 + i * 50} 
          r="20" 
          fill="#34D399"
        />
      ))}
      <text x="200" y="280" textAnchor="middle" fill="#4B5563">Hidden Layer</text>
    </g>

    {/* Output Layer */}
    <g>
      {[0, 1].map((i) => (
        <circle 
          key={`output-${i}`}
          cx="300" 
          cy={110 + i * 60} 
          r="20" 
          fill="#F87171"
        />
      ))}
      <text x="360" y="160" textAnchor="middle" fill="#4B5563">Output Layer</text>
    </g>

    {/* Connections from Input to Hidden Layer */}
    {[0, 1, 2].map((input) => (
      [0, 1, 2, 3].map((hidden) => (
        <line 
          key={`conn1-${input}-${hidden}`}
          x1="120" 
          y1={80 + input * 60} 
          x2="180" 
          y2={60 + hidden * 50} 
          stroke="#94A3B8" 
          strokeWidth="1"
        />
      ))
    ))}

    {/* Connections from Hidden to Output Layer */}
    {[0, 1, 2, 3].map((hidden) => (
      [0, 1].map((output) => (
        <line 
          key={`conn2-${hidden}-${output}`}
          x1="220" 
          y1={60 + hidden * 50} 
          x2="280" 
          y2={110 + output * 60} 
          stroke="#94A3B8" 
          strokeWidth="1"
        />
      ))
    ))}
  </svg>
  <p className="text-sm text-gray-600 mt-4">
    <strong>Figure 7 - Neural Network Architecture:</strong> Basic structure of a feedforward neural network, 
    demonstrating how information flows from input through hidden layers to output, with each connection 
    representing a learnable weight.
  </p>
</div>
    </div>
  );
}