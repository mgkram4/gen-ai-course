export default function Unit2Terminology() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 2: Language Models - Key Terms</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">LLM Providers</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">OpenAI</h3>
            <p className="text-gray-700">A leading AI research company offering GPT-4 and GPT-3.5 models through API access, known for specialized models and function calling capabilities with pay-per-token pricing.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Anthropic</h3>
            <p className="text-gray-700">Provider of Claude and Claude 2 models, featuring longer context windows, advanced reasoning capabilities, and a focus on constitutional AI principles.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Open Source Models</h3>
            <p className="text-gray-700">Including LLaMA and its derivatives, offering local deployment options, community-driven development, and customization flexibility.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Cloud Providers</h3>
            <p className="text-gray-700">Major platforms like Google (PaLM, Gemini), Azure OpenAI Service, and AWS Bedrock, focusing on enterprise integration and scalability.</p>
          </div>
        </div>

        {/* Provider Ecosystem Diagram */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">LLM Provider Ecosystem</h3>
          <svg viewBox="0 0 800 400" className="w-full max-w-3xl mx-auto">
            {/* Proprietary Providers */}
            <rect x="50" y="50" width="300" height="150" rx="5" fill="#60A5FA" />
            <text x="200" y="85" textAnchor="middle" fill="white" className="font-bold">Proprietary Providers</text>
            <rect x="70" y="100" width="120" height="80" rx="3" fill="#3B82F6" />
            <text x="130" y="135" textAnchor="middle" fill="white" className="text-xs">OpenAI</text>
            <text x="130" y="155" textAnchor="middle" fill="white" className="text-xs">GPT-4, GPT-3.5</text>
            <rect x="210" y="100" width="120" height="80" rx="3" fill="#3B82F6" />
            <text x="270" y="135" textAnchor="middle" fill="white" className="text-xs">Anthropic</text>
            <text x="270" y="155" textAnchor="middle" fill="white" className="text-xs">Claude, Claude 2</text>

            {/* Cloud Providers */}
            <rect x="50" y="220" width="300" height="150" rx="5" fill="#34D399" />
            <text x="200" y="255" textAnchor="middle" fill="white" className="font-bold">Cloud Providers</text>
            <rect x="70" y="270" width="80" height="80" rx="3" fill="#2FB344" />
            <text x="110" y="305" textAnchor="middle" fill="white" className="text-xs">Google</text>
            <text x="110" y="325" textAnchor="middle" fill="white" className="text-xs">PaLM/Gemini</text>
            <rect x="160" y="270" width="80" height="80" rx="3" fill="#2FB344" />
            <text x="200" y="305" textAnchor="middle" fill="white" className="text-xs">Azure</text>
            <text x="200" y="325" textAnchor="middle" fill="white" className="text-xs">OpenAI</text>
            <rect x="250" y="270" width="80" height="80" rx="3" fill="#2FB344" />
            <text x="290" y="305" textAnchor="middle" fill="white" className="text-xs">AWS</text>
            <text x="290" y="325" textAnchor="middle" fill="white" className="text-xs">Bedrock</text>

            {/* Open Source Section */}
            <rect x="400" y="50" width="350" height="320" rx="5" fill="#F87171" />
            <text x="575" y="85" textAnchor="middle" fill="white" className="font-bold">Open Source Models</text>
            
            <rect x="420" y="100" width="140" height="80" rx="3" fill="#EF4444" />
            <text x="490" y="135" textAnchor="middle" fill="white" className="text-xs">Meta</text>
            <text x="490" y="155" textAnchor="middle" fill="white" className="text-xs">LLaMA, LLaMA 2</text>
            
            <rect x="590" y="100" width="140" height="80" rx="3" fill="#EF4444" />
            <text x="660" y="135" textAnchor="middle" fill="white" className="text-xs">Community Models</text>
            <text x="660" y="155" textAnchor="middle" fill="white" className="text-xs">Mistral, Falcon</text>
            
            <rect x="420" y="200" width="310" height="150" rx="3" fill="#EF4444" />
            <text x="575" y="235" textAnchor="middle" fill="white" className="text-xs">Fine-tuned Variants</text>
            <text x="575" y="275" textAnchor="middle" fill="white" className="text-xs">Vicuna, Alpaca, OpenHermes</text>
            <text x="575" y="315" textAnchor="middle" fill="white" className="text-xs">Local Deployment Options</text>

            {/* Connecting Lines */}
            <path d="M350 125 L400 125" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M350 295 L400 295" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 3:</strong> Overview of the LLM provider ecosystem, showing the relationships between proprietary providers, cloud services, and open-source models.
          </p>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Application Development</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Message History</h3>
            <p className="text-gray-700">A system for tracking and managing conversation context in chat applications, enabling coherent multi-turn interactions.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">State Management</h3>
            <p className="text-gray-700">Techniques for maintaining conversation and application state between requests in LLM-powered applications.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">LangChain</h3>
            <p className="text-gray-700">A framework for developing LLM applications, providing tools for prompt management, chain operations, and common patterns like summarization and Q&A.</p>
          </div>
        </div>

        {/* Enhanced LangChain Architecture */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">Enhanced LangChain Architecture</h3>
          <svg viewBox="0 0 800 400" className="w-full max-w-3xl mx-auto">
            {/* Core Components Layer */}
            <rect x="50" y="50" width="700" height="80" rx="5" fill="#60A5FA" />
            <text x="400" y="90" textAnchor="middle" fill="white" className="font-bold">Core Components</text>
            <rect x="70" y="70" width="120" height="40" rx="3" fill="#3B82F6" />
            <text x="130" y="95" textAnchor="middle" fill="white" className="text-xs">Prompt Templates</text>
            <rect x="200" y="70" width="120" height="40" rx="3" fill="#3B82F6" />
            <text x="260" y="95" textAnchor="middle" fill="white" className="text-xs">LLM Chain</text>
            <rect x="330" y="70" width="120" height="40" rx="3" fill="#3B82F6" />
            <text x="390" y="95" textAnchor="middle" fill="white" className="text-xs">Memory</text>
            <rect x="460" y="70" width="120" height="40" rx="3" fill="#3B82F6" />
            <text x="520" y="95" textAnchor="middle" fill="white" className="text-xs">Agents</text>
            <rect x="590" y="70" width="120" height="40" rx="3" fill="#3B82F6" />
            <text x="650" y="95" textAnchor="middle" fill="white" className="text-xs">Tools</text>

            {/* Integration Layer */}
            <rect x="50" y="150" width="700" height="80" rx="5" fill="#34D399" />
            <text x="400" y="190" textAnchor="middle" fill="white" className="font-bold">Integration Layer</text>
            <rect x="70" y="170" width="120" height="40" rx="3" fill="#2FB344" />
            <text x="130" y="195" textAnchor="middle" fill="white" className="text-xs">Document Loaders</text>
            <rect x="200" y="170" width="120" height="40" rx="3" fill="#2FB344" />
            <text x="260" y="195" textAnchor="middle" fill="white" className="text-xs">Vector Stores</text>
            <rect x="330" y="170" width="120" height="40" rx="3" fill="#2FB344" />
            <text x="390" y="195" textAnchor="middle" fill="white" className="text-xs">Embeddings</text>
            <rect x="460" y="170" width="120" height="40" rx="3" fill="#2FB344" />
            <text x="520" y="195" textAnchor="middle" fill="white" className="text-xs">Retrievers</text>
            <rect x="590" y="170" width="120" height="40" rx="3" fill="#2FB344" />
            <text x="650" y="195" textAnchor="middle" fill="white" className="text-xs">Output Parsers</text>

            {/* Application Layer */}
            <rect x="50" y="250" width="700" height="80" rx="5" fill="#F87171" />
            <text x="400" y="290" textAnchor="middle" fill="white" className="font-bold">Application Layer</text>
            <rect x="70" y="270" width="120" height="40" rx="3" fill="#EF4444" />
            <text x="130" y="295" textAnchor="middle" fill="white" className="text-xs">Chat Models</text>
            <rect x="200" y="270" width="120" height="40" rx="3" fill="#EF4444" />
            <text x="260" y="295" textAnchor="middle" fill="white" className="text-xs">Q&A Systems</text>
            <rect x="330" y="270" width="120" height="40" rx="3" fill="#EF4444" />
            <text x="390" y="295" textAnchor="middle" fill="white" className="text-xs">Summarization</text>
            <rect x="460" y="270" width="120" height="40" rx="3" fill="#EF4444" />
            <text x="520" y="295" textAnchor="middle" fill="white" className="text-xs">Code Generation</text>
            <rect x="590" y="270" width="120" height="40" rx="3" fill="#EF4444" />
            <text x="650" y="295" textAnchor="middle" fill="white" className="text-xs">Custom Apps</text>

            {/* Connecting Lines */}
            <path d="M400 130 L400 150" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M400 230 L400 250" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 4:</strong> Comprehensive view of the LangChain framework architecture, showing the relationships between core components, integration layer, and application layer.
          </p>
        </div>

        {/* State Management Flow */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">State Management Flow</h3>
          <svg viewBox="0 0 800 200" className="w-full max-w-3xl mx-auto">
            {/* User Input */}
            <rect x="50" y="80" width="120" height="40" rx="5" fill="#60A5FA" />
            <text x="110" y="105" textAnchor="middle" fill="white">User Input</text>

            {/* State Manager */}
            <rect x="250" y="50" width="300" height="100" rx="5" fill="#34D399" />
            <text x="400" y="85" textAnchor="middle" fill="white" className="font-bold">State Manager</text>
            <rect x="270" y="95" width="80" height="40" rx="3" fill="#2FB344" />
            <text x="310" y="120" textAnchor="middle" fill="white" className="text-xs">History</text>
            <rect x="360" y="95" width="80" height="40" rx="3" fill="#2FB344" />
            <text x="400" y="120" textAnchor="middle" fill="white" className="text-xs">Context</text>
            <rect x="450" y="95" width="80" height="40" rx="3" fill="#2FB344" />
            <text x="490" y="120" textAnchor="middle" fill="white" className="text-xs">Memory</text>

            {/* Response Generation */}
            <rect x="630" y="80" width="120" height="40" rx="5" fill="#F87171" />
            <text x="690" y="105" textAnchor="middle" fill="white">Response</text>

            {/* Connecting Arrows */}
            <path d="M170 100 L250 100" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M550 100 L630 100" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M690 120 L690 160 L110 160 L110 120" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" fill="none" />
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 5:</strong> State management flow in LLM applications, showing how user input, context, and memory interact to generate coherent responses.
          </p>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">API Integration Concepts</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">API Wrapper</h3>
            <p className="text-gray-700">A reusable code layer that simplifies interaction with LLM APIs by handling authentication, request formatting, and response processing.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Function Calling</h3>
            <p className="text-gray-700">A capability allowing AI models to invoke predefined functions, enabling them to perform specific actions or retrieve external information.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Chat Completion</h3>
            <p className="text-gray-700">An API endpoint type that processes conversational messages and generates contextually appropriate responses.</p>
          </div>
        </div>

        {/* Enhanced API Integration Flow */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">Advanced API Integration Flow</h3>
          <svg viewBox="0 0 800 250" className="w-full max-w-3xl mx-auto">
            {/* Application Layer */}
            <rect x="50" y="40" width="120" height="60" rx="5" fill="#60A5FA" />
            <text x="110" y="75" textAnchor="middle" fill="white">Application</text>
            <text x="110" y="90" textAnchor="middle" fill="white" className="text-xs">User Interface</text>

            {/* API Wrapper Layer */}
            <rect x="250" y="40" width="120" height="160" rx="5" fill="#34D399" />
            <text x="310" y="65" textAnchor="middle" fill="white">API Wrapper</text>
            <rect x="270" y="80" width="80" height="30" rx="3" fill="#2FB344" />
            <text x="310" y="100" textAnchor="middle" fill="white" className="text-xs">Auth</text>
            <rect x="270" y="120" width="80" height="30" rx="3" fill="#2FB344" />
            <text x="310" y="140" textAnchor="middle" fill="white" className="text-xs">Rate Limiting</text>
            <rect x="270" y="160" width="80" height="30" rx="3" fill="#2FB344" />
            <text x="310" y="180" textAnchor="middle" fill="white" className="text-xs">Error Handling</text>

            {/* LLM Provider */}
            <rect x="450" y="40" width="120" height="160" rx="5" fill="#F87171" />
            <text x="510" y="65" textAnchor="middle" fill="white">LLM Provider</text>
            <rect x="470" y="80" width="80" height="30" rx="3" fill="#EF4444" />
            <text x="510" y="100" textAnchor="middle" fill="white" className="text-xs">Completion</text>
            <rect x="470" y="120" width="80" height="30" rx="3" fill="#EF4444" />
            <text x="510" y="140" textAnchor="middle" fill="white" className="text-xs">Chat</text>
            <rect x="470" y="160" width="80" height="30" rx="3" fill="#EF4444" />
            <text x="510" y="180" textAnchor="middle" fill="white" className="text-xs">Embeddings</text>

            {/* Response Flow */}
            <rect x="650" y="40" width="120" height="160" rx="5" fill="#818CF8" />
            <text x="710" y="65" textAnchor="middle" fill="white">Response</text>
            <rect x="670" y="80" width="80" height="30" rx="3" fill="#6366F1" />
            <text x="710" y="100" textAnchor="middle" fill="white" className="text-xs">Parsing</text>
            <rect x="670" y="120" width="80" height="30" rx="3" fill="#6366F1" />
            <text x="710" y="140" textAnchor="middle" fill="white" className="text-xs">Validation</text>
            <rect x="670" y="160" width="80" height="30" rx="3" fill="#6366F1" />
            <text x="710" y="180" textAnchor="middle" fill="white" className="text-xs">Formatting</text>

            {/* Connecting Arrows */}
            <path d="M170 70 L250 70" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M370 120 L450 120" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M570 120 L650 120" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M770 120 L790 120 L790 220 L110 220 L110 100" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" fill="none" />

            {/* Arrow Marker Definition */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#94A3B8" />
              </marker>
            </defs>
          </svg>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Prompt Engineering Techniques</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Chain of Thought</h3>
            <p className="text-gray-700">A prompting technique that breaks down complex problems into steps, improving reasoning through explicit step-by-step thinking.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Zero-Shot Learning</h3>
            <p className="text-gray-700">The ability to handle tasks without examples by relying on clear instructions and format specifications, best suited for simpler, well-defined tasks.</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-bold mb-2">Few-Shot Learning</h3>
            <p className="text-gray-700">A technique using a small number of examples in the prompt to demonstrate the desired pattern or format for the model to follow.</p>
          </div>
        </div>

        {/* Prompt Engineering Visualization */}
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 mt-6">
          <h3 className="font-bold mb-4">Prompt Engineering Techniques Comparison</h3>
          <svg viewBox="0 0 800 400" className="w-full max-w-3xl mx-auto">
            {/* Zero-Shot Section */}
            <rect x="50" y="50" width="200" height="100" rx="5" fill="#60A5FA" />
            <text x="150" y="85" textAnchor="middle" fill="white" className="font-bold">Zero-Shot</text>
            <text x="150" y="115" textAnchor="middle" fill="white" className="text-xs">Direct Task Description</text>
            <text x="150" y="135" textAnchor="middle" fill="white" className="text-xs">No Examples Needed</text>

            {/* Few-Shot Section */}
            <rect x="300" y="50" width="200" height="300" rx="5" fill="#34D399" />
            <text x="400" y="85" textAnchor="middle" fill="white" className="font-bold">Few-Shot</text>
            <rect x="320" y="100" width="160" height="60" rx="3" fill="#2FB344" />
            <text x="400" y="130" textAnchor="middle" fill="white" className="text-xs">Example 1</text>
            <rect x="320" y="170" width="160" height="60" rx="3" fill="#2FB344" />
            <text x="400" y="200" textAnchor="middle" fill="white" className="text-xs">Example 2</text>
            <rect x="320" y="240" width="160" height="60" rx="3" fill="#2FB344" />
            <text x="400" y="270" textAnchor="middle" fill="white" className="text-xs">Task Input</text>

            {/* Chain of Thought Section */}
            <rect x="550" y="50" width="200" height="300" rx="5" fill="#F87171" />
            <text x="650" y="85" textAnchor="middle" fill="white" className="font-bold">Chain of Thought</text>
            <rect x="570" y="100" width="160" height="40" rx="3" fill="#EF4444" />
            <text x="650" y="125" textAnchor="middle" fill="white" className="text-xs">Step 1: Initial Analysis</text>
            <rect x="570" y="150" width="160" height="40" rx="3" fill="#EF4444" />
            <text x="650" y="175" textAnchor="middle" fill="white" className="text-xs">Step 2: Reasoning</text>
            <rect x="570" y="200" width="160" height="40" rx="3" fill="#EF4444" />
            <text x="650" y="225" textAnchor="middle" fill="white" className="text-xs">Step 3: Intermediate Steps</text>
            <rect x="570" y="250" width="160" height="40" rx="3" fill="#EF4444" />
            <text x="650" y="275" textAnchor="middle" fill="white" className="text-xs">Step 4: Final Conclusion</text>

            {/* Connecting Lines */}
            <path d="M250 100 L300 100" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />
            <path d="M500 100 L550 100" stroke="#94A3B8" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Arrow Marker Definition */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#94A3B8" />
              </marker>
            </defs>
          </svg>
          <p className="text-sm text-gray-600 mt-4">
            <strong>Figure 6:</strong> Comparison of different prompt engineering approaches, showing increasing complexity and structure from zero-shot to chain of thought reasoning.
          </p>
        </div>
      </section>
    </div>
  );
}