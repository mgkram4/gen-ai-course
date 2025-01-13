import { Binary, Code, Download, FileSearch, Github, Search } from 'lucide-react';

const ProjectShowcase = () => {
  const repoUrl = "https://github.com/yourusername/digital-forensics-toolkit";

  const features = [
    {
      icon: <Binary className="w-6 h-6 text-purple-500" />,
      title: "Binary Analysis",
      description: "Performs deep analysis of binary files, including file signatures, hidden data detection, and entropy analysis."
    },
    {
      icon: <FileSearch className="w-6 h-6 text-green-500" />,
      title: "Metadata Extraction",
      description: "Extracts and analyzes metadata from various file types including images, documents, and system files."
    },
    {
      icon: <Download className="w-6 h-6 text-blue-500" />,
      title: "Data Recovery",
      description: "Recovers deleted files and analyzes file system artifacts using carving techniques."
    },
    {
      icon: <Search className="w-6 h-6 text-red-500" />,
      title: "Pattern Analysis",
      description: "Identifies patterns and anomalies in file structures and content using statistical analysis."
    }
  ];

  const technologies = [
    "Python", "Flask", "PyPDF2", "python-magic",
    "exifread", "binwalk", "scapy", "matplotlib"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <FileSearch className="w-16 h-16 text-purple-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Digital Forensics Toolkit
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            A comprehensive Python-based toolkit for digital forensics analysis and data recovery
          </p>
          <a
            href={repoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-6 py-3 bg-purple-600 text-white rounded-full hover:bg-purple-700 transition-colors"
          >
            <Github className="w-5 h-5 mr-2" />
            View on GitHub
          </a>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {features.map((feature, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-lg hover:shadow-xl transition-shadow border border-gray-100">
              <div className="flex items-center mb-4">
                {feature.icon}
                <h3 className="ml-3 text-lg font-semibold text-gray-900">
                  {feature.title}
                </h3>
              </div>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Technologies Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-12 border border-gray-100">
          <h2 className="text-2xl font-bold text-gray-900 mb-4 flex items-center">
            <Code className="w-6 h-6 mr-2 text-purple-500" />
            Technologies Used
          </h2>
          <div className="flex flex-wrap gap-2">
            {technologies.map((tech, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>

        {/* Getting Started */}
        <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-100">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Getting Started
          </h2>
          <div className="bg-gray-50 rounded p-4">
            <code className="text-sm text-gray-800">
              # Clone the repository
              <br />
              git clone {repoUrl}
              <br />
              cd digital-forensics-toolkit
              <br />
              <br />
              # Set up virtual environment
              <br />
              python -m venv venv
              <br />
              source venv/bin/activate  # or venv\Scripts\activate on Windows
              <br />
              <br />
              # Install dependencies
              <br />
              pip install -r requirements.txt
              <br />
              <br />
              # Run the application
              <br />
              python app.py
            </code>
          </div>
          <p className="mt-4 text-gray-600">
            Access the toolkit at <span className="font-mono bg-gray-100 px-2 py-1 rounded">localhost:5000</span>
          </p>
        </div>

        {/* Feature Details */}
        <div className="mt-12 space-y-8">
          <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-100">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Key Features In-Depth
            </h2>
            
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-purple-600 mb-2">Binary Analysis</h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li>File signature detection and validation</li>
                  <li>Entropy analysis for encryption detection</li>
                  <li>Hidden data and steganography detection</li>
                  <li>Binary pattern matching and analysis</li>
                </ul>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-green-600 mb-2">Metadata Extraction</h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li>EXIF data extraction from images</li>
                  <li>Document metadata analysis</li>
                  <li>File system metadata recovery</li>
                  <li>Timeline reconstruction</li>
                </ul>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-blue-600 mb-2">Data Recovery</h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li>Deleted file recovery</li>
                  <li>File carving from raw data</li>
                  <li>Fragmented file reconstruction</li>
                  <li>File system artifact analysis</li>
                </ul>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-red-600 mb-2">Pattern Analysis</h3>
                <ul className="list-disc list-inside text-gray-600 space-y-1">
                  <li>Statistical analysis of file contents</li>
                  <li>Anomaly detection in data structures</li>
                  <li>Pattern matching and recognition</li>
                  <li>Behavioral analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectShowcase;