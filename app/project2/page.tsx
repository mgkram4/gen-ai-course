import { BarChart, Brain, Code, Github, Mail, Users } from 'lucide-react';

const ProjectShowcase = () => {
  const repoUrl = "https://github.com/yourusername/cyberphish-defender-py";

  const features = [
    {
      icon: <Mail className="w-6 h-6 text-blue-500" />,
      title: "Email Analysis Engine",
      description: "Uses scikit-learn and NLTK to analyze email content and headers for phishing indicators with machine learning classification."
    },
    {
      icon: <Users className="w-6 h-6 text-green-500" />,
      title: "Training Module",
      description: "Generate simulated phishing emails based on templates and track user responses through a simple web interface."
    },
    {
      icon: <Brain className="w-6 h-6 text-purple-500" />,
      title: "Pattern Recognition",
      description: "Implements multiple analysis techniques including regex patterns, URL checking, and content similarity scoring."
    },
    {
      icon: <BarChart className="w-6 h-6 text-orange-500" />,
      title: "Performance Tracking",
      description: "Track and visualize detection accuracy and training results using matplotlib and seaborn."
    }
  ];

  const technologies = [
    "Python", "Flask", "scikit-learn", "NLTK",
    "Beautiful Soup", "Matplotlib", "Pandas", "Regular Expressions"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <Mail className="w-16 h-16 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            CyberPhish Defender
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            Python-Based Phishing Detection and Training Platform
          </p>
          <a
            href={repoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors"
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
            <Code className="w-6 h-6 mr-2 text-blue-500" />
            Technologies Used
          </h2>
          <div className="flex flex-wrap gap-2">
            {technologies.map((tech, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
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
              cd cyberphish-defender-py
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
              # Download NLTK data
              <br />
              python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
              <br />
              <br />
              # Run the application
              <br />
              python app.py
            </code>
          </div>
          <p className="mt-4 text-gray-600">
            Access the application at <span className="font-mono bg-gray-100 px-2 py-1 rounded">localhost:5000</span>
          </p>
        </div>

        {/* Project Structure Preview */}
        <div className="bg-white rounded-lg shadow-lg p-6 mt-6 border border-gray-100">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Project Structure
          </h2>
          <div className="bg-gray-50 rounded p-4">
            <code className="text-sm text-gray-800">
              cyberphish-defender-py/
              <br />
              ├── app.py                 # Main Flask application
              <br />
              ├── requirements.txt       # Project dependencies
              <br />
              ├── static/               # Static files (CSS, JS)
              <br />
              ├── templates/            # HTML templates
              <br />
              ├── models/               # ML models and training
              <br />
              │   ├── classifier.py     # Email classification
              <br />
              │   └── trainer.py        # Model training
              <br />
              ├── utils/                # Utility functions
              <br />
              │   ├── email_parser.py   # Email parsing
              <br />
              │   ├── feature_extract.py # Feature extraction
              <br />
              │   └── visualizer.py     # Result visualization
              <br />
              └── data/                 # Training data and results
            </code>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProjectShowcase;