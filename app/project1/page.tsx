import { Activity, AlertTriangle, Code, Github, RefreshCw, Shield } from 'lucide-react';

const ProjectShowcase = () => {
  const repoUrl = "https://github.com/yourusername/cybersecurity-monitoring";

  const features = [
    {
      icon: <Shield className="w-6 h-6 text-blue-500" />,
      title: "Real-time Threat Detection",
      description: "Monitors and detects security threats including brute force attempts, SQL injections, and port scanning in real-time."
    },
    {
      icon: <AlertTriangle className="w-6 h-6 text-yellow-500" />,
      title: "Multiple Attack Pattern Recognition",
      description: "Identifies various attack patterns including XSS, file inclusion attempts, and suspicious network behavior."
    },
    {
      icon: <Activity className="w-6 h-6 text-green-500" />,
      title: "Visual Security Dashboard",
      description: "Interactive dashboard showing real-time security status, active threats, and system metrics."
    },
    {
      icon: <RefreshCw className="w-6 h-6 text-purple-500" />,
      title: "Automated Response",
      description: "Configurable thresholds and automated threat detection with customizable response actions."
    }
  ];

  const technologies = [
    "Python", "Flask", "JavaScript", "HTML/CSS",
    "Real-time Monitoring", "Security Patterns", "Testing"
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Cybersecurity Monitoring System
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            An educational project demonstrating real-time security monitoring and threat detection
          </p>
          <a
            href={repoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-6 py-3 bg-black text-white rounded-full hover:bg-gray-800 transition-colors"
          >
            <Github className="w-5 h-5 mr-2" />
            View on GitHub
          </a>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {features.map((feature, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow">
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
        <div className="bg-white rounded-lg shadow p-6 mb-12">
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
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Getting Started
          </h2>
          <div className="bg-gray-50 rounded p-4">
            <code className="text-sm text-gray-800">
              # Clone the repository
              <br />
              git clone {repoUrl}
              <br />
              cd cybersecurity-monitoring
              <br />
              <br />
              # Set up virtual environment
              <br />
              python3 -m venv venv
              <br />
              source venv/bin/activate
              <br />
              <br />
              # Install dependencies
              <br />
              pip install -r requirements.txt
              <br />
              <br />
              # Run the application
              <br />
              python app/app.py
            </code>
          </div>
          <p className="mt-4 text-gray-600">
            Visit <span className="font-mono bg-gray-100 px-2 py-1 rounded">localhost:5000</span> to access the dashboard
          </p>
        </div>
      </div>
    </div>
  );
};

export default ProjectShowcase;