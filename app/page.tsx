import { BeakerIcon, BookOpenIcon } from 'lucide-react';
import Link from 'next/link';

export default function GenAIHomePage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <h1 className="text-5xl font-bold mb-8 text-center text-blue-600">
        Introduction to Generative AI
      </h1>
      
      <div className="mb-12 text-xl text-center text-gray-700">
        <p className="mb-4">
          Welcome to Introduction to Generative AI! This course covers the fundamentals of generative AI technologies,
          from basic concepts to practical applications and deployment strategies.
        </p>
        <p>
          Youll learn about large language models, image generation, audio AI, and more while building real-world
          projects and understanding best practices for responsible AI development.
        </p>
      </div>

      <div className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Course Prerequisites</h2>
        <p className="mb-4 text-gray-700">
          To get the most out of this course, you should have:
        </p>
        <ul className="list-disc list-inside mb-4 text-gray-700">
          <li>Basic Python programming experience</li>
          <li>Familiarity with REST APIs</li>
          <li>Understanding of basic machine learning concepts</li>
          <li>Experience with web development fundamentals</li>
        </ul>
        <p className="text-gray-700">
          Throughout the course, youll use popular AI tools and APIs to build practical applications
          and gain hands-on experience with generative AI technologies.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center mb-4">
            <BookOpenIcon className="w-6 h-6 text-blue-500 mr-2" />
            <h3 className="text-xl font-semibold">Core Learning Path</h3>
          </div>
          <ul className="space-y-2 text-gray-700">
            <li>• AI & Deep Learning Foundations</li>
            <li>• Large Language Models</li>
            <li>• Image Generation</li>
            <li>• Audio & Speech AI</li>
            <li>• Agents & Automation</li>
          </ul>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
          <div className="flex items-center mb-4">
            <BeakerIcon className="w-6 h-6 text-green-500 mr-2" />
            <h3 className="text-xl font-semibold">Practical Projects</h3>
          </div>
          <ul className="space-y-2 text-gray-700">
            <li>• ChatGPT Clone</li>
            <li>• Document QA System</li>
            <li>• Image Generation Studio</li>
            <li>• AI Research Assistant</li>
            <li>• Production AI Service</li>
          </ul>
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Course Modules</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((unit) => (
            <Link href={`/unit${unit}`} key={unit} className="block">
              <div className="p-4 border border-gray-200 rounded-lg hover:border-blue-500 hover:shadow-md transition-all">
                <h3 className="font-semibold text-lg mb-2">Unit {unit}:</h3>
                <p className="text-gray-600">
                  {getUnitDescription(unit)}
                </p>
              </div>
            </Link>
          ))}
        </div>
      </div>

      <div className="text-center">
        <Link href="/unit1" className="inline-block bg-blue-600 text-white text-xl font-semibold px-6 py-3 rounded-lg hover:bg-blue-700 transition duration-200">
          Start Learning!
        </Link>
      </div>
    </div>
  );
}

function getUnitDescription(unit: number): string {
  const descriptions = {
    1: "Foundations of Generative AI",
    2: "Working with Language Models",
    3: "Advanced LLM Concepts",
    4: "Image Generation & Computer Vision",
    5: "Audio & Speech AI",
    6: "AI Agents & Automation",
    7: "Deployment & Production",
    8: "AI Ethics & Safety",
    9: "Business Applications"
  };
  return descriptions[unit as keyof typeof descriptions] || "";
}