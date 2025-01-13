"use client"

import React, { useState } from 'react';

interface MultipleChoiceQuestionProps {
  question: string;
  options: string[];
  correctAnswer: number;
  explanations: string[];
}

const MultipleChoiceQuestion: React.FC<MultipleChoiceQuestionProps> = ({ 
  question, 
  options, 
  correctAnswer,
  explanations
}) => {
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState<boolean>(false);

  const handleSubmit = () => {
    setShowResult(true);
  };

  return (
    <div className="mb-8 w-full max-w-2xl mx-auto">
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
      {showResult && selectedAnswer !== null && (
        <div className="mt-4 p-4 rounded bg-gray-50">
          <p className={`font-semibold ${selectedAnswer === correctAnswer ? 'text-green-600' : 'text-red-600'}`}>
            {selectedAnswer === correctAnswer ? 'Correct!' : 'Incorrect'}
          </p>
          <p className="text-gray-700 mt-2">{explanations[selectedAnswer]}</p>
        </div>
      )}
    </div>
  );
};

const SecurityExercises: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 6: Understanding Cyber Attacks</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Common Attack Types</h2>
        
        <MultipleChoiceQuestion
          question="Which of these SQL injection attacks would attempt to bypass a login form?"
          options={[
            "DROP TABLE users;",
            "' OR '1'='1",
            "SELECT * FROM users;",
            "UPDATE users SET admin=true"
          ]}
          correctAnswer={1}
          explanations={[
            "This attack tries to delete data rather than bypass authentication.",
            "Correct! This attack creates a condition that's always true ('1'='1'), potentially allowing access without valid credentials.",
            "This query only lists users but doesn't help bypass authentication.",
            "This attack attempts to gain admin privileges but doesn't bypass the login itself."
          ]}
        />

        <MultipleChoiceQuestion
          question="What type of attack is occurring when a malicious website makes requests to a banking site using your active session?"
          options={[
            "SQL Injection",
            "Cross-Site Scripting (XSS)",
            "Cross-Site Request Forgery (CSRF)",
            "Session Hijacking"
          ]}
          correctAnswer={2}
          explanations={[
            "SQL injection attacks the database, not the user's session.",
            "XSS involves injecting malicious scripts into web pages, not making unauthorized requests.",
            "Correct! CSRF attacks trick your browser into making unwanted requests to another site using your active session.",
            "Session hijacking involves stealing session tokens, not using an existing session."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which attack exploits vulnerable software to gain control of system memory?"
          options={[
            "Buffer Overflow",
            "Phishing",
            "DNS Spoofing",
            "Password Spraying"
          ]}
          correctAnswer={0}
          explanations={[
            "Correct! Buffer overflow attacks overwrite memory beyond allocated space to execute malicious code.",
            "Phishing is a social engineering attack that tricks users, not a memory exploitation.",
            "DNS spoofing redirects traffic by corrupting DNS records, not memory.",
            "Password spraying is a brute force technique trying common passwords across many accounts."
          ]}
        />

        <MultipleChoiceQuestion
          question="What type of attack floods a network with traffic to make services unavailable?"
          options={[
            "Ransomware",
            "Denial of Service (DoS)",
            "Zero-day exploit",
            "Man-in-the-Middle"
          ]}
          correctAnswer={1}
          explanations={[
            "Ransomware encrypts data for ransom, it doesn't flood networks.",
            "Correct! DoS attacks overwhelm systems with traffic to make them unavailable to legitimate users.",
            "Zero-day exploits are unknown vulnerabilities, not flooding attacks.",
            "MITM attacks intercept communications, they don't flood networks."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which attack uses deceptive emails to trick users into revealing sensitive information?"
          options={[
            "Phishing",
            "Pharming",
            "Sniffing",
            "Spoofing"
          ]}
          correctAnswer={0}
          explanations={[
            "Correct! Phishing uses fraudulent communications to trick users into revealing sensitive data.",
            "Pharming redirects website traffic to fraudulent sites at the DNS level.",
            "Sniffing involves capturing network traffic to steal data.",
            "Spoofing involves impersonating trusted devices or addresses."
          ]}
        />

        <MultipleChoiceQuestion
          question="What attack type injects malicious code into web pages to steal user data?"
          options={[
            "Cross-Site Scripting (XSS)",
            "Path Traversal",
            "ARP Poisoning",
            "Dictionary Attack"
          ]}
          correctAnswer={0}
          explanations={[
            "Correct! XSS injects malicious scripts into web pages that run in users' browsers.",
            "Path traversal accesses files outside intended directories.",
            "ARP poisoning redirects network traffic by corrupting ARP tables.",
            "Dictionary attacks attempt to crack passwords using common words."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which attack modifies network traffic between a user and their intended destination?"
          options={[
            "Backdoor",
            "Rootkit",
            "Man-in-the-Middle",
            "Brute Force"
          ]}
          correctAnswer={2}
          explanations={[
            "Backdoors provide unauthorized access but don't modify traffic.",
            "Rootkits hide malicious software but don't modify network traffic.",
            "Correct! MITM attacks intercept and potentially modify traffic between two parties.",
            "Brute force attacks attempt to guess credentials, not modify traffic."
          ]}
        />
      </section>
    </div>
  );
};

export default SecurityExercises;