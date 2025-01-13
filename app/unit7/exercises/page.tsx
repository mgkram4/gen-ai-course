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
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 7: Cybersecurity Defense & Incident Response</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Defense Systems and Incident Response</h2>
        
        <MultipleChoiceQuestion
          question="Which defense component actively blocks malicious traffic without human intervention?"
          options={[
            "Intrusion Detection System (IDS)",
            "Intrusion Prevention System (IPS)",
            "Security Information and Event Management (SIEM)",
            "Network Monitor"
          ]}
          correctAnswer={1}
          explanations={[
            "IDS only detects and alerts about threats, it doesn't block them automatically.",
            "Correct! IPS automatically blocks detected threats in real-time without requiring human intervention.",
            "SIEM collects and analyzes security data but doesn't directly block threats.",
            "Network monitors observe traffic patterns but don't take blocking action."
          ]}
        />

        <MultipleChoiceQuestion
          question="What is the first step in the incident response process after detecting a potential breach?"
          options={[
            "System Recovery",
            "Incident Containment",
            "Incident Analysis",
            "Evidence Collection"
          ]}
          correctAnswer={2}
          explanations={[
            "Recovery comes after the threat has been contained and eliminated.",
            "Containment happens after we understand what we're dealing with.",
            "Correct! We must first analyze the incident to understand its scope and nature before taking action.",
            "Evidence collection is part of analysis but not the first step."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which security approach divides a network into separate security zones?"
          options={[
            "Defense in Depth",
            "Network Segmentation",
            "Access Control Lists",
            "Perimeter Defense"
          ]}
          correctAnswer={1}
          explanations={[
            "Defense in Depth uses multiple security layers but doesn't specifically divide networks.",
            "Correct! Network segmentation divides networks into separate security zones to contain threats.",
            "ACLs control access to resources but don't divide the network.",
            "Perimeter defense protects the network edge but doesn't create internal divisions."
          ]}
        />

        <MultipleChoiceQuestion
          question="In RBAC (Role-Based Access Control), what defines the actions a user can perform?"
          options={[
            "User ID",
            "Department",
            "Role Permissions",
            "Security Clearance"
          ]}
          correctAnswer={2}
          explanations={[
            "User ID identifies the user but doesn't define permissions.",
            "Department might influence role assignment but doesn't directly control permissions.",
            "Correct! Role permissions explicitly define what actions users in that role can perform.",
            "Security clearance is a different access control model."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which type of authentication factor is a fingerprint scan?"
          options={[
            "Something you know",
            "Something you have",
            "Something you are",
            "Something you do"
          ]}
          correctAnswer={2}
          explanations={[
            "Passwords and PINs are things you know.",
            "Security tokens and phones are things you have.",
            "Correct! Biometric data like fingerprints are something you are.",
            "Behavioral patterns are something you do."
          ]}
        />

        <MultipleChoiceQuestion
          question="What security event would trigger an incident response process?"
          options={[
            "Failed login attempt",
            "System update installation",
            "Detected malware infection",
            "User password change"
          ]}
          correctAnswer={2}
          explanations={[
            "Single failed logins are normal events unless part of a pattern.",
            "System updates are routine maintenance, not security incidents.",
            "Correct! A malware infection is a security incident requiring immediate response.",
            "Password changes are normal maintenance tasks."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which system monitors and analyzes security events across an organization?"
          options={[
            "Firewall",
            "Antivirus",
            "SIEM",
            "VPN"
          ]}
          correctAnswer={2}
          explanations={[
            "Firewalls filter traffic but don't analyze security events across systems.",
            "Antivirus software protects against malware but doesn't analyze broader security events.",
            "Correct! SIEM systems collect and analyze security events from multiple sources across an organization.",
            "VPNs provide secure connections but don't monitor security events."
          ]}
        />

        <MultipleChoiceQuestion
          question="During incident recovery, what should be done before restoring systems?"
          options={[
            "User notification",
            "Threat eradication",
            "Log analysis",
            "System update"
          ]}
          correctAnswer={1}
          explanations={[
            "User notification happens after recovery plan is ready.",
            "Correct! The threat must be completely removed before restoring systems to prevent reinfection.",
            "Log analysis is part of the investigation phase.",
            "Updates happen during restoration, not before."
          ]}
        />
      </section>
    </div>
  );
};

export default SecurityExercises;