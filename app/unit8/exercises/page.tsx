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

const PrivacyExercises: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 8: Data Privacy & Protection</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-6 text-blue-500">Data Privacy Assessment</h2>
        
        <MultipleChoiceQuestion
          question="Under GDPR, which of these is NOT a valid legal basis for processing personal data?"
          options={[
            "Consent",
            "Contract fulfillment",
            "Monetary compensation",
            "Legitimate interests"
          ]}
          correctAnswer={2}
          explanations={[
            "Consent is a valid legal basis under GDPR Article 6.",
            "Processing necessary for contract performance is valid under GDPR.",
            "Correct! Paying for data is not a legal basis for processing under GDPR.",
            "Legitimate interests is a valid basis when balanced against individual rights."
          ]}
        />

        <MultipleChoiceQuestion
          question="What is the primary purpose of data minimization?"
          options={[
            "Reduce storage costs",
            "Collect only necessary data",
            "Improve data accuracy",
            "Speed up processing"
          ]}
          correctAnswer={1}
          explanations={[
            "Cost reduction is a benefit but not the primary purpose.",
            "Correct! Data minimization means collecting only data that's necessary for specific purposes.",
            "Accuracy is important but separate from minimization.",
            "Processing speed is not related to data minimization."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which process converts personal data into a format where individuals can't be identified?"
          options={[
            "Encryption",
            "Anonymization",
            "Pseudonymization",
            "Tokenization"
          ]}
          correctAnswer={1}
          explanations={[
            "Encryption protects data but can be reversed with the key.",
            "Correct! Anonymization permanently removes identifying information.",
            "Pseudonymization replaces identifiers but can be reversed.",
            "Tokenization replaces sensitive data with tokens but is reversible."
          ]}
        />

        <MultipleChoiceQuestion
          question="What is required for valid consent under privacy laws?"
          options={[
            "Written documentation",
            "Verbal agreement",
            "Freely given and specific",
            "One-time acknowledgment"
          ]}
          correctAnswer={2}
          explanations={[
            "Written documentation alone doesn't ensure valid consent.",
            "Verbal agreement is not sufficient for valid consent.",
            "Correct! Consent must be freely given, specific, informed, and unambiguous.",
            "One-time acknowledgment doesn't meet consent requirements."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which data subject right allows individuals to transfer their data between services?"
          options={[
            "Right to Access",
            "Right to Portability",
            "Right to Rectification",
            "Right to Erasure"
          ]}
          correctAnswer={1}
          explanations={[
            "Access right only allows viewing data.",
            "Correct! Data portability right enables transfer of data between services.",
            "Rectification allows correction of inaccurate data.",
            "Erasure right enables data deletion."
          ]}
        />

        <MultipleChoiceQuestion
          question="What is Privacy by Design's key principle?"
          options={[
            "Reactive protection",
            "Cost efficiency",
            "Proactive privacy",
            "Minimal security"
          ]}
          correctAnswer={2}
          explanations={[
            "Privacy by Design is not reactive.",
            "Cost efficiency is not a primary principle.",
            "Correct! Privacy by Design emphasizes proactive privacy protection from the start.",
            "Minimal security contradicts Privacy by Design principles."
          ]}
        />

        <MultipleChoiceQuestion
          question="For international data transfers, what is an adequacy decision?"
          options={[
            "Internal policy",
            "Technical measure",
            "Country approval",
            "Security audit"
          ]}
          correctAnswer={2}
          explanations={[
            "Internal policies don't determine adequacy.",
            "Technical measures aren't adequacy decisions.",
            "Correct! An adequacy decision means a country's data protection is deemed adequate by regulators.",
            "Security audits don't determine adequacy."
          ]}
        />

        <MultipleChoiceQuestion
          question="What does a Data Protection Impact Assessment (DPIA) evaluate?"
          options={[
            "System performance",
            "Privacy risks",
            "Employee skills",
            "Business costs"
          ]}
          correctAnswer={1}
          explanations={[
            "DPIAs don't assess system performance.",
            "Correct! DPIAs evaluate privacy risks of processing activities.",
            "Employee skills aren't part of DPIAs.",
            "Business costs aren't the focus of DPIAs."
          ]}
        />

        <MultipleChoiceQuestion
          question="Which is a key requirement for Privacy-Preserving Data Analysis?"
          options={[
            "Maximum data collection",
            "Public data sharing",
            "Differential privacy",
            "Unrestricted access"
          ]}
          correctAnswer={2}
          explanations={[
            "Maximum collection violates privacy principles.",
            "Public sharing increases privacy risks.",
            "Correct! Differential privacy helps protect individual privacy in data analysis.",
            "Unrestricted access compromises privacy."
          ]}
        />

        <MultipleChoiceQuestion
          question="What's required when using third-party data processors?"
          options={[
            "Verbal agreement",
            "Data processing contract",
            "Informal permission",
            "General policy"
          ]}
          correctAnswer={1}
          explanations={[
            "Verbal agreements are insufficient.",
            "Correct! A written data processing agreement is legally required.",
            "Informal permission doesn't meet legal requirements.",
            "General policies alone are inadequate."
          ]}
        />
      </section>
    </div>
  );
};

export default PrivacyExercises;