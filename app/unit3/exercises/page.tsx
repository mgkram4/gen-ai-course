'use client';

import CodeBlock from '@/app/components/CodeBlock';
import React, { useState } from 'react';

interface MultipleChoiceQuestionProps {
  question: string;
  options: string[];
  correctAnswer: number;
}

const MultipleChoiceQuestion: React.FC<MultipleChoiceQuestionProps> = ({ question, options, correctAnswer }) => {
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState<boolean>(false);

  const handleSubmit = () => {
    setShowResult(true);
  };

  return (
    <div className="mb-8 w-full max-w-md mx-auto">
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
      {showResult && (
        <p className={`mt-2 ${selectedAnswer === correctAnswer ? 'text-green-600' : 'text-red-600'}`}>
          {selectedAnswer === correctAnswer ? 'Correct!' : 'Incorrect. Try again!'}
        </p>
      )}
    </div>
  );
};

interface GraphicsExerciseProps {
  question: string;
  inputPlaceholder: string;
  checkAnswer: (input: string) => boolean;
}

const GraphicsExercise: React.FC<GraphicsExerciseProps> = ({ question, inputPlaceholder, checkAnswer }) => {
  const [input, setInput] = useState<string>('');
  const [result, setResult] = useState<boolean | null>(null);

  const handleSubmit = () => {
    setResult(checkAnswer(input));
  };

  return (
    <div className="mb-8 w-full max-w-md mx-auto">
      <h3 className="text-xl font-semibold mb-2">{question}</h3>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={inputPlaceholder}
        className="border p-2 mr-2 w-full mb-2"
      />
      <button
        onClick={handleSubmit}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 w-full"
      >
        Submit
      </button>
      {result !== null && (
        <p className={`mt-2 ${result ? 'text-green-600' : 'text-red-600'}`}>
          {result ? 'Correct!' : 'Incorrect. Try again!'}
        </p>
      )}
    </div>
  );
};

const Unit3Exercises: React.FC = () => {
  return (
    <div className="container mx-auto px-4">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 3: Computer Graphics Exercises</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Multiple Choice Questions</h2>
        
        <MultipleChoiceQuestion
          question="Which color model is commonly used for digital displays?"
          options={["CMYK", "RGB", "HSL", "Grayscale"]}
          correctAnswer={1}
        />

        <MultipleChoiceQuestion
          question="What is the process of converting 3D coordinates to 2D screen coordinates called?"
          options={["Rasterization", "Projection", "Tessellation", "Ray tracing"]}
          correctAnswer={1}
        />

        <MultipleChoiceQuestion
          question="Which graphics primitive is NOT commonly used in vector graphics?"
          options={["Line", "Circle", "Pixel", "Polygon"]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="What type of shader is responsible for determining the final color of a pixel?"
          options={["Vertex shader", "Geometry shader", "Fragment shader", "Compute shader"]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="Which file format is typically used for vector graphics?"
          options={["JPG", "PNG", "SVG", "GIF"]}
          correctAnswer={2}
        />

        <MultipleChoiceQuestion
          question="What technique is used to make 3D objects appear smooth despite having a limited number of polygons?"
          options={["Anti-aliasing", "Texture mapping", "Normal mapping", "Mip mapping"]}
          correctAnswer={2}
        />
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Graphics Calculations</h2>

        <GraphicsExercise
          question="Calculate the aspect ratio of a 1920x1080 display (express as a decimal to 2 places):"
          inputPlaceholder="Enter aspect ratio"
          checkAnswer={(input: string) => parseFloat(input) === 1.78}
        />

        <GraphicsExercise
          question="How many total pixels are in a 4K display (3840x2160)?"
          inputPlaceholder="Enter number of pixels"
          checkAnswer={(input: string) => parseInt(input) === 8294400}
        />

        <GraphicsExercise
          question="Convert RGB(255, 0, 0) to hexadecimal:"
          inputPlaceholder="Enter hex code (include #)"
          checkAnswer={(input: string) => input.toLowerCase() === "#ff0000"}
        />

        <GraphicsExercise
          question="Calculate the rotation matrix determinant: [[cos θ, -sin θ], [sin θ, cos θ]] where θ = 90°"
          inputPlaceholder="Enter determinant"
          checkAnswer={(input: string) => parseFloat(input) === 1}
        />

        <GraphicsExercise
          question="If a screen refreshes at 60 FPS, what is the time per frame in milliseconds?"
          inputPlaceholder="Enter time in ms"
          checkAnswer={(input: string) => parseFloat(input) === 16.67}
        />
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Code Exercises</h2>
        
        <div className="mb-8">
          <h3 className="text-xl mb-4">1. Basic Shape Drawing</h3>
          <p className="mb-4">Implement a function to draw a regular polygon with n sides:</p>
          <CodeBlock code={`
def draw_regular_polygon(ctx, x, y, radius, sides):
    """
    Draw a regular polygon using HTML5 Canvas
    ctx: canvas context
    x, y: center coordinates
    radius: radius of circumscribed circle
    sides: number of sides
    """
    # Your implementation here
    pass

# Example usage:
# draw_regular_polygon(ctx, 100, 100, 50, 6)  # Draws a hexagon
          `} language="python" />
        </div>

        <div className="mb-8">
          <h3 className="text-xl mb-4">2. Color Space Conversion</h3>
          <p className="mb-4">Implement RGB to HSL conversion:</p>
          <CodeBlock code={`
def rgb_to_hsl(r: int, g: int, b: int) -> tuple:
    """
    Convert RGB (0-255) to HSL (0-360°, 0-100%, 0-100%)
    Returns: (hue, saturation, lightness)
    """
    # Your implementation here
    pass

# Test cases:
print(rgb_to_hsl(255, 0, 0))    # Should output approximately (0, 100, 50)
print(rgb_to_hsl(0, 255, 0))    # Should output approximately (120, 100, 50)
          `} language="javascript" />
        </div>

        <div className="mb-8">
          <h3 className="text-xl mb-4">3. 3D Transformation</h3>
          <p className="mb-4">Implement a 3D rotation matrix around the Y axis:</p>
          <CodeBlock code={`
def create_y_rotation_matrix(angle_degrees: float) -> list:
    """
    Create a 3x3 rotation matrix for rotation around Y axis
    angle_degrees: rotation angle in degrees
    Returns: 3x3 matrix as nested list
    """
    # Your implementation here
    pass

# Test case:
matrix = create_y_rotation_matrix(90)
point = [1, 0, 0]
# Rotating [1,0,0] by 90° should give approximately [0,0,-1]
          `} language="python" />
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">Challenge Problems</h2>
        
        <div className="mb-8">
          <h3 className="text-xl mb-4">Ray-Triangle Intersection</h3>
          <p className="mb-4">Implement the Möller–Trumbore ray-triangle intersection algorithm:</p>
          <CodeBlock code={`
def ray_triangle_intersection(
    ray_origin: list,
    ray_direction: list,
    vertex0: list,
    vertex1: list,
    vertex2: list
) -> tuple:
    """
    Implement ray-triangle intersection using Möller–Trumbore algorithm
    Returns: (bool: hit, float: distance) or (False, None) if no hit
    """
    # Your implementation here
    pass

# Test case:
ray_origin = [0, 0, 0]
ray_direction = [0, 0, 1]
triangle = ([0, 1, 2], [1, -1, 2], [-1, -1, 2])
result = ray_triangle_intersection(ray_origin, ray_direction, *triangle)
          `} language="python" />
        </div>
      </section>
    </div>
  );
};

export default Unit3Exercises;