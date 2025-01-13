import CodeBlock from '../components/CodeBlock';

export default function Unit3() {
  return (

      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 3: Computer Graphics and Visualization</h1>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. Fundamentals of Computer Graphics</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Computer graphics involves creating, manipulating, and storing visual content using computers. The field encompasses both 2D and 3D graphics, with applications ranging from user interfaces to virtual reality.</p>
            <p className="mb-4 text-gray-700">Key concepts in computer graphics include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Pixels and resolution</li>
              <li>Color models (RGB, CMYK, HSL)</li>
              <li>Vector vs. raster graphics</li>
              <li>Coordinate systems</li>
              <li>Rendering pipelines</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Basic Pixel Manipulation</h4>
            <CodeBlock code={`
import numpy as np
from PIL import Image

def create_gradient():
    # Create a 256x256 RGB gradient
    width = 256
    height = 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Create a gradient from black to white
            value = int((x + y) / 2)
            image[y, x] = [value, value, value]
    
    return Image.fromarray(image)

# Create and save the gradient
gradient = create_gradient()
gradient.save('gradient.png')
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. 2D Graphics and Drawing</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">2D graphics form the foundation of digital imaging and user interface design. Understanding basic shapes, paths, and transformations is essential for creating effective visual content.</p>
            <p className="mb-4 text-gray-700">Important 2D graphics concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Basic shapes (lines, rectangles, circles)</li>
              <li>Paths and curves</li>
              <li>Transformations (translation, rotation, scaling)</li>
              <li>Fill and stroke properties</li>
              <li>Canvas and SVG rendering</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Drawing Shapes with Canvas</h4>
            <CodeBlock code={`
// Create a simple shape drawing function
function drawShapes(canvas) {
    const ctx = canvas.getContext('2d');
    
    // Rectangle
    ctx.fillStyle = 'blue';
    ctx.fillRect(10, 10, 100, 50);
    
    // Circle
    ctx.beginPath();
    ctx.arc(200, 35, 25, 0, Math.PI * 2);
    ctx.fillStyle = 'red';
    ctx.fill();
    
    // Triangle
    ctx.beginPath();
    ctx.moveTo(300, 10);
    ctx.lineTo(350, 60);
    ctx.lineTo(250, 60);
    ctx.closePath();
    ctx.fillStyle = 'green';
    ctx.fill();
}
            `} language="javascript" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. 3D Graphics and Modeling</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">3D graphics add depth to visual representations, enabling the creation of realistic scenes and interactive environments. Understanding 3D concepts is crucial for game development, CAD, and visualization.</p>
            <p className="mb-4 text-gray-700">Key 3D graphics concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Vertices, edges, and faces</li>
              <li>3D transformations and matrices</li>
              <li>Lighting and shading</li>
              <li>Texture mapping</li>
              <li>Camera perspectives</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Basic 3D Transformations</h4>
            <CodeBlock code={`
import numpy as np

def rotate_3d_point(point, angle_x, angle_y, angle_z):
    # Create rotation matrices
    def rotation_matrix(angle, axis):
        c = np.cos(angle)
        s = np.sin(angle)
        if axis == 'x':
            return np.array([[1, 0, 0],
                           [0, c, -s],
                           [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s],
                           [0, 1, 0],
                           [-s, 0, c]])
        else:  # z
            return np.array([[c, -s, 0],
                           [s, c, 0],
                           [0, 0, 1]])
    
    # Apply rotations
    point = np.dot(rotation_matrix(angle_x, 'x'), point)
    point = np.dot(rotation_matrix(angle_y, 'y'), point)
    point = np.dot(rotation_matrix(angle_z, 'z'), point)
    
    return point

# Example usage
point = np.array([1, 0, 0])
rotated = rotate_3d_point(point, np.pi/4, 0, 0)
print(f"Original point: {point}")
print(f"Rotated point: {rotated}")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Data Visualization</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Data visualization transforms raw data into meaningful visual representations, making complex information more accessible and understandable.</p>
            <p className="mb-4 text-gray-700">Important visualization concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Chart types and their uses</li>
              <li>Visual encoding principles</li>
              <li>Interactive visualizations</li>
              <li>Color theory in visualization</li>
              <li>Data-to-visual mapping</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Creating a Basic Chart</h4>
            <CodeBlock code={`
import matplotlib.pyplot as plt
import numpy as np

def create_visualization(data, labels):
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    ax1.bar(labels, data)
    ax1.set_title('Bar Chart')
    ax1.set_ylabel('Values')
    
    # Pie chart
    ax2.pie(data, labels=labels, autopct='%1.1f%%')
    ax2.set_title('Pie Chart')
    
    return fig

# Example usage
data = [30, 20, 25, 15]
labels = ['A', 'B', 'C', 'D']
fig = create_visualization(data, labels)
plt.savefig('visualization.png')
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">5. Animation and Interaction</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Animation and interaction bring graphics to life, creating engaging user experiences and dynamic visualizations.</p>
            <p className="mb-4 text-gray-700">Key animation concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Keyframe animation</li>
              <li>Tweening and easing</li>
              <li>Event handling</li>
              <li>Animation timing</li>
              <li>Performance optimization</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Simple Animation</h4>
            <CodeBlock code={`
class AnimatedCircle {
    constructor(canvas) {
        this.ctx = canvas.getContext('2d');
        this.x = 0;
        this.y = canvas.height / 2;
        this.radius = 20;
        this.dx = 2;
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
        this.ctx.beginPath();
        this.ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        this.ctx.fillStyle = 'purple';
        this.ctx.fill();
        this.ctx.closePath();
    }
    
    update() {
        if (this.x + this.radius > this.ctx.canvas.width || this.x - this.radius < 0) {
            this.dx = -this.dx;
        }
        this.x += this.dx;
    }
    
    animate() {
        this.update();
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}
            `} language="javascript" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">6. Advanced Graphics Techniques</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Advanced graphics techniques enable the creation of more sophisticated and realistic visual content.</p>
            <p className="mb-4 text-gray-700">Important advanced concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Ray tracing</li>
              <li>Shader programming</li>
              <li>Particle systems</li>
              <li>Physics-based rendering</li>
              <li>Global illumination</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Basic Ray Tracer</h4>
            <CodeBlock code={`
import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)

class Sphere:
    def __init__(self, center, radius, color):
        self.center = np.array(center)
        self.radius = radius
        self.color = np.array(color)

def ray_sphere_intersection(ray, sphere):
    oc = ray.origin - sphere.center
    a = np.dot(ray.direction, ray.direction)
    b = 2 * np.dot(oc, ray.direction)
    c = np.dot(oc, oc) - sphere.radius ** 2
    discriminant = b ** 2 - 4 * a * c
    
    if discriminant < 0:
        return None
    
    t = (-b - np.sqrt(discriminant)) / (2 * a)
    return t if t > 0 else None

# Example usage
ray = Ray([0, 0, 0], [1, 0, 0])
sphere = Sphere([5, 0, 0], 1, [1, 0, 0])
t = ray_sphere_intersection(ray, sphere)
if t is not None:
    intersection_point = ray.origin + t * ray.direction
    print(f"Intersection at: {intersection_point}")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Conclusion</h2>
          <p className="mb-4 text-gray-700">This unit has covered the fundamental aspects of computer graphics and visualization, including:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>Basic principles of computer graphics</li>
            <li>2D and 3D graphics techniques</li>
            <li>Data visualization methods</li>
            <li>Animation and interaction</li>
            <li>Advanced rendering techniques</li>
          </ul>
          <p className="mb-4 text-gray-700">Understanding these concepts provides a strong foundation for:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>Game development</li>
            <li>Web design and development</li>
            <li>Scientific visualization</li>
            <li>Computer-aided design (CAD)</li>
            <li>Virtual and augmented reality</li>
          </ul>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">7. Image Processing</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Image processing involves manipulating digital images to enhance or extract useful information. This field combines elements of computer vision and signal processing.</p>
            <p className="mb-4 text-gray-700">Key image processing concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Filters and convolutions</li>
              <li>Edge detection</li>
              <li>Color space transformations</li>
              <li>Image enhancement techniques</li>
              <li>Morphological operations</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Basic Image Filters</h4>
            <CodeBlock code={`
import numpy as np
from PIL import Image

def apply_kernel(image, kernel):
    # Convert image to numpy array
    img_array = np.array(image)
    kernel_size = len(kernel)
    padding = kernel_size // 2
    
    # Create padded image
    padded = np.pad(img_array, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    result = np.zeros_like(img_array)
    
    # Apply convolution
    for i in range(padding, padded.shape[0] - padding):
        for j in range(padding, padded.shape[1] - padding):
            for c in range(3):  # RGB channels
                window = padded[i-padding:i+padding+1, j-padding:j+padding+1, c]
                result[i-padding, j-padding, c] = np.sum(window * kernel)
    
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))

# Example kernels
blur_kernel = np.ones((3, 3)) / 9
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# Usage
image = Image.open('input.jpg')
blurred = apply_kernel(image, blur_kernel)
sharpened = apply_kernel(image, sharpen_kernel)
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">8. WebGL and GPU Programming</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">WebGL enables high-performance 3D graphics in web browsers by providing direct access to GPU capabilities. Understanding GPU programming is essential for creating efficient graphics applications.</p>
            <p className="mb-4 text-gray-700">Important WebGL concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Shader programming (GLSL)</li>
              <li>Vertex and fragment shaders</li>
              <li>Buffer objects and attributes</li>
              <li>Texture mapping</li>
              <li>WebGL context and state management</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Basic WebGL Setup</h4>
            <CodeBlock code={`
// Vertex shader
const vertexShaderSource = \`
attribute vec4 aPosition;
attribute vec4 aColor;
varying vec4 vColor;

void main() {
    gl_Position = aPosition;
    vColor = aColor;
}
\`;

// Fragment shader
const fragmentShaderSource = \`
precision mediump float;
varying vec4 vColor;

void main() {
    gl_FragColor = vColor;
}
\`;

function initWebGL(canvas) {
    const gl = canvas.getContext('webgl');
    if (!gl) {
        throw new Error('WebGL not supported');
    }
    
    // Create shader program
    const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
    gl.useProgram(program);
    
    // Create buffers
    const positions = new Float32Array([
        -0.5, -0.5,
         0.5, -0.5,
         0.0,  0.5
    ]);
    
    const colors = new Float32Array([
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0
    ]);
    
    // Set up attributes
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    const aPosition = gl.getAttribLocation(program, 'aPosition');
    gl.enableVertexAttribArray(aPosition);
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);
    
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    
    const aColor = gl.getAttribLocation(program, 'aColor');
    gl.enableVertexAttribArray(aColor);
    gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, 0, 0);
    
    return gl;
}
            `} language="javascript" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">9. Real-time Graphics</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Real-time graphics focuses on generating visual content at interactive frame rates. This is crucial for games, simulations, and interactive visualizations.</p>
            <p className="mb-4 text-gray-700">Essential real-time graphics concepts include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Frame rate optimization</li>
              <li>Level of detail (LOD)</li>
              <li>Culling and visibility determination</li>
              <li>Scene graph management</li>
              <li>Animation systems</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Game Loop Implementation</h4>
            <CodeBlock code={`
class GameLoop {
    constructor() {
        this.lastTime = 0;
        this.objects = [];
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
    }
    
    addObject(object) {
        this.objects.push(object);
    }
    
    update(deltaTime) {
        for (const object of this.objects) {
            object.update(deltaTime);
        }
    }
    
    render(ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        for (const object of this.objects) {
            object.render(ctx);
        }
        
        // Display FPS
        ctx.fillStyle = 'white';
        ctx.fillText(\`FPS: \${Math.round(this.fps)}\`, 10, 20);
    }
    
    loop(currentTime) {
        // Convert to seconds
        currentTime *= 0.001;
        const deltaTime = currentTime - this.lastTime;
        this.lastTime = currentTime;
        
        // Update FPS counter
        this.frameCount++;
        if (currentTime - this.lastFpsUpdate >= 1) {
            this.fps = this.frameCount / (currentTime - this.lastFpsUpdate);
            this.frameCount = 0;
            this.lastFpsUpdate = currentTime;
        }
        
        this.update(deltaTime);
        this.render(ctx);
        
        requestAnimationFrame((time) => this.loop(time));
    }
    
    start() {
        this.lastTime = performance.now() * 0.001;
        requestAnimationFrame((time) => this.loop(time));
    }
}
            `} language="javascript" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">10. Future Trends</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Emerging Technologies</h3>
            <p className="mb-4 text-gray-700">The field of computer graphics continues to evolve with new technologies and techniques. Some important emerging trends include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Real-time ray tracing</li>
              <li>Machine learning in graphics</li>
              <li>Advanced VR/AR systems</li>
              <li>Neural rendering</li>
              <li>Quantum computing applications</li>
            </ul>
            <p className="mb-4 text-gray-700">These developments are shaping the future of:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Entertainment and gaming</li>
              <li>Scientific visualization</li>
              <li>Digital art and design</li>
              <li>Medical imaging</li>
              <li>Industrial applications</li>
            </ul>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Review and Practice</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Key Takeaways</h3>
            <p className="mb-4 text-gray-700">Throughout this unit, weve covered:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Fundamental graphics concepts and terminology</li>
              <li>2D and 3D graphics programming techniques</li>
              <li>Data visualization and animation</li>
              <li>Image processing and filtering</li>
              <li>GPU programming and optimization</li>
              <li>Real-time graphics systems</li>
              <li>Emerging trends and technologies</li>
            </ul>
            <p className="mb-4 text-gray-700">These concepts provide a strong foundation for further study in specialized areas such as game development, scientific visualization, or computer vision.</p>
          </div>
        </section>
      </div>

  );
}