export default function Unit3Terminology() {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 3: Computer Graphics Terms</h1>
  
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-blue-500">Fundamental Concepts</h2>
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Pixel</h3>
              <p className="text-gray-700">The smallest addressable element in a digital image or display. Short for picture element, it represents a single point of color.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Resolution</h3>
              <p className="text-gray-700">The number of pixels in each dimension of an image or display, typically expressed as width × height (e.g., 1920×1080).</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Vector Graphics</h3>
              <p className="text-gray-700">Images defined by mathematical equations rather than pixels, allowing them to scale without loss of quality.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Raster Graphics</h3>
              <p className="text-gray-700">Images composed of a grid of pixels, each containing color information. Common in photographs and detailed images.</p>
            </div>
          </div>
        </section>
  
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-blue-500">Color and Display</h2>
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">RGB Color Model</h3>
              <p className="text-gray-700">An additive color model where red, green, and blue light are combined in various proportions to produce other colors.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">CMYK Color Model</h3>
              <p className="text-gray-700">A subtractive color model using cyan, magenta, yellow, and key (black), primarily used in printing.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Color Depth</h3>
              <p className="text-gray-700">The number of bits used to represent the color of each pixel. Higher depth allows more color variations.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Gamma Correction</h3>
              <p className="text-gray-700">The process of adjusting the luminance of display output to account for human perception of brightness.</p>
            </div>
          </div>
        </section>
  
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-blue-500">3D Graphics</h2>
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Vertex</h3>
              <p className="text-gray-700">A point in 3D space defined by x, y, and z coordinates, often used as the corner of a polygon.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Polygon</h3>
              <p className="text-gray-700">A shape composed of vertices connected by edges, forming the basic building blocks of 3D models.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Texture Mapping</h3>
              <p className="text-gray-700">The process of applying 2D images to 3D surfaces to add detail and realism to models.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Normal Vector</h3>
              <p className="text-gray-700">A vector perpendicular to a surface, used for calculating lighting and shading effects.</p>
            </div>
          </div>
        </section>
  
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-blue-500">Rendering Concepts</h2>
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Rasterization</h3>
              <p className="text-gray-700">The process of converting vector graphics or 3D models into pixels for display on a screen.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Ray Tracing</h3>
              <p className="text-gray-700">A rendering technique that simulates the physical behavior of light by tracing rays through a scene.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Shader</h3>
              <p className="text-gray-700">A program that runs on the GPU to determine the final appearance of rendered objects, including colors and lighting effects.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Z-Buffer</h3>
              <p className="text-gray-700">A technique for managing depth information in 3D rendering, determining which objects are visible and which are hidden.</p>
            </div>
          </div>
        </section>
  
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-blue-500">Animation and Motion</h2>
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Keyframe</h3>
              <p className="text-gray-700">A frame in an animation that defines a specific pose or state, with intermediate frames interpolated between keyframes.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Tweening</h3>
              <p className="text-gray-700">The process of generating intermediate frames between keyframes to create smooth animation.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Frame Rate</h3>
              <p className="text-gray-700">The number of frames displayed per second (FPS). Higher rates result in smoother animation.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Motion Blur</h3>
              <p className="text-gray-700">A visual effect that simulates the blur of moving objects, adding realism to animations.</p>
            </div>
          </div>
        </section>
  
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-6 text-blue-500">Graphics Pipeline</h2>
          <div className="grid grid-cols-1 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Vertex Shader</h3>
              <p className="text-gray-700">A program that processes individual vertices, handling transformations and position calculations.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Fragment Shader</h3>
              <p className="text-gray-700">A program that determines the color of each pixel in the final rendered image.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Geometry Shader</h3>
              <p className="text-gray-700">An optional shader stage that can create or modify geometric primitives during rendering.</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Pipeline State</h3>
              <p className="text-gray-700">The configuration of various graphics pipeline stages that determines how objects are rendered.</p>
            </div>
          </div>
        </section>
      </div>
    );
  }