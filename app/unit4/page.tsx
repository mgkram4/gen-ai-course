// pages/unit1/index.tsx
import CodeBlock from '../components/CodeBlock';
import Layout from '../layout';


export default function Unit1() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 1: Digital Information</h1>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. Bits and Bytes</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">In digital systems, information is represented using binary digits, or bits. A bit is the smallest unit of digital information and can have only two values: 0 or 1. This binary system forms the foundation of all digital computations and storage.</p>
            <p className="mb-4 text-gray-700">A byte is a unit of digital information that consists of 8 bits. Bytes are commonly used to represent characters in text or small integers. The concept of bytes is crucial in understanding data storage and transmission in computer systems.</p>
            <p className="mb-4 text-gray-700">The relationship between bits and bytes extends to larger units of digital information:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>1 byte = 8 bits</li>
              <li>1 kilobyte (KB) = 1024 bytes</li>
              <li>1 megabyte (MB) = 1024 KB</li>
              <li>1 gigabyte (GB) = 1024 MB</li>
              <li>1 terabyte (TB) = 1024 GB</li>
            </ul>
            <p className="mb-4 text-gray-700">Understanding these relationships is fundamental to working with digital systems, from managing storage on devices to optimizing data transmission over networks.</p>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Converting Bits to Bytes</h4>
            <CodeBlock code={`
def bits_to_bytes(bits):
    return bits // 8

def bytes_to_bits(bytes):
    return bytes * 8

# Test the functions
print(f"64 bits is {bits_to_bytes(64)} bytes")
print(f"8 bytes is {bytes_to_bits(8)} bits")
            `} language="python" />
            <h4 className="text-xl mt-4 mb-2 text-blue-300">Size Converter</h4>
            <CodeBlock code={`
def size_converter(size, from_unit, to_unit):
    units = ['b', 'B', 'KB', 'MB', 'GB', 'TB']
    
    if from_unit not in units or to_unit not in units:
        return "Invalid unit"
    
    from_index = units.index(from_unit)
    to_index = units.index(to_unit)
    
    if from_index == 0 and to_index == 1:  # bits to bytes
        return size / 8
    elif from_index == 1 and to_index == 0:  # bytes to bits
        return size * 8
    else:
        return size * (1024 ** (from_index - to_index))

# Test the function
print(f"1024 KB is {size_converter(1024, 'KB', 'MB')} MB")
print(f"1 GB is {size_converter(1, 'GB', 'MB')} MB")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. Binary Numbers</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Binary is a base-2 number system used in digital systems. It uses only two digits: 0 and 1. This system is fundamental to computer science because it directly corresponds to the on/off states of electronic switches in computer hardware.</p>
            <p className="mb-4 text-gray-700">In binary, each digit represents a power of 2:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>The rightmost digit represents 2^0 (1)</li>
              <li>The next digit represents 2^1 (2)</li>
              <li>Then 2^2 (4), 2^3 (8), and so on</li>
            </ul>
            <p className="mb-4 text-gray-700">Converting between binary and decimal:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>To convert binary to decimal, multiply each digit by its corresponding power of 2 and sum the results.</li>
              <li>To convert decimal to binary, repeatedly divide by 2 and keep track of the remainders, which form the binary number from right to left.</li>
            </ul>
            <p className="mb-4 text-gray-700">Binary arithmetic forms the basis of all computations in digital systems. Understanding binary operations is crucial for low-level programming and hardware design.</p>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Binary Conversion</h4>
            <CodeBlock code={`
def decimal_to_binary(decimal):
    return bin(decimal)[2:]  # [2:] to remove the '0b' prefix

def binary_to_decimal(binary):
    return int(binary, 2)

# Test the functions
print(f"42 in binary is {decimal_to_binary(42)}")
print(f"101010 in decimal is {binary_to_decimal('101010')}")
            `} language="python" />
            <h4 className="text-xl mt-4 mb-2 text-blue-300">Binary Addition</h4>
            <CodeBlock code={`
def add_binary(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]

# Test the function
print(f"1010 + 1011 = {add_binary('1010', '1011')}")
            `} language="python" />
          </div>
        </section>

        {/* Additional sections would follow a similar pattern */}

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Conclusion</h2>
          <p className="mb-4 text-gray-700">This unit has covered the fundamentals of digital information, including:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>The basics of bits and bytes</li>
            <li>Binary and hexadecimal number systems</li>
            <li>Limitations of storing numbers in digital systems</li>
            <li>Techniques for storing and compressing text and data</li>
            <li>The process of converting analog data to digital format</li>
            <li>Concepts of digital copyright and licensing</li>
          </ul>
          <p className="mb-4 text-gray-700">Understanding these concepts is crucial for anyone working with digital systems and forms the foundation for more advanced topics in computer science and information technology. As you progress in your studies, youll find these fundamental principles underlying many complex systems and algorithms.</p>
        </section>
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. Hexadecimal Numbers</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Hexadecimal is a base-16 number system, using 16 distinct symbols: the numbers 0-9 and the letters A-F. Its widely used in computing and digital systems for several reasons:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Compact representation of binary data</li>
              <li>Easier for humans to read and write compared to long binary strings</li>
              <li>Direct correlation with 4-bit binary values (each hexadecimal digit represents 4 bits)</li>
            </ul>
            <p className="mb-4 text-gray-700">The hexadecimal system is particularly useful in contexts such as:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Color representation in web design (e.g., #FF0000 for red)</li>
              <li>Memory addresses in low-level programming</li>
              <li>Error codes and debugging information</li>
            </ul>
            <p className="mb-4 text-gray-700">Understanding hexadecimal and its relationship to binary is crucial for many areas of computer science and digital systems design.</p>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Hexadecimal Conversion</h4>
            <CodeBlock code={`
def decimal_to_hex(decimal):
    return hex(decimal)[2:]  # [2:] to remove the '0x' prefix

def hex_to_decimal(hex_string):
    return int(hex_string, 16)

# Test the functions
print(f"255 in hexadecimal is {decimal_to_hex(255)}")
print(f"ff in decimal is {hex_to_decimal('ff')}")
            `} language="python" />
            <h4 className="text-xl mt-4 mb-2 text-blue-300">Hexadecimal and Binary Conversion</h4>
            <CodeBlock code={`
def hex_to_binary(hex_string):
    return bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)

def binary_to_hex(binary):
    return hex(int(binary, 2))[2:]

# Test the functions
print(f"ff in binary is {hex_to_binary('ff')}")
print(f"11111111 in hexadecimal is {binary_to_hex('11111111')}")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Limitations of Storing Numbers</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Digital systems have inherent limitations when it comes to storing and representing numbers. Understanding these limitations is crucial for developing robust software and avoiding subtle bugs. The main challenges include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li><strong>Floating-point representation:</strong> Can lead to precision issues with decimal numbers due to the finite number of bits used to represent them.</li>
              <li><strong>Integer overflow:</strong> Occurs when an arithmetic operation results in a value outside the range that can be represented with a fixed number of bits.</li>
              <li><strong>Underflow:</strong> Happens when a number is too small to be represented in the available precision, often resulting in unexpected behavior or loss of significance.</li>
            </ul>
            <p className="mb-4 text-gray-700">These limitations arise from the finite nature of computer memory and the binary representation of numbers. They can lead to various issues in computational systems, such as:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Rounding errors in financial calculations</li>
              <li>Unexpected behavior in loop conditions</li>
              <li>Security vulnerabilities if not properly handled</li>
            </ul>
            <p className="mb-4 text-gray-700">Awareness of these limitations is essential for writing correct and robust software, especially in domains requiring high precision or dealing with very large or small numbers.</p>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Floating-Point Limitation</h4>
            <CodeBlock code={`
def demonstrate_float_limitation():
    a = 0.1
    b = 0.2
    c = 0.3
    print(f"0.1 + 0.2 = {a + b}")
    print(f"Is 0.1 + 0.2 equal to 0.3? {a + b == c}")
    print(f"Difference: {(a + b) - c}")

demonstrate_float_limitation()
            `} language="python" />
            <h4 className="text-xl mt-4 mb-2 text-blue-300">Integer Overflow</h4>
            <CodeBlock code={`
def demonstrate_integer_overflow():
    import sys
    max_int = sys.maxsize
    print(f"Maximum integer on this system: {max_int}")
    try:
        result = max_int + 1
        print(f"max_int + 1 = {result}")
    except OverflowError:
        print("OverflowError occurred")

demonstrate_integer_overflow()
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">5. Storing Text in Binary</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Storing text in computer systems involves encoding characters into binary representations. This process is fundamental to how computers handle and process textual data. The most common encoding systems are:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li><strong>ASCII (American Standard Code for Information Interchange):</strong> A 7-bit encoding system that represents 128 characters, including English letters, numbers, and basic punctuation.</li>
              <li><strong>Unicode:</strong> A universal character encoding standard that aims to represent all writing systems in the world. It includes over 140,000 characters.</li>
              <li><strong>UTF-8:</strong> A variable-width encoding that uses 1 to 4 bytes per character. Its backward compatible with ASCII and is widely used on the internet.</li>
            </ul>
            <p className="mb-4 text-gray-700">Understanding text encoding is crucial for:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Internationalization and localization of software</li>
              <li>Efficient storage and transmission of text data</li>
              <li>Avoiding character encoding-related bugs and data corruption</li>
            </ul>
            <p className="mb-4 text-gray-700">The choice of encoding system can significantly impact how text is stored, processed, and displayed in different contexts and across various systems.</p>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Text to Binary Conversion</h4>
            <CodeBlock code={`
def text_to_binary(text):
    return ' '.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    return ''.join(chr(int(byte, 2)) for byte in binary.split())

# Test the functions
message = "Hello, World!"
binary = text_to_binary(message)
print(f"'{message}' in binary: {binary}")
print(f"Binary back to text: {binary_to_text(binary)}")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">6. Converting Analog Data to Binary</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Analog-to-digital conversion (ADC) is a fundamental process in digital signal processing and data acquisition. It involves transforming continuous analog signals into discrete digital values. This process is crucial in various fields, including audio processing, image capture, and sensor data collection.</p>
            <p className="mb-4 text-gray-700">The ADC process typically involves three main steps:</p>
            <ol className="list-decimal list-inside mb-4 text-gray-700">
              <li><strong>Sampling:</strong> Measuring the analog signal at regular intervals.</li>
              <li><strong>Quantization:</strong> Assigning discrete values to the sampled measurements.</li>
              <li><strong>Encoding:</strong> Converting the quantized values into binary.</li>
            </ol>
            <p className="mb-4 text-gray-700">Key concepts in ADC include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li><strong>Sampling rate:</strong> The number of samples taken per second, which affects the frequency range that can be accurately represented.</li>
              <li><strong>Bit depth:</strong> The number of bits used to represent each sample, which determines the resolution and dynamic range of the digital signal.</li>
              <li><strong>Nyquist-Shannon sampling theorem:</strong> States that to accurately represent a signal, the sampling rate must be at least twice the highest frequency component of the signal.</li>
            </ul>
            <p className="mb-4 text-gray-700">Understanding ADC is crucial for working with any system that interfaces between the analog and digital worlds, such as audio recording equipment, digital cameras, and various types of sensors.</p>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Simple Analog to Digital Conversion</h4>
            <CodeBlock code={`
import math

def generate_analog_signal(duration, frequency):
    """Generate an analog signal (sine wave)."""
    return [math.sin(2 * math.pi * frequency * t / duration) for t in range(duration)]

def analog_to_digital(analog_signal, num_bits):
    """Convert analog signal to digital using specified number of bits."""
    max_value = 2 ** num_bits - 1
    return [round((sample + 1) / 2 * max_value) for sample in analog_signal]

# Generate an analog signal
duration = 100
frequency = 1
analog_signal = generate_analog_signal(duration, frequency)

# Convert to digital
num_bits = 3  # Using 3 bits for quantization
digital_signal = analog_to_digital(analog_signal, num_bits)

print(f"First 10 analog values: {analog_signal[:10]}")
print(f"First 10 digital values: {digital_signal[:10]}")
            `} language="python" />
          </div>
        </section>

        {/* Additional sections would follow a similar pattern */}

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Conclusion and Further Study</h2>
          <p className="mb-4 text-gray-700">This unit has provided a comprehensive introduction to the fundamental concepts of digital information. As you progress in your study of computer science and information technology, youll find these principles underlying many advanced topics:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>Digital circuit design and computer architecture</li>
            <li>Data compression algorithms and information theory</li>
            <li>Cryptography and secure communication protocols</li>
            <li>Signal processing and multimedia systems</li>
            <li>Big data analytics and data storage technologies</li>
          </ul>
          <p className="mb-4 text-gray-700">To deepen your understanding, consider exploring these related areas:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>Boolean algebra and logic gates</li>
            <li>Error detection and correction codes</li>
            <li>Information theory and entropy</li>
            <li>Quantum computing and its implications for digital information</li>
          </ul>
          <p className="mb-4 text-gray-700">Remember, the concepts learned in this unit form the foundation of all digital technologies. A solid grasp of these principles will serve you well in any area of computer science or digital technology you choose to pursue.</p>
        </section>
      </div>
    </Layout>
  );
}