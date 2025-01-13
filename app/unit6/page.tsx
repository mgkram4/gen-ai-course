import CodeBlock from '../components/CodeBlock';
import Layout from '../layout';

export default function Unit6() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 6: Understanding and Defending Against Cyber Attacks</h1>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. SQL Injection</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">SQL injection occurs when malicious SQL code is inserted into application inputs to manipulate the database. This type of attack can lead to unauthorized data access, data manipulation, or even complete system compromise.</p>
            <p className="mb-4 text-gray-700">Common examples of SQL injection include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Entering {"`' OR '1'='1`"} in login forms</li>
              <li>Adding {"`; DROP TABLE users;`"} to query strings</li>
              <li>Using UNION statements to extract additional data</li>
            </ul>
            <p className="mb-4 text-gray-700">Prevention strategies focus on proper input validation, parameterized queries, and secure coding practices.</p>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Unsafe vs Safe Database Queries</h4>
            <CodeBlock code={`
# Bad practice - vulnerable to SQL injection
def unsafe_login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    # Execute query directly - UNSAFE!

# Good practice - using parameterized queries
def safe_login(username, password):
    from sqlalchemy import text
    query = text("SELECT * FROM users WHERE username=:username AND password=:password")
    result = connection.execute(query, {"username": username, "password": password})
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. Cross-Site Scripting (XSS)</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">XSS attacks allow attackers to inject malicious scripts into web pages viewed by other users. These scripts can steal session cookies, redirect users to malicious sites, or manipulate page content.</p>
            <p className="mb-4 text-gray-700">Types of XSS attacks include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Stored XSS (persists in database)</li>
              <li>Reflected XSS (reflected off web server)</li>
              <li>DOM-based XSS (occurs in client-side JavaScript)</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Examples</h3>
            <h4 className="text-xl mb-2 text-blue-300">Content Sanitization</h4>
            <CodeBlock code={`
from html import escape

def render_user_content(user_input):
    # Sanitize user input before displaying
    safe_content = escape(user_input)
    return f"<div>{safe_content}</div>"

# Using Content Security Policy headers
from flask import Flask, make_response
app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. Brute Force Protection</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Brute force attacks attempt to guess passwords or encryption keys through exhaustive trial and error. Protection mechanisms include rate limiting, account lockouts, and strong password policies.</p>
            <p className="mb-4 text-gray-700">Key defense strategies include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Implementing rate limiting</li>
              <li>Using strong password policies</li>
              <li>Multi-factor authentication</li>
              <li>Account lockout policies</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Rate Limiting Implementation</h4>
            <CodeBlock code={`
from time import sleep
from functools import wraps

def rate_limit(max_attempts=3, lockout_time=300):
    def decorator(f):
        attempts = {}
        
        @wraps(f)
        def wrapped(username, *args, **kwargs):
            if username in attempts:
                if attempts[username]['count'] >= max_attempts:
                    time_passed = time.time() - attempts[username]['last_attempt']
                    if time_passed < lockout_time:
                        raise Exception("Account temporarily locked")
                    attempts[username]['count'] = 0
                    
            result = f(username, *args, **kwargs)
            
            if not result:  # Failed attempt
                attempts.setdefault(username, {'count': 0})
                attempts[username]['count'] += 1
                attempts[username]['last_attempt'] = time.time()
            
            return result
        return wrapped
    return decorator
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Secure Communication</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Secure communication is essential for protecting data in transit. Man-in-the-Middle (MITM) attacks can intercept and modify traffic between parties.</p>
            <p className="mb-4 text-gray-700">Key security measures include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>TLS/SSL encryption</li>
              <li>Certificate validation</li>
              <li>Public key infrastructure</li>
              <li>Secure protocols</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Secure Socket Implementation</h4>
            <CodeBlock code={`
import ssl
import socket

def create_secure_socket():
    context = ssl.create_default_context()
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_default_certs()
    
    sock = socket.create_connection(('example.com', 443))
    ssock = context.wrap_socket(sock, server_hostname='example.com')
    return ssock
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">5. DoS Protection</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Denial of Service (DoS) attacks attempt to make systems unavailable by overwhelming them with traffic or requests. Protection involves rate limiting, traffic filtering, and resource scaling.</p>
            <p className="mb-4 text-gray-700">Key defense strategies include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Rate limiting and request throttling</li>
              <li>Load balancing</li>
              <li>Traffic filtering</li>
              <li>DDoS mitigation services</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Rate Limiting with Redis</h4>
            <CodeBlock code={`
from redis import Redis
redis_client = Redis()

def request_handler(client_ip):
    pipe = redis_client.pipeline()
    now = time.time()
    key = f'rate_limit:{client_ip}'
    
    # Clean old requests
    pipe.zremrangebyscore(key, 0, now - 60)
    # Add new request
    pipe.zadd(key, {now: now})
    # Count requests
    pipe.zcard(key)
    results = pipe.execute()
    
    request_count = results[-1]
    
    if request_count > 100:  # More than 100 requests per minute
        raise Exception("Rate limit exceeded")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">Conclusion</h2>
          <p className="mb-4 text-gray-700">This unit has covered essential cybersecurity concepts including:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>SQL injection prevention</li>
            <li>Cross-site scripting protection</li>
            <li>Brute force attack mitigation</li>
            <li>Secure communication practices</li>
            <li>DoS attack prevention</li>
          </ul>
          <p className="mb-4 text-gray-700">Further areas of study include:</p>
          <ul className="list-disc list-inside mb-4 text-gray-700">
            <li>Advanced cryptography</li>
            <li>Security frameworks and standards</li>
            <li>Incident response procedures</li>
            <li>Threat modeling techniques</li>
          </ul>
          <p className="mb-4 text-gray-700">Understanding these security concepts and implementing proper defenses is crucial for developing and maintaining secure applications in todays digital landscape.</p>
        </section>
        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">6. Memory-Based Attacks</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Memory-based attacks exploit vulnerabilities in program memory management. Common types include buffer overflows, heap spraying, and use-after-free vulnerabilities.</p>
            <p className="mb-4 text-gray-700">Key prevention strategies include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Input validation and bounds checking</li>
              <li>Memory-safe programming languages</li>
              <li>Address Space Layout Randomization (ASLR)</li>
              <li>Stack canaries and memory protection</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <h4 className="text-xl mb-2 text-blue-300">Safe Buffer Handling</h4>
            <CodeBlock code={`
def safe_string_copy(input_str: str, max_length: int) -> str:
    """Safely copy a string with length validation"""
    if len(input_str) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
    return input_str[:max_length]

# Example of safe array bounds checking
def safe_array_access(arr: list, index: int) -> any:
    """Safely access array elements with bounds checking"""
    if not (0 <= index < len(arr)):
        raise IndexError("Array index out of bounds")
    return arr[index]
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">7. Session Management</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Secure session management is crucial for web applications. Common vulnerabilities include session fixation, hijacking, and token exposure.</p>
            <p className="mb-4 text-gray-700">Essential security measures include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Secure session token generation</li>
              <li>Proper cookie attributes</li>
              <li>Session timeout and rotation</li>
              <li>HTTPS-only sessions</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <CodeBlock code={`
from secrets import token_urlsafe
from time import time

class SecureSessionManager:
    def __init__(self, expiry_minutes=30):
        self.sessions = {}
        self.expiry_minutes = expiry_minutes
    
    def create_session(self, user_id: str) -> str:
        # Generate secure token
        token = token_urlsafe(32)
        
        # Store session with expiry
        self.sessions[token] = {
            'user_id': user_id,
            'created_at': time(),
            'last_accessed': time()
        }
        
        return token
    
    def validate_session(self, token: str) -> bool:
        if token not in self.sessions:
            return False
            
        session = self.sessions[token]
        current_time = time()
        
        # Check expiry
        if current_time - session['last_accessed'] > (self.expiry_minutes * 60):
            del self.sessions[token]
            return False
            
        # Update last accessed time
        session['last_accessed'] = current_time
        return True
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">8. File Upload Security</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">File uploads can introduce significant security risks including malware, server-side execution, and denial of service.</p>
            <p className="mb-4 text-gray-700">Security measures include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>File type validation</li>
              <li>Size restrictions</li>
              <li>Malware scanning</li>
              <li>Safe storage locations</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <CodeBlock code={`
import os
import magic
from werkzeug.utils import secure_filename

class SecureFileUploader:
    def __init__(self, upload_dir, max_size_bytes=5*1024*1024):
        self.upload_dir = upload_dir
        self.max_size_bytes = max_size_bytes
        self.allowed_types = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'application/pdf': '.pdf'
        }
    
    def validate_file(self, file) -> tuple[bool, str]:
        # Check file size
        if file.content_length > self.max_size_bytes:
            return False, "File too large"
        
        # Read first chunk for MIME type detection
        chunk = file.read(2048)
        file.seek(0)  # Reset file pointer
        
        # Detect MIME type
        mime = magic.from_buffer(chunk, mime=True)
        if mime not in self.allowed_types:
            return False, "File type not allowed"
            
        return True, ""
    
    def save_file(self, file) -> str:
        filename = secure_filename(file.filename)
        path = os.path.join(self.upload_dir, filename)
        
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Save file
        file.save(path)
        
        return filename
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">9. API Security</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">API security involves protecting web services from unauthorized access and abuse. Common concerns include authentication, rate limiting, and data exposure.</p>
            <p className="mb-4 text-gray-700">Essential security measures include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Strong authentication</li>
              <li>Request validation</li>
              <li>Rate limiting</li>
              <li>Input sanitization</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <CodeBlock code={`
from functools import wraps
from flask import Flask, request, jsonify
from jwt import decode, InvalidTokenError

app = Flask(__name__)

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "No API key provided"}), 401
            
        if not validate_api_key(api_key):
            return jsonify({"error": "Invalid API key"}), 403
            
        return f(*args, **kwargs)
    return decorated

def validate_jwt(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        try:
            payload = decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
            request.user = payload
        except InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
            
        return f(*args, **kwargs)
    return decorated

@app.route('/api/secure-endpoint', methods=['POST'])
@require_api_key
@validate_jwt
def secure_endpoint():
    # Process request with authenticated user
    return jsonify({"message": "Success"})
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">10. Cryptography Implementation</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">Proper implementation of cryptographic functions is crucial for data security. Common uses include password hashing, data encryption, and secure communication.</p>
            <p className="mb-4 text-gray-700">Key considerations include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Using proven cryptographic libraries</li>
              <li>Proper key management</li>
              <li>Secure random number generation</li>
              <li>Algorithm selection</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <CodeBlock code={`
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecureDataHandler:
    def __init__(self, key_salt: bytes):
        self.key_salt = key_salt
    
    def derive_key(self, password: str) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.key_salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: str, password: str) -> bytes:
        key = self.derive_key(password)
        f = Fernet(key)
        return f.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes, password: str) -> str:
        key = self.derive_key(password)
        f = Fernet(key)
        return f.decrypt(encrypted_data).decode()

# Example usage
handler = SecureDataHandler(key_salt=os.urandom(16))
encrypted = handler.encrypt_data("sensitive data", "password123")
decrypted = handler.decrypt_data(encrypted, "password123")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">11. Security Headers</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Theory</h3>
            <p className="mb-4 text-gray-700">HTTP security headers help protect web applications from various attacks including XSS, clickjacking, and information disclosure.</p>
            <p className="mb-4 text-gray-700">Important security headers include:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>Content-Security-Policy (CSP)</li>
              <li>X-Frame-Options</li>
              <li>X-Content-Type-Options</li>
              <li>Strict-Transport-Security</li>
            </ul>
          </div>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Code Example</h3>
            <CodeBlock code={`
from flask import Flask, Response

app = Flask(__name__)

def add_security_headers(response: Response) -> Response:
    """Add security headers to all responses"""
    
    # Content Security Policy
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self';"
    )
    
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Enable HSTS
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # XSS protection (modern browsers use CSP instead)
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

@app.after_request
def security_headers(response: Response) -> Response:
    return add_security_headers(response)
            `} language="python" />
          </div>
        </section>
      </div>
    </Layout>
  );
}