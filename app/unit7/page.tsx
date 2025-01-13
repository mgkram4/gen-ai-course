"use client"

import CodeBlock from '../components/CodeBlock';


export default function Unit7() {
  return (

      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 7: Cybersecurity Defense & Incident Response</h1>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. Network Defense Systems</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Understanding Network Security</h3>
            <p className="mb-4 text-gray-700">Network defense involves multiple layers of protection, implementing the concept of Defense in Depth. Each layer serves a specific purpose while working in concert with other security measures:</p>
            
            <h4 className="text-xl font-medium mb-2 text-blue-300">Core Defense Components</h4>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Firewalls - Filter network traffic based on predefined rules and policies</li>
              <li>IDS (Intrusion Detection Systems) - Monitor network traffic for suspicious patterns</li>
              <li>IPS (Intrusion Prevention Systems) - Automatically detect and block potential threats</li>
              <li>Network Monitoring - Track traffic patterns and system behavior</li>
              <li>VPN (Virtual Private Network) - Secure encrypted connections for remote access</li>
              <li>WAF (Web Application Firewall) - Protect web applications from common attacks</li>
            </ul>

            <h4 className="text-xl font-medium mb-2 text-blue-300">Defense in Depth Strategy</h4>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Perimeter Defense: DMZ implementation, edge firewalls, and border routers</li>
              <li>Network Segmentation: VLAN implementation and microsegmentation</li>
              <li>Access Control: Role-based permissions and principle of least privilege</li>
              <li>Continuous Monitoring: Real-time threat detection and analysis</li>
              <li>Data Protection: Encryption at rest and in transit</li>
              <li>Endpoint Security: Anti-malware, host-based firewalls, and EDR solutions</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Advanced IDS Implementation</h3>
            <CodeBlock code={`
class AdvancedIDS:
    def __init__(self):
        self.normal_patterns = {
            'max_login_attempts': 3,
            'allowed_ports': [80, 443, 22, 3306, 5432],
            'allowed_ips': ['192.168.1.0/24', '10.0.0.0/8'],
            'max_bandwidth': 1000000,  # 1 MB/s
            'suspicious_patterns': [
                r'(?i)(union\s+select|drop\s+table|delete\s+from)',
                r'(?i)(exec\s+xp_|exec\s+sp_)',
                r'../../../',
                r'<script>.*?</script>'
            ]
        }
        self.alerts = []
        self.threat_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        
    def monitor_traffic(self, connection):
        """Monitor network connection for suspicious activity"""
        alerts = []
        threat_score = 0
        
        # Port analysis
        if connection['port'] not in self.normal_patterns['allowed_ports']:
            threat_score += self.threat_levels['MEDIUM']
            alerts.append({
                'type': 'SUSPICIOUS_PORT',
                'severity': 'MEDIUM',
                'message': f"Suspicious port access: {connection['port']}"
            })
            
        # IP analysis
        if not self._is_ip_allowed(connection['ip']):
            threat_score += self.threat_levels['HIGH']
            alerts.append({
                'type': 'UNAUTHORIZED_IP',
                'severity': 'HIGH',
                'message': f"Unauthorized IP: {connection['ip']}"
            })
            
        # Bandwidth analysis
        if connection['bandwidth'] > self.normal_patterns['max_bandwidth']:
            threat_score += self.threat_levels['MEDIUM']
            alerts.append({
                'type': 'EXCESSIVE_BANDWIDTH',
                'severity': 'MEDIUM',
                'message': f"Excessive bandwidth: {connection['bandwidth']}"
            })
            
        # Payload analysis
        if 'payload' in connection:
            for pattern in self.normal_patterns['suspicious_patterns']:
                if re.search(pattern, connection['payload']):
                    threat_score += self.threat_levels['CRITICAL']
                    alerts.append({
                        'type': 'MALICIOUS_PAYLOAD',
                        'severity': 'CRITICAL',
                        'message': f"Suspicious pattern detected in payload"
                    })
                    break
        
        return {
            'alerts': alerts,
            'threat_score': threat_score,
            'timestamp': datetime.now().isoformat()
        }
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. Authentication & Access Control</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Advanced Authentication Methods</h3>
            
            <h4 className="text-xl font-medium mb-2 text-blue-300">Multi-Factor Authentication (MFA)</h4>
            <p className="mb-4 text-gray-700">Modern authentication systems should implement multiple verification factors:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Knowledge Factors: Passwords, PINs, security questions</li>
              <li>Possession Factors: Mobile devices, security tokens, smart cards</li>
              <li>Inherence Factors: Biometrics including fingerprints, facial recognition, voice patterns</li>
              <li>Location Factors: GPS coordinates, network location</li>
              <li>Behavioral Factors: Typing patterns, mouse movements, usage patterns</li>
            </ul>

            <h4 className="text-xl mb-2 text-blue-300">Zero Trust Architecture</h4>
            <p className="mb-4 text-gray-700">Implementing Zero Trust principles:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Never Trust, Always Verify: Authenticate every access attempt</li>
              <li>Least Privilege Access: Minimal permissions needed for tasks</li>
              <li>Micro-segmentation: Isolate resources and limit lateral movement</li>
              <li>Real-time Monitoring: Continuous verification of access patterns</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Enhanced RBAC Implementation</h3>
            <CodeBlock code={`
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import jwt
import bcrypt

class EnhancedRBACSystem:
    def __init__(self):
        self.permissions = {
            'admin': ['read', 'write', 'delete', 'configure', 'audit'],
            'security_analyst': ['read', 'audit', 'investigate'],
            'developer': ['read', 'write', 'debug'],
            'user': ['read']
        }
        self.secret_key = 'your-secret-key'  # In production, use secure key management
        self.session_duration = timedelta(hours=8)
        self.mfa_required_roles = ['admin', 'security_analyst']
        
    def authenticate(self, user: User, password: str, mfa_code: Optional[str] = None) -> Dict:
        """Authenticate user with password and optional MFA"""
        if not self._verify_password(password, user.password_hash):
            return {'success': False, 'message': 'Invalid credentials'}
            
        if user.role in self.mfa_required_roles:
            if not mfa_code or not self._verify_mfa(user, mfa_code):
                return {'success': False, 'message': 'MFA required'}
                
        # Generate session token
        token = self._generate_token(user)
        
        # Log authentication event
        self._log_auth_event(user, True)
        
        return {
            'success': True,
            'token': token,
            'permissions': self.permissions[user.role]
        }
    
    def verify_permission(self, token: str, required_action: str) -> bool:
        """Verify if token grants permission for action"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check token expiration
            if datetime.fromtimestamp(payload['exp']) < datetime.now():
                return False
                
            # Check user permissions
            user_role = payload['role']
            return required_action in self.permissions.get(user_role, [])
            
        except jwt.InvalidTokenError:
            return False
    
    def _generate_token(self, user: User) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'exp': datetime.now() + self.session_duration
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _verify_mfa(self, user: User, code: str) -> bool:
        """Verify MFA code - implement actual MFA verification logic"""
        # In production, implement proper MFA verification
        return code == "123456"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against stored hash"""
        return bcrypt.checkpw(
            password.encode('utf-8'),
            password_hash.encode('utf-8')
        )
    
    def _log_auth_event(self, user: User, success: bool):
        """Log authentication events for audit"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'user': user.username,
            'role': user.role,
            'success': success,
            'ip_address': user.last_ip,
            'user_agent': user.user_agent
        }
        # In production, implement secure audit logging
        print(f"Auth event: {event}")
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. Advanced Threat Detection</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Modern Attack Patterns</h3>
            <p className="mb-4 text-gray-700">Contemporary systems must detect sophisticated attack patterns:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Advanced Persistent Threats (APT): Long-term targeted attacks</li>
              <li>Zero-Day Exploits: Attacks using unknown vulnerabilities</li>
              <li>Supply Chain Attacks: Compromised software distributions</li>
              <li>Living Off The Land (LOL): Use of legitimate tools for attacks</li>
              <li>Ransomware: Data encryption and extortion</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Advanced Pattern Detection</h3>
            <CodeBlock code={`
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict
import re
import json

@dataclass
class SecurityEvent:
    timestamp: datetime
    event_type: str
    source_ip: str
    destination_ip: str
    payload: str
    metadata: Dict

class AdvancedThreatDetector:
    def __init__(self):
        self.threat_patterns = {
            'brute_force': {
                'max_attempts': 5,
                'time_window': 300,  # 5 minutes
                'severity': 'HIGH'
            },
            'port_scan': {
                'max_ports': 10,
                'time_window': 60,  # 1 minute
                'severity': 'MEDIUM'
            },
            'sql_injection': {
                'patterns': [
                    r'(?i)(union\s+select|drop\s+table|delete\s+from)',
                    r'(?i)(exec\s+xp_|exec\s+sp_)',
                    r'--[\\s\\r\\n]+',
                    r'(?i)(benchmark|sleep)\\s*\\([^)]+\\)'
                ],
                'severity': 'CRITICAL'
            },
            'xss': {
                'patterns': [
                    r'<script[^>]*>[\\s\\S]*?</script>',
                    r'javascript:[\\s\\S]*\\(',
                    r'on\\w+=["\'][\\s\\S]*?["\']'
                ],
                'severity': 'HIGH'
            },
            'file_inclusion': {
                'patterns': [
                    r'\\.\\./\\.\\./',
                    r'(?i)(include|require)(_once)?\\s*\\(["\']\\s*https?://',
                    r'(?i)php://input'
                ],
                'severity': 'CRITICAL'
            }
        }
        self.events: List[SecurityEvent] = []
        
    def analyze_event(self, event: SecurityEvent) -> List[Dict]:
        """Analyze security event for threats"""
        threats = []
        
        # Store event for pattern analysis
        self.events.append(event)
        self._clean_old_events()
        
        # Check for brute force
        if self._detect_brute_force(event):
            threats.append({
                'type': 'BRUTE_FORCE',
                'severity': self.threat_patterns['brute_force']['severity'],
                'source_ip': event.source_ip,
                'timestamp': event.timestamp
            })
        
        # Check for port scanning
        if self._detect_port_scan(event):
            threats.append({
                'type': 'PORT_SCAN',
                'severity': self.threat_patterns['port_scan']['severity'],
                'source_ip': event.source_ip,
                'timestamp': event.timestamp
            })
        
        # Check payload for various injection attempts
        payload_threats = self._analyze_payload(event.payload)
        threats.extend(payload_threats)
        
        # Check for unusual patterns
        if anomaly := self._detect_anomalies(event):
            threats.append(anomaly)
        
        return threats
    
     def _detect_brute_force(self, event: SecurityEvent) -> bool:
        """Detect potential brute force attacks"""
        if event.event_type != 'LOGIN_ATTEMPT':
            return False
            
        recent_attempts = [
            e for e in self.events
            if e.event_type == 'LOGIN_ATTEMPT'
            and e.source_ip == event.source_ip
            and (event.timestamp - e.timestamp) < timedelta(seconds=self.threat_patterns['brute_force']['time_window'])
        ]
        
        return len(recent_attempts) >= self.threat_patterns['brute_force']['max_attempts']
    
    def _detect_port_scan(self, event: SecurityEvent) -> bool:
        """Detect potential port scanning activity"""
        if event.event_type != 'PORT_ACCESS':
            return False
            
        recent_scans = [
            e for e in self.events
            if e.event_type == 'PORT_ACCESS'
            and e.source_ip == event.source_ip
            and (event.timestamp - e.timestamp) < timedelta(seconds=self.threat_patterns['port_scan']['time_window'])
        ]
        
        unique_ports = {e.metadata.get('port') for e in recent_scans if 'port' in e.metadata}
        return len(unique_ports) >= self.threat_patterns['port_scan']['max_ports']
    
    def _analyze_payload(self, payload: str) -> List[Dict]:
        """Analyze payload for various injection attempts"""
        threats = []
        
        # Check for SQL injection
        for pattern in self.threat_patterns['sql_injection']['patterns']:
            if re.search(pattern, payload):
                threats.append({
                    'type': 'SQL_INJECTION',
                    'severity': self.threat_patterns['sql_injection']['severity'],
                    'pattern': pattern
                })
                break
        
        # Check for XSS
        for pattern in self.threat_patterns['xss']['patterns']:
            if re.search(pattern, payload):
                threats.append({
                    'type': 'XSS',
                    'severity': self.threat_patterns['xss']['severity'],
                    'pattern': pattern
                })
                break
        
        # Check for file inclusion
        for pattern in self.threat_patterns['file_inclusion']['patterns']:
            if re.search(pattern, payload):
                threats.append({
                    'type': 'FILE_INCLUSION',
                    'severity': self.threat_patterns['file_inclusion']['severity'],
                    'pattern': pattern
                })
                break
        
        return threats
    
    def _detect_anomalies(self, event: SecurityEvent) -> Optional[Dict]:
        """Detect unusual patterns or behavior"""
        # Implement machine learning or statistical analysis here
        # This is a simplified example
        if self._is_unusual_timing(event) or self._is_unusual_volume(event):
            return {
                'type': 'ANOMALY',
                'severity': 'MEDIUM',
                'details': 'Unusual activity pattern detected'
            }
        return None
    
    def _clean_old_events(self):
        """Remove events older than the maximum time window"""
        max_window = max(
            self.threat_patterns['brute_force']['time_window'],
            self.threat_patterns['port_scan']['time_window']
        )
        cutoff = datetime.now() - timedelta(seconds=max_window)
        self.events = [e for e in self.events if e.timestamp > cutoff]
            `} language="python" />

          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Enhanced Incident Response</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">SOAR Integration</h3>
            <p className="mb-4 text-gray-700">Security Orchestration, Automation and Response (SOAR) capabilities enhance incident handling:</p>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Automated incident triage and categorization</li>
              <li>Playbook-driven response procedures</li>
              <li>Integration with security tools and systems</li>
              <li>Automated containment actions</li>
              <li>Standardized investigation workflows</li>
            </ul>
          </div>

          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Advanced Incident Handler</h3>
            <CodeBlock code={`
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional
import json
import requests

class IncidentStatus(Enum):
    NEW = "new"
    TRIAGING = "triaging"
    ANALYZING = "analyzing"
    CONTAINING = "containing"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    CLOSED = "closed"

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AdvancedIncidentHandler:
    def __init__(self):
        self.incidents: Dict[int, Dict] = {}
        self.response_team = {
            'security': {'email': 'security@example.com', 'phone': '+1234567890'},
            'network': {'email': 'network@example.com', 'phone': '+1234567891'},
            'systems': {'email': 'systems@example.com', 'phone': '+1234567892'}
        }
        self.playbooks = self._load_playbooks()
        
    def create_incident(self, type: str, severity: IncidentSeverity, description: str, 
                       source_data: Dict) -> int:
        """Create new incident record with automated triage"""
        incident_id = len(self.incidents) + 1
        
        # Analyze source data for initial categorization
        category = self._categorize_incident(type, source_data)
        
        incident = {
            'id': incident_id,
            'type': type,
            'category': category,
            'severity': severity,
            'description': description,
            'status': IncidentStatus.NEW,
            'timeline': [{
                'timestamp': datetime.now(),
                'status': IncidentStatus.NEW,
                'action': 'Incident created',
                'details': 'Automated creation based on alert'
            }],
            'affected_systems': [],
            'containment_actions': [],
            'evidence': [],
            'source_data': source_data
        }
        
        self.incidents[incident_id] = incident
        
        # Start automated response
        self._initiate_response(incident)
        
        return incident_id
    
    def _initiate_response(self, incident: Dict):
        """Initialize automated response based on playbooks"""
        playbook = self._get_playbook(incident['category'])
        if not playbook:
            return
            
        # Execute immediate actions
        for action in playbook['immediate_actions']:
            self._execute_action(incident, action)
        
        # Notify appropriate team members
        self._notify_team(incident, playbook['notification_template'])
        
        # Update status to triaging
        self.update_status(
            incident['id'],
            IncidentStatus.TRIAGING,
            'Automated triage initiated'
        )
    
    def update_status(self, incident_id: int, new_status: IncidentStatus, action: str,
                     details: Optional[Dict] = None):
        """Update incident status with validation"""
        if incident_id not in self.incidents:
            raise ValueError("Incident not found")
            
        incident = self.incidents[incident_id]
        current_status = incident['status']
        
        # Validate status transition
        if not self._is_valid_transition(current_status, new_status):
            raise ValueError(f"Invalid status transition: {current_status} -> {new_status}")
        
        incident['status'] = new_status
        
        # Record the status change
        incident['timeline'].append({
            'timestamp': datetime.now(),
            'status': new_status,
            'action': action,
            'details': details or {}
        })
        
        # Execute status-specific actions
        self._handle_status_change(incident, new_status)
        
        # Notify team
        self._notify_team(incident)
    
    def add_evidence(self, incident_id: int, evidence_type: str, content: str,
                    metadata: Optional[Dict] = None):
        """Add evidence to incident record"""
        if incident_id not in self.incidents:
            raise ValueError("Incident not found")
            
        evidence = {
            'timestamp': datetime.now(),
            'type': evidence_type,
            'content': content,
            'metadata': metadata or {},
            'hash': self._calculate_hash(content)
        }
        
        self.incidents[incident_id]['evidence'].append(evidence)
    
    def generate_report(self, incident_id: int) -> Dict:
        """Generate comprehensive incident report"""
        if incident_id not in self.incidents:
            raise ValueError("Incident not found")
            
        incident = self.incidents[incident_id]
        
        return {
            'incident_summary': {
                'id': incident['id'],
                'type': incident['type'],
                'category': incident['category'],
                'severity': incident['severity'],
                'status': incident['status'],
                'creation_date': incident['timeline'][0]['timestamp'],
                'resolution_date': self._get_resolution_date(incident)
            },
            'timeline': incident['timeline'],
            'affected_systems': incident['affected_systems'],
            'containment_actions': incident['containment_actions'],
            'evidence_summary': self._summarize_evidence(incident['evidence']),
            'lessons_learned': self._generate_lessons_learned(incident)
        }
            `} language="python" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">5. Advanced System Recovery</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Comprehensive Recovery Process</h3>
            <p className="mb-4 text-gray-700">Modern system recovery requires a systematic approach:</p>
            
            <h4 className="text-xl font-medium mb-2 text-blue-300">Initial Assessment</h4>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Damage assessment and scope determination</li>
              <li>Impact analysis on business operations</li>
              <li>Recovery priority classification</li>
              <li>Resource requirement identification</li>
            </ul>

            <h4 className="text-xl font-medium mb-2 text-blue-300">Recovery Steps</h4>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>System isolation and containment verification</li>
              <li>Backup validation and integrity checks</li>
              <li>Clean system reconstruction from verified images</li>
              <li>Security patches and updates implementation</li>
              <li>Configuration hardening and validation</li>
              <li>Data restoration and integrity verification</li>
              <li>System testing and security validation</li>
              <li>Monitoring implementation for reoccurrence</li>
            </ul>

            <h4 className="text-xl font-medium mb-2 text-blue-300">Post-Recovery Actions</h4>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Documentation of recovery procedures</li>
              <li>Root cause analysis and lesson learning</li>
              <li>Security control enhancement recommendations</li>
              <li>Team training and procedure updates</li>
              <li>Long-term monitoring strategy implementation</li>
            </ul>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-3xl font-semibold mb-4 text-blue-500">6. Security Metrics and Reporting</h2>
          <div className="mb-6">
            <h3 className="text-2xl font-medium mb-2 text-blue-400">Key Performance Indicators</h3>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Mean Time to Detect (MTTD) security incidents</li>
              <li>Mean Time to Respond (MTTR) to security incidents</li>
              <li>Number of false positives vs true positives</li>
              <li>Security control effectiveness measurements</li>
              <li>Incident resolution rates and trends</li>
              <li>System vulnerability metrics</li>
              <li>Security training completion rates</li>
            </ul>

            <h4 className="text-xl font-medium mb-2 text-blue-300">Automated Reporting</h4>
            <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
              <li>Real-time security dashboards</li>
              <li>Automated compliance reporting</li>
              <li>Incident trend analysis</li>
              <li>Risk assessment scorecards</li>
              <li>Security posture visualization</li>
            </ul>
          </div>
        </section>
      </div>

  );
}