"use client"

export default function Unit8() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-blue-600">Unit 8: Data Privacy & Protection</h1>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">1. Data Classification & Handling</h2>
        <div className="mb-6">
          <h3 className="text-2xl font-medium mb-2 text-blue-400">Types of Sensitive Data</h3>
          <p className="mb-4 text-gray-700">Understanding different categories of protected data:</p>
          
          <h4 className="text-xl font-medium mb-2 text-blue-300">Personal Data Categories</h4>
          <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
            <li>Personally Identifiable Information (PII): Names, SSNs, addresses</li>
            <li>Protected Health Information (PHI): Medical records, health insurance details</li>
            <li>Financial Data: Credit card numbers, bank account details</li>
            <li>Biometric Data: Fingerprints, facial recognition data</li>
            <li>Online Identifiers: IP addresses, cookie data, device IDs</li>
          </ul>

          <div className="bg-gray-100 p-4 rounded-lg mb-4">
            <h5 className="font-medium mb-2">Definition:</h5>
            <p className="text-gray-700">PII is any data that can be used to identify a specific individual, either directly or in combination with other information.</p>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <h5 className="font-medium mb-2">Real-Life Example:</h5>
            <p className="text-gray-700">A healthcare providers patient database containing names, dates of birth, and medical history. Each record must be protected under HIPAA regulations.</p>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg mb-6">
            <h5 className="text-white font-medium mb-2">Code Example: Data Classification System</h5>
            <pre className="text-gray-100 overflow-x-auto">
{`class DataClassifier {
  constructor() {
    this.sensitivityLevels = {
      PUBLIC: 0,
      INTERNAL: 1,
      CONFIDENTIAL: 2,
      RESTRICTED: 3,
      CRITICAL: 4
    };
  }

  classifyData(data) {
    if (this.containsPII(data)) {
      return this.sensitivityLevels.RESTRICTED;
    }
    // Additional classification logic
    return this.sensitivityLevels.PUBLIC;
  }

  containsPII(data) {
    const piiPatterns = {
      ssn: /\d{3}-\d{2}-\d{4}/,
      email: /[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/i,
      phone: /\d{3}[-.]?\d{3}[-.]?\d{4}/
    };
    
    return Object.values(piiPatterns).some(pattern => 
      pattern.test(data)
    );
  }
}`}
            </pre>
          </div>

          <h4 className="text-xl font-medium mb-2 text-blue-300">Data Sensitivity Levels</h4>
          <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
            <li>Public Data: Marketing materials, public records</li>
            <li>Internal Data: Employee directories, internal procedures</li>
            <li>Confidential Data: Business strategies, customer lists</li>
            <li>Restricted Data: Medical records, financial transactions</li>
            <li>Critical Data: Trade secrets, encryption keys</li>
          </ul>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg mb-6">
          <h5 className="text-white font-medium mb-2">Code Example: Intrusion Detection System</h5>
          <pre className="text-gray-100 overflow-x-auto">
{`class IntrusionDetectionSystem {
  constructor() {
    this.anomalyThreshold = 0.8;
    this.knownPatterns = new Set();
  }

  detectIntrusion(accessPattern) {
    const anomalyScore = this.calculateAnomalyScore(accessPattern);
    
    if (anomalyScore > this.anomalyThreshold) {
      this.triggerAlert({
        timestamp: new Date(),
        pattern: accessPattern,
        score: anomalyScore
      });
      return true;
    }
    return false;
  }

  calculateAnomalyScore(pattern) {
    // Implement machine learning-based anomaly detection
    return Math.random(); // Simplified for example
  }

  triggerAlert(details) {
    console.log('ALERT: Suspicious access pattern detected');
    console.log(details);
    // Implement alert notification system
  }
}`}
          </pre>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">2. Data Privacy Principles</h2>
        <div className="mb-6">
          <h3 className="text-2xl font-medium mb-2 text-blue-400">Core Privacy Concepts</h3>
          
          <div className="bg-gray-100 p-4 rounded-lg mb-4">
            <h5 className="font-medium mb-2">Definition:</h5>
            <p className="text-gray-700">Privacy by Design is an approach that ensures privacy and data protection are embedded into systems and processes from the very beginning, rather than being added as an afterthought.</p>
          </div>

          <h4 className="text-xl font-medium mb-2 text-blue-300">Data Collection Principles</h4>
          <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
            <li>Purpose Limitation: Only collect data for specified purposes</li>
            <li>Data Minimization: Collect only whats necessary</li>
            <li>Storage Limitation: Retain data only as long as needed</li>
            <li>Accuracy: Ensure data is correct and up-to-date</li>
            <li>Transparency: Clear communication about data practices</li>
          </ul>

          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <h5 className="font-medium mb-2">Real-Life Example:</h5>
            <p className="text-gray-700">An e-commerce website implementing data minimization by only collecting essential information for shipping (name, address) and not requiring additional details like gender or birth date that arent necessary for the service.</p>
          </div>

          <div className="bg-gray-800 p-4 rounded-lg mb-6">
            <h5 className="text-white font-medium mb-2">Code Example: Role-Based Access Control</h5>
            <pre className="text-gray-100 overflow-x-auto">
{`class RBACSystem {
  constructor() {
    this.roles = new Map();
    this.userRoles = new Map();
  }

  defineRole(role, permissions) {
    this.roles.set(role, new Set(permissions));
  }

  assignRole(userId, role) {
    if (!this.roles.has(role)) {
      throw new Error('Invalid role');
    }
    this.userRoles.set(userId, role);
  }

  checkPermission(userId, permission) {
    const userRole = this.userRoles.get(userId);
    if (!userRole) return false;
    
    const rolePermissions = this.roles.get(userRole);
    return rolePermissions.has(permission);
  }
}

// Usage example
const rbac = new RBACSystem();
rbac.defineRole('admin', ['read', 'write', 'delete']);
rbac.defineRole('user', ['read']);

rbac.assignRole('user123', 'user');
console.log(rbac.checkPermission('user123', 'read')); // true
console.log(rbac.checkPermission('user123', 'write')); // false`}
            </pre>
          </div>
        </div>
      </section>

      {/* Continue with remaining sections, adding definitions, examples, and code blocks similarly */}
      
      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">3. Data Processing & Storage</h2>
        <div className="bg-gray-800 p-4 rounded-lg mb-6">
          <h5 className="text-white font-medium mb-2">Code Example: Data Encryption Handler</h5>
          <pre className="text-gray-100 overflow-x-auto">
{`class DataEncryptionHandler {
  constructor(encryptionKey) {
    this.key = encryptionKey;
    this.algorithm = 'aes-256-gcm';
  }

  async encrypt(data) {
    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv(this.algorithm, this.key, iv);
    
    let encrypted = cipher.update(data, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  }

  async decrypt(encryptedData) {
    const decipher = crypto.createDecipheriv(
      this.algorithm,
      this.key,
      Buffer.from(encryptedData.iv, 'hex')
    );
    
    decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
    
    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }
}`}
          </pre>
        </div>
      </section>

      
      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">4. Data Subject Rights</h2>
        
        <div className="bg-gray-100 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Definition:</h5>
          <p className="text-gray-700">Data Subject Rights are the fundamental rights individuals have regarding their personal data, including the right to access, correct, delete, and transfer their information.</p>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Real-Life Example:</h5>
          <p className="text-gray-700">A social media platform implementing a Download Your Data feature that allows users to export all their posts, photos, and activity history in a machine-readable format, fulfilling the right to data portability.</p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg mb-6">
          <h5 className="text-white font-medium mb-2">Code Example: Data Subject Rights Handler</h5>
          <pre className="text-gray-100 overflow-x-auto">
{`class DataSubjectRightsHandler {
  constructor(database) {
    this.db = database;
    this.requestQueue = new Map();
    this.RESPONSE_DEADLINE_DAYS = 30;
  }

  async handleAccessRequest(userId) {
    const requestId = this.generateRequestId();
    this.requestQueue.set(requestId, {
      type: 'ACCESS',
      status: 'PENDING',
      userId,
      deadline: this.calculateDeadline()
    });

    try {
      const userData = await this.collectUserData(userId);
      const formattedData = this.formatDataForExport(userData);
      
      this.requestQueue.set(requestId, {
        ...this.requestQueue.get(requestId),
        status: 'COMPLETED'
      });

      return {
        requestId,
        data: formattedData,
        format: 'JSON'
      };
    } catch (error) {
      this.handleRequestError(requestId, error);
    }
  }

  async handleDeletionRequest(userId) {
    const requestId = this.generateRequestId();
    
    try {
      await this.verifyUserIdentity(userId);
      await this.backupDataBeforeDeletion(userId);
      await this.deleteUserData(userId);
      await this.notifyThirdParties(userId);
      
      return {
        requestId,
        status: 'COMPLETED',
        timestamp: new Date()
      };
    } catch (error) {
      this.handleRequestError(requestId, error);
    }
  }

  calculateDeadline() {
    const deadline = new Date();
    deadline.setDate(deadline.getDate() + this.RESPONSE_DEADLINE_DAYS);
    return deadline;
  }

  async verifyUserIdentity(userId) {
    // Implement identity verification logic
  }

  private async notifyThirdParties(userId) {
    // Notify all data processors about deletion
  }
}`}
          </pre>
        </div>

        <div className="mb-6">
          <h3 className="text-2xl font-medium mb-2 text-blue-400">Implementation Best Practices</h3>
          <ul className="list-disc list-inside mb-4 text-gray-700 space-y-2">
            <li>Establish clear verification procedures for rights requests</li>
            <li>Implement automated systems for tracking request deadlines</li>
            <li>Maintain detailed logs of all rights-related actions</li>
            <li>Create standardized response templates</li>
            <li>Regular staff training on handling rights requests</li>
          </ul>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">5. Data Sharing & Transfer</h2>
        
        <div className="bg-gray-100 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Definition:</h5>
          <p className="text-gray-700">Data transfer refers to the movement of personal data between organizations, systems, or countries, requiring specific safeguards and legal mechanisms to ensure continued protection.</p>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Real-Life Example:</h5>
          <p className="text-gray-700">A multinational company implementing Standard Contractual Clauses (SCCs) to legally transfer employee data from its EU offices to its US-based HR system, ensuring GDPR compliance.</p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg mb-6">
          <h5 className="text-white font-medium mb-2">Code Example: Secure Data Transfer System</h5>
          <pre className="text-gray-100 overflow-x-auto">
{`class SecureDataTransfer {
  constructor() {
    this.transferLog = new TransferLog();
    this.encryptionService = new EncryptionService();
  }

  async initiateTransfer(data, recipient) {
    // Validate transfer requirements
    this.validateTransferRequirements(recipient);
    
    // Create transfer record
    const transferId = await this.transferLog.createEntry({
      recipient,
      timestamp: new Date(),
      dataType: this.classifyData(data)
    });

    try {
      // Encrypt data for transfer
      const encryptedPackage = await this.encryptionService
        .encryptForTransfer(data, recipient.publicKey);

      // Perform the transfer
      const transferResult = await this.executeTransfer(
        encryptedPackage,
        recipient
      );

      // Update transfer log
      await this.transferLog.updateStatus(
        transferId,
        'COMPLETED',
        transferResult
      );

      return {
        transferId,
        status: 'SUCCESS',
        timestamp: new Date()
      };

    } catch (error) {
      await this.transferLog.updateStatus(
        transferId,
        'FAILED',
        error.message
      );
      throw new TransferError(error.message);
    }
  }

  validateTransferRequirements(recipient) {
    // Check adequacy decision
    if (this.requiresAdequacyDecision(recipient.country)) {
      this.validateAdequacyDecision(recipient.country);
    }

    // Verify SCCs if needed
    if (this.requiresSCCs(recipient)) {
      this.validateSCCs(recipient);
    }

    // Check transfer impact assessment
    this.validateTransferImpact(recipient);
  }

  async executeTransfer(encryptedPackage, recipient) {
    // Implement secure transfer protocol
    const channel = await SecureChannel.establish(recipient);
    return channel.transfer(encryptedPackage);
  }
}`}
          </pre>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">6. Data Privacy Documentation</h2>
        
        <div className="bg-gray-100 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Definition:</h5>
          <p className="text-gray-700">Privacy documentation encompasses all records, policies, and procedures that demonstrate an organizations compliance with data protection requirements and its privacy program implementation.</p>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Real-Life Example:</h5>
          <p className="text-gray-700">A financial institution maintaining a comprehensive Record of Processing Activities (ROPA) that documents all personal data processing operations, including payment processing, fraud detection, and marketing activities.</p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg mb-6">
          <h5 className="text-white font-medium mb-2">Code Example: Privacy Documentation System</h5>
          <pre className="text-gray-100 overflow-x-auto">
{`class PrivacyDocumentationSystem {
  constructor() {
    this.documentStore = new Map();
    this.versionControl = new VersionControl();
    this.reviewSchedule = new ReviewScheduler();
  }

  async createDocument(type, content) {
    const document = {
      id: this.generateDocumentId(),
      type,
      content,
      version: '1.0',
      created: new Date(),
      lastUpdated: new Date(),
      status: 'DRAFT'
    };

    // Store document
    await this.documentStore.set(document.id, document);

    // Schedule review
    this.reviewSchedule.scheduleReview(document.id, {
      frequency: this.getReviewFrequency(type),
      assignee: this.determineReviewer(type)
    });

    return document;
  }

  async updateDocument(id, changes) {
    const current = await this.documentStore.get(id);
    if (!current) throw new Error('Document not found');

    // Create new version
    const newVersion = this.versionControl.createNewVersion(current);
    
    // Apply changes
    const updated = {
      ...newVersion,
      ...changes,
      lastUpdated: new Date()
    };

    // Store update
    await this.documentStore.set(id, updated);
    
    // Log update
    await this.logDocumentUpdate(id, current, updated);
    
    return updated;
  }

  async generateROPA() {
    const processingActivities = await this.collectProcessingActivities();
    const ropa = {
      id: 'ROPA-' + new Date().getFullYear(),
      activities: processingActivities,
      generated: new Date(),
      approvedBy: null
    };

    return this.createDocument('ROPA', ropa);
  }

  private getReviewFrequency(documentType) {
    const frequencies = {
      'POLICY': 365, // Annual review
      'PROCEDURE': 180, // Bi-annual review
      'ROPA': 90, // Quarterly review
      'DPIA': 180 // Bi-annual review
    };
    return frequencies[documentType] || 365;
  }
}`}
          </pre>
        </div>
      </section>

      <section className="mb-12">
        <h2 className="text-3xl font-semibold mb-4 text-blue-500">7. Privacy-Enhancing Technologies</h2>
        
        <div className="bg-gray-100 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Definition:</h5>
          <p className="text-gray-700">Privacy-Enhancing Technologies (PETs) are specialized technologies designed to protect personal data and enable privacy-preserving data processing, often using advanced cryptographic techniques.</p>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg mb-4">
          <h5 className="font-medium mb-2">Real-Life Example:</h5>
          <p className="text-gray-700">A medical research institution using homomorphic encryption to analyze patient data without decrypting it, allowing researchers to compute statistics while maintaining patient privacy.</p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg mb-6">
          <h5 className="text-white font-medium mb-2">Code Example: Privacy-Preserving Analytics</h5>
          <pre className="text-gray-100 overflow-x-auto">
{`class PrivacyPreservingAnalytics {
  constructor(epsilon = 0.1) {
    this.epsilon = epsilon; // Privacy budget
    this.queries = new Map();
  }

  addLaplacianNoise(value, sensitivity) {
    const scale = sensitivity / this.epsilon;
    const noise = this.generateLaplacianNoise(scale);
    return value + noise;
  }

  generateLaplacianNoise(scale) {
    const u = Math.random() - 0.5;
    return scale * Math.sign(u) * 
      Math.log(1 - 2 * Math.abs(u));
  }

  async computePrivateAverage(data) {
    const sum = data.reduce((a, b) => a + b, 0);
    const count = data.length;
    
    // Add noise to both sum and count
    const noisySum = this.addLaplacianNoise(sum, 1.0);
    const noisyCount = this.addLaplacianNoise(count, 0.0);
    
    return noisySum / noisyCount;
  }

  async performKAnonymization(dataset, k, sensitiveAttributes) {
    // Implement k-anonymization
    const anonymizedData = this.generalizeAttributes(
      dataset,
      this.identifyQuasiIdentifiers(dataset)
    );

    // Verify k-anonymity
    if (!this.verifyKAnonymity(anonymizedData, k)) {
      throw new Error('K-anonymity requirement not met');
    }

    return anonymizedData;
  }

  private identifyQuasiIdentifiers(dataset) {
    // Implement quasi-identifier detection logic
  }

  private generalizeAttributes(dataset, quasiIdentifiers) {
    // Implement attribute generalization
  }

  private verifyKAnonymity(dataset, k) {
    // Verify that each equivalence class has at least k records
    const equivalenceClasses = this.groupByQuasiIdentifiers(dataset);
    return equivalenceClasses.every(class => class.length >= k);
  }
}`}
          </pre>
        </div>
      </section>
      
    </div>
  );
}

