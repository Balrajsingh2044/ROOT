# 🏥 HealthChain Enhanced - Complete Installation Guide

## 📋 Project Overview

HealthChain Enhanced is a complete Web3-powered healthcare platform with:
- ✅ **Dual Authentication**: Separate login for Patients and Doctors
- ✅ **Blockchain Integration**: Patient-owned health records on Ethereum
- ✅ **Video Consultation**: Schedule and conduct video calls with doctors
- ✅ **Consent Management**: Privacy-first data sharing
- ✅ **IPFS Storage**: Decentralized file storage
- ✅ **Smart Contracts**: Immutable ownership and access control

---

## 📁 Project Structure

```
HealthChain-Enhanced/
├── frontend/
│   ├── index.html              # Main application (with login)
│   ├── css/
│   │   └── styles.css          # Enhanced styles with login screens
│   └── js/
│       ├── app.js              # Main application logic
│       ├── auth.js             # Authentication logic
│       └── video.js            # Video consultation features
│
├── backend/
│   ├── server.js               # Express server
│   ├── config/
│   │   └── database.js         # MongoDB configuration
│   ├── models/
│   │   ├── User.js             # User model (Patient/Doctor)
│   │   ├── Record.js           # Health record model
│   │   └── Appointment.js      # Appointment model
│   ├── routes/
│   │   ├── auth.js             # Authentication routes
│   │   ├── records.js          # Health records routes
│   │   ├── appointments.js     # Appointment scheduling
│   │   ├── video.js            # Video call routes
│   │   └── web3.js             # Blockchain routes
│   ├── middleware/
│   │   ├── auth.js             # JWT authentication
│   │   └── roleCheck.js        # Role-based access control
│   └── controllers/
│       ├── authController.js
│       ├── recordController.js
│       └── appointmentController.js
│
├── smart-contracts/
│   ├── HealthRecord.sol        # Main health record contract
│   ├── AccessControl.sol       # Access management contract
│   └── deploy.js               # Deployment script
│
├── package.json                # Backend dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

---

## 🚀 Quick Start (15 Minutes)

### Prerequisites

1. **Node.js** (v16+) - https://nodejs.org/
2. **MongoDB** (v5+) - https://www.mongodb.com/try/download/community
3. **MetaMask** - https://metamask.io/
4. **Git** (optional) - https://git-scm.com/

---

## 📦 Installation Steps

### Step 1: Install MongoDB

**Windows:**
```bash
# Download MongoDB Community Server from:
# https://www.mongodb.com/try/download/community
# Install and start MongoDB service
```

**Mac:**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Linux:**
```bash
sudo apt-get install mongodb
sudo systemctl start mongodb
sudo systemctl enable mongodb
```

Verify MongoDB is running:
```bash
mongosh
# Should connect successfully
```

### Step 2: Install Backend Dependencies

```bash
cd backend
npm install
```

This installs:
- express, cors, multer (web server)
- mongoose (MongoDB ORM)
- jsonwebtoken, bcryptjs (authentication)
- ethers (blockchain)
- socket.io (video calls)
- dotenv (environment variables)

### Step 3: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` file:
```env
# Server
PORT=5000
NODE_ENV=development

# MongoDB
MONGODB_URI=mongodb://localhost:27017/healthchain

# JWT Secret (generate random string)
JWT_SECRET=your_super_secret_jwt_key_here

# Blockchain (Sepolia Testnet)
RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
NETWORK=sepolia

# Video Call (Optional - for production)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=

# Frontend URL
FRONTEND_URL=http://localhost:3000
```

**Get Infura API Key (Free):**
1. Go to https://infura.io/
2. Sign up and create project
3. Copy Project ID
4. Add to RPC_URL

### Step 4: Start Backend Server

```bash
cd backend
npm start
```

You should see:
```
✅ MongoDB Connected
✅ Server running on http://localhost:5000
✅ Web3 provider initialized
```

### Step 5: Open Frontend

**Option 1: Simple HTTP Server**
```bash
cd frontend
npx http-server -p 3000
```

**Option 2: Live Server (VS Code)**
- Install "Live Server" extension
- Right-click `index.html`
- Select "Open with Live Server"

**Option 3: Direct File**
- Simply open `frontend/index.html` in browser
- ⚠️ Note: Some features may not work without a server

Frontend opens at: `http://localhost:3000`

---

## 👥 User Accounts

### Test Accounts (Pre-seeded)

**Patient Account:**
- Email: `patient@test.com`
- Password: `patient123`
- Wallet: Connect your MetaMask

**Doctor Account:**
- Email: `doctor@test.com`  
- Password: `doctor123`
- Medical ID: `DOC001`
- Wallet: Connect your MetaMask

### Create New Account

1. Click "Register" on login screen
2. Choose Patient or Doctor
3. Fill in details
4. Connect MetaMask (optional)
5. Click "Create Account"

---

## 🔐 Authentication Flow

### Patient Login

**Method 1: Email/Password**
```
1. Enter email
2. Enter password
3. Click "Login"
4. Optionally connect MetaMask
```

**Method 2: MetaMask Only**
```
1. Click "Connect with MetaMask"
2. Approve in MetaMask
3. Sign message to verify
4. Auto-login with wallet
```

### Doctor Login

**Method 1: Medical ID**
```
1. Enter Medical ID (e.g., DOC001)
2. Enter password
3. Click "Login"
4. Access granted
```

**Method 2: MetaMask + Verification**
```
1. Click "Connect with MetaMask"
2. System verifies doctor credentials
3. Additional verification required
4. Access granted
```

---

## 🎯 Features Guide

### For Patients

#### 1. Upload Health Records
```
Dashboard → Upload → Fill Form → Choose File → Upload
```
- Supported formats: PDF, JPG, PNG
- Max size: 10MB
- Auto-encrypted before upload
- Stored on IPFS
- Ownership recorded on blockchain

#### 2. Grant Access to Doctors
```
Access Control → Grant New Access → Enter Doctor Details → Set Expiration
```
- Search doctor by name/ID
- Set access level (View/Full)
- Set expiration date
- Revoke anytime

#### 3. Schedule Video Consultation
```
Appointments → Schedule New → Select Doctor → Choose Date/Time
```
- View doctor availability
- Pick available slot
- Confirm appointment
- Receive confirmation

#### 4. Join Video Call
```
Appointments → Upcoming → Join Call (5 min before)
```
- Auto-opens video interface
- Share screen option
- Chat during call
- Record consultation (with consent)

### For Doctors

#### 1. View Patient Records (with permission)
```
Patients → Select Patient → View Records
```
- Only accessible with patient consent
- All access logged on blockchain
- View medical history
- Download files

#### 2. Upload Consultation Notes
```
After Consultation → Upload Notes → Attach Files
```
- Create visit summary
- Upload test results
- Prescribe medications
- Patient gets notified

#### 3. Manage Availability
```
Schedule → Set Availability → Select Dates/Times
```
- Set working hours
- Block specific dates
- Set consultation duration
- Auto-updated for patients

#### 4. Conduct Video Consultations
```
Appointments → Today's Appointments → Start Call
```
- Video/audio controls
- View patient records during call
- Add notes in real-time
- End and save consultation

---

## 🔗 Blockchain Integration

### Smart Contract Deployment

**Step 1: Install Hardhat**
```bash
cd smart-contracts
npm install
```

**Step 2: Configure Network**
Edit `hardhat.config.js`:
```javascript
networks: {
  sepolia: {
    url: process.env.RPC_URL,
    accounts: [process.env.PRIVATE_KEY]
  }
}
```

**Step 3: Deploy Contracts**
```bash
npx hardhat run scripts/deploy.js --network sepolia
```

**Step 4: Update Backend**
Copy deployed contract addresses to `.env`:
```env
HEALTH_RECORD_CONTRACT=0x...
ACCESS_CONTROL_CONTRACT=0x...
```

### How Blockchain is Used

**1. Record Ownership**
```
Upload Record → File to IPFS → Mint NFT → Store IPFS hash on-chain
```

**2. Access Control**
```
Grant Access → Smart contract updated → Access key encrypted → Logged on-chain
```

**3. Consent Management**
```
Patient approves → Smart contract logs consent → Doctor can access → All tracked
```

---

## 📹 Video Call Setup

### Basic Setup (Peer-to-Peer)

Already configured! Uses WebRTC for direct peer-to-peer calls.

**How it works:**
1. Patient schedules appointment
2. Both join at scheduled time
3. WebRTC connection established
4. Video/audio streams shared
5. Call recorded on blockchain (metadata only)

### Production Setup (Using Twilio)

For better quality and reliability:

**Step 1: Create Twilio Account**
1. Go to https://www.twilio.com/
2. Sign up for free trial
3. Get Account SID and Auth Token

**Step 2: Configure Backend**
Add to `.env`:
```env
TWILIO_ACCOUNT_SID=ACxxxxx
TWILIO_AUTH_TOKEN=xxxxx
```

**Step 3: Restart Server**
```bash
npm restart
```

Now video calls use Twilio's infrastructure!

---

## 🧪 Testing Guide

### Test the Complete Flow

**1. Patient Journey**
```bash
1. Register as patient (patient2@test.com)
2. Connect MetaMask
3. Upload a test PDF (medical report)
4. Grant access to doctor@test.com
5. Schedule appointment for tomorrow
6. Check dashboard for confirmation
```

**2. Doctor Journey**
```bash
1. Login as doctor (doctor@test.com)
2. View patients list
3. See patient2's record (with permission)
4. Check upcoming appointments
5. Accept appointment request
```

**3. Video Consultation**
```bash
1. Both users login at appointment time
2. Patient clicks "Join Call"
3. Doctor clicks "Start Consultation"
4. Video call connects
5. Test video/audio
6. Doctor adds notes
7. End call
8. Check blockchain for consultation record
```

### API Testing

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Register User:**
```bash
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123",
    "role": "patient",
    "name": "Test User"
  }'
```

**Login:**
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123"
  }'
```

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error: MongoDB connection failed**
```bash
# Check if MongoDB is running
mongosh

# If not running:
# Windows: Start MongoDB service
# Mac: brew services start mongodb-community
# Linux: sudo systemctl start mongodb
```

**Error: Port 5000 already in use**
```bash
# Change port in .env
PORT=5001

# Or kill process on port 5000
lsof -i :5000
kill -9 <PID>
```

### Frontend Issues

**MetaMask not connecting**
- Make sure MetaMask is installed
- Switch to Sepolia testnet
- Unlock MetaMask
- Refresh page

**Login failing**
- Check backend is running (http://localhost:5000/api/health)
- Check MongoDB is connected
- Clear browser cache
- Check console for errors (F12)

**Video call not connecting**
- Allow camera/microphone permissions
- Check both users are online
- Try different browser (Chrome recommended)
- Check firewall settings

### Blockchain Issues

**Transactions failing**
- Get test ETH: https://sepoliafaucet.com/
- Check wallet has enough gas
- Verify contract is deployed
- Check network is Sepolia

---

## 📊 Database Schema

### Users Collection
```javascript
{
  _id: ObjectId,
  email: String,
  password: String (hashed),
  role: "patient" | "doctor",
  name: String,
  walletAddress: String,
  // Doctor specific
  medicalId: String,
  specialization: String,
  // Patient specific
  dateOfBirth: Date,
  bloodGroup: String
}
```

### Records Collection
```javascript
{
  _id: ObjectId,
  patientId: ObjectId,
  title: String,
  type: String,
  ipfsHash: String,
  txHash: String,
  uploadedBy: ObjectId,
  createdAt: Date,
  accessGrants: [{
    doctorId: ObjectId,
    grantedAt: Date,
    expiresAt: Date
  }]
}
```

### Appointments Collection
```javascript
{
  _id: ObjectId,
  patientId: ObjectId,
  doctorId: ObjectId,
  scheduledDate: Date,
  duration: Number,
  status: "pending" | "confirmed" | "completed" | "cancelled",
  videoCallLink: String,
  notes: String,
  createdAt: Date
}
```

---

## 🔐 Security Features

### Authentication
- ✅ JWT tokens (expires in 24h)
- ✅ Password hashing (bcrypt)
- ✅ Role-based access control
- ✅ Wallet signature verification

### Data Protection
- ✅ End-to-end encryption
- ✅ HTTPS only (production)
- ✅ CORS configured
- ✅ Input sanitization
- ✅ SQL injection prevention

### Blockchain Security
- ✅ Smart contract access control
- ✅ Immutable audit logs
- ✅ Encrypted IPFS storage
- ✅ Multi-signature support (Phase 2)

---

## 📚 API Documentation

### Authentication Endpoints

**POST /api/auth/register**
```json
Request:
{
  "email": "user@example.com",
  "password": "password123",
  "role": "patient",
  "name": "John Doe"
}

Response:
{
  "success": true,
  "token": "eyJhbGc...",
  "user": {
    "id": "123",
    "email": "user@example.com",
    "role": "patient"
  }
}
```

**POST /api/auth/login**
```json
Request:
{
  "email": "user@example.com",
  "password": "password123"
}

Response:
{
  "success": true,
  "token": "eyJhbGc...",
  "user": { ... }
}
```

### Records Endpoints

**GET /api/records** (requires auth)
```json
Response:
{
  "success": true,
  "records": [...]
}
```

**POST /api/records/upload** (requires auth)
```json
Request: (multipart/form-data)
{
  "title": "Blood Test",
  "type": "lab",
  "file": <binary>
}

Response:
{
  "success": true,
  "record": {
    "id": "123",
    "ipfsHash": "Qm...",
    "txHash": "0x..."
  }
}
```

### Appointment Endpoints

**GET /api/appointments/doctor/:doctorId/availability**
```json
Response:
{
  "success": true,
  "slots": [
    "2024-02-10T09:00:00Z",
    "2024-02-10T10:00:00Z"
  ]
}
```

**POST /api/appointments/schedule**
```json
Request:
{
  "doctorId": "456",
  "date": "2024-02-10T09:00:00Z",
  "duration": 30
}

Response:
{
  "success": true,
  "appointment": { ... }
}
```

---

## 🎓 Technology Stack

**Frontend:**
- HTML5, CSS3, JavaScript (ES6+)
- WebRTC (video calls)
- Web3.js / Ethers.js (blockchain)

**Backend:**
- Node.js + Express
- MongoDB + Mongoose
- JSON Web Tokens (JWT)
- Socket.IO (real-time)
- Multer (file uploads)

**Blockchain:**
- Solidity (smart contracts)
- Ethereum / Polygon
- IPFS (file storage)
- Hardhat (development)

**Video:**
- WebRTC (peer-to-peer)
- Twilio (optional, production)
- Socket.IO (signaling)

---

## 🚀 Deployment (Production)

### Backend Deployment (Heroku)

```bash
# Install Heroku CLI
npm install -g heroku

# Login
heroku login

# Create app
heroku create healthchain-backend

# Add MongoDB
heroku addons:create mongocloud:free

# Set environment variables
heroku config:set JWT_SECRET=your_secret
heroku config:set RPC_URL=your_infura_url

# Deploy
git push heroku main
```

### Frontend Deployment (Vercel)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd frontend
vercel

# Follow prompts
```

### Smart Contracts Deployment

```bash
cd smart-contracts
npx hardhat run scripts/deploy.js --network sepolia

# Save contract addresses
# Update backend .env
```

---

## ✅ Success Checklist

Before saying "it works":

- [ ] MongoDB is running
- [ ] Backend starts without errors
- [ ] Health check returns OK
- [ ] Frontend opens in browser
- [ ] Can register new patient
- [ ] Can register new doctor
- [ ] Can login with both accounts
- [ ] MetaMask connects
- [ ] Can upload a record
- [ ] Record appears in list
- [ ] Can grant access
- [ ] Can schedule appointment
- [ ] Doctor can see patient records
- [ ] Video call connects

---

## 📞 Support & Resources

**Documentation:**
- MongoDB: https://docs.mongodb.com/
- Express: https://expressjs.com/
- Ethers.js: https://docs.ethers.org/
- WebRTC: https://webrtc.org/

**Get Help:**
- Check backend logs
- Check browser console (F12)
- Test API with Postman
- Verify MongoDB connection

---

**🎉 You're ready to run HealthChain Enhanced!**

Next: Open INSTALLATION_DETAILED.md for step-by-step screenshots and troubleshooting.
