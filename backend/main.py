"""
UPI Fraud Detection Backend
Run: pip install fastapi uvicorn scikit-learn pandas numpy faker websockets
Then: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import random
import threading
import time
from datetime import datetime
from typing import List
import numpy as np

app = FastAPI(title="UPI Fraud Detection API")

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# FAKE ML MODEL (No dataset needed!)
# Works like a real model but uses smart rules
# Replace with real sklearn model if you want
# ─────────────────────────────────────────

class FraudDetectionEngine:
    def __init__(self):
        self.user_profiles = {}  # stores normal behavior per user

    def update_profile(self, user_id, amount, hour):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "amounts": [],
                "hours": [],
                "txn_count": 0
            }
        profile = self.user_profiles[user_id]
        profile["amounts"].append(amount)
        profile["hours"].append(hour)
        profile["txn_count"] += 1
        # Keep only last 50 transactions
        if len(profile["amounts"]) > 50:
            profile["amounts"].pop(0)
            profile["hours"].pop(0)

    def get_avg_amount(self, user_id):
        if user_id not in self.user_profiles:
            return 1000
        amounts = self.user_profiles[user_id]["amounts"]
        return sum(amounts) / len(amounts) if amounts else 1000

    def analyze(self, transaction):
        flags = []
        score = 0

        amount = transaction["amount"]
        hour = transaction["hour"]
        user_id = transaction["user_id"]
        merchant = transaction["merchant"]
        city = transaction["city"]
        usual_city = transaction["usual_city"]
        txns_last_hour = transaction["txns_last_hour"]
        is_new_merchant = transaction["is_new_merchant"]

        avg_amount = self.get_avg_amount(user_id)

        # ── Rule 1: Amount anomaly
        if avg_amount > 0:
            ratio = amount / avg_amount
            if ratio > 10:
                flags.append(f"Amount is {int(ratio)}x above average")
                score += 35
            elif ratio > 5:
                flags.append(f"Amount is {int(ratio)}x above average")
                score += 20
            elif ratio > 3:
                score += 10

        # ── Rule 2: Unusual hour
        if hour < 5 or hour > 23:
            flags.append(f"Transaction at {hour}:00 AM — unusual hour")
            score += 20
        elif hour < 7:
            score += 10

        # ── Rule 3: Frequency burst
        if txns_last_hour > 10:
            flags.append(f"{txns_last_hour} transactions in last hour — burst detected")
            score += 25
        elif txns_last_hour > 6:
            flags.append(f"{txns_last_hour} transactions in last hour")
            score += 12

        # ── Rule 4: Location change
        if city != usual_city:
            flags.append(f"Transaction from {city} — usually in {usual_city}")
            score += 20

        # ── Rule 5: New/suspicious merchant
        if is_new_merchant:
            flags.append(f"First time transaction with '{merchant}'")
            score += 15

        # ── Rule 6: Very high absolute amount
        if amount > 50000:
            score += 15
        if amount > 80000:
            flags.append(f"Very high amount: ₹{amount:,}")

        # Update user profile with this transaction
        self.update_profile(user_id, amount, hour)

        score = min(score, 100)

        if score >= 70:
            status = "FRAUD"
            color = "red"
        elif score >= 40:
            status = "SUSPICIOUS"
            color = "yellow"
        else:
            status = "SAFE"
            color = "green"

        return {
            "score": score,
            "status": status,
            "color": color,
            "flags": flags
        }

engine = FraudDetectionEngine()

# ─────────────────────────────────────────
# TRANSACTION SIMULATOR
# ─────────────────────────────────────────

USERS = [
    {"id": "u1", "name": "Priya Sharma",   "upi": "priya@okicici",  "city": "Mumbai",    "avg": 800},
    {"id": "u2", "name": "Rahul Verma",    "upi": "rahul@oksbi",    "city": "Delhi",     "avg": 1200},
    {"id": "u3", "name": "Sneha Reddy",    "upi": "sneha@okhdfc",   "city": "Hyderabad", "avg": 600},
    {"id": "u4", "name": "Arjun Nair",     "upi": "arjun@okaxis",   "city": "Bangalore", "avg": 2000},
    {"id": "u5", "name": "Meera Joshi",    "upi": "meera@okpaytm",  "city": "Pune",      "avg": 950},
    {"id": "u6", "name": "Vikram Singh",   "upi": "vikram@ybl",     "city": "Chennai",   "avg": 1500},
]

NORMAL_MERCHANTS = [
    "Zomato", "Swiggy", "Amazon India", "Flipkart",
    "BigBazaar", "DMart", "IRCTC", "MakeMyTrip",
    "Ola Cabs", "Rapido", "BookMyShow", "Reliance Fresh",
    "PhonePe Merchant", "Paytm Mall", "Blinkit", "Zepto"
]

SUSPICIOUS_MERCHANTS = [
    "Unknown Vendor 7823", "CRYPTO_EX_99", "FastCash247",
    "QuickTransfer_XYZ", "Merchant_9981", "IntlTransfer",
    "CoinBase_India", "Unknown_UPI_8821"
]

CITIES = ["Mumbai", "Delhi", "Hyderabad", "Bangalore", "Chennai", "Pune", "Kolkata", "Jaipur"]

current_scenario = {"mode": "normal"}  # normal / attack / story
txn_counter = {"count": 1000}
stats = {
    "total": 0,
    "fraud_caught": 0,
    "amount_saved": 0,
    "suspicious": 0
}

def make_transaction(force_fraud=False):
    user = random.choice(USERS)
    is_fraud = force_fraud or (current_scenario["mode"] == "attack" and random.random() < 0.65)

    txn_counter["count"] += 1
    now = datetime.now()

    if is_fraud:
        amount = random.randint(25000, 95000)
        hour = random.randint(1, 5)
        city = random.choice([c for c in CITIES if c != user["city"]])
        merchant = random.choice(SUSPICIOUS_MERCHANTS)
        is_new_merchant = True
        txns_last_hour = random.randint(8, 18)
    else:
        amount = int(random.gauss(user["avg"], user["avg"] * 0.3))
        amount = max(50, min(amount, user["avg"] * 3))
        hour = random.randint(8, 22)
        city = user["city"]
        merchant = random.choice(NORMAL_MERCHANTS)
        is_new_merchant = random.random() < 0.1
        txns_last_hour = random.randint(1, 4)

    transaction = {
        "txn_id": f"UPI{txn_counter['count']}",
        "user_id": user["id"],
        "user": user["name"],
        "upi": user["upi"],
        "amount": amount,
        "merchant": merchant,
        "hour": hour,
        "city": city,
        "usual_city": user["city"],
        "txns_last_hour": txns_last_hour,
        "is_new_merchant": is_new_merchant,
        "timestamp": now.strftime("%H:%M:%S"),
        "date": now.strftime("%d %b %Y"),
    }

    result = engine.analyze(transaction)
    transaction.update(result)

    # Update stats
    stats["total"] += 1
    if result["status"] == "FRAUD":
        stats["fraud_caught"] += 1
        stats["amount_saved"] += amount
    elif result["status"] == "SUSPICIOUS":
        stats["suspicious"] += 1

    return transaction

# ─────────────────────────────────────────
# WEBSOCKET MANAGER
# ─────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# ─────────────────────────────────────────
# BACKGROUND TRANSACTION GENERATOR
# ─────────────────────────────────────────

async def transaction_broadcaster():
    while True:
        if manager.active_connections:
            txn = make_transaction()
            await manager.broadcast({
                "type": "transaction",
                "data": txn,
                "stats": {
                    "total": stats["total"],
                    "fraud_caught": stats["fraud_caught"],
                    "amount_saved": stats["amount_saved"],
                    "suspicious": stats["suspicious"]
                }
            })
        delay = 1.5 if current_scenario["mode"] == "attack" else 2.0
        await asyncio.sleep(delay)

@app.on_event("startup")
async def startup():
    asyncio.create_task(transaction_broadcaster())

# ─────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "scenario":
                current_scenario["mode"] = msg["mode"]
                await manager.broadcast({
                    "type": "scenario_change",
                    "mode": msg["mode"]
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/scenario")
async def set_scenario(body: dict):
    current_scenario["mode"] = body.get("mode", "normal")
    await manager.broadcast({
        "type": "scenario_change",
        "mode": current_scenario["mode"]
    })
    return {"status": "ok", "mode": current_scenario["mode"]}

@app.post("/api/trigger-fraud")
async def trigger_fraud(body: dict = {}):
    txn = make_transaction(force_fraud=True)
    await manager.broadcast({
        "type": "transaction",
        "data": txn,
        "stats": {
            "total": stats["total"],
            "fraud_caught": stats["fraud_caught"],
            "amount_saved": stats["amount_saved"],
            "suspicious": stats["suspicious"]
        }
    })
    return txn

@app.get("/api/stats")
def get_stats():
    return stats

@app.get("/")
def root():
    return {"status": "UPI Fraud Detection API Running ✅"}
