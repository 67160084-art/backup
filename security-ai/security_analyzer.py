import json
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

LOG_FILE = Path("/var/log/nginx/security_access.log")
OUTPUT_FILE = Path("/opt/security-ai/data/analysis.json")

WINDOW_MINUTES = 30
MIN_AI_PROFILES = 10

HIGH_REQUEST_RATE = 120
HIGH_LOGIN_RATE = 15
HIGH_DENIED_RATE = 10
HIGH_NOT_FOUND_RATE = 20

SUSPICIOUS_PATHS = [
    "../",
    "%2e%2e",
    ".env",
    "wp-config",
    "phpmyadmin",
    "/.git",
    "etc/passwd",
    "xmlrpc.php"
]

records = []
cutoff = datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)

for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
    try:
        r = json.loads(line)
        t = datetime.fromisoformat(r["time"])

        if t < cutoff:
            continue

        r["time_obj"] = t
        r["client_ip"] = str(r.get("client_ip", "unknown"))
        r["method"] = str(r.get("method", ""))
        r["uri"] = str(r.get("uri", ""))
        r["status"] = int(r.get("status", 0))
        r["bytes"] = int(r.get("bytes", 0))
        r["request_time"] = float(r.get("request_time", 0))
        r["user_agent"] = str(r.get("user_agent", ""))

        records.append(r)

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        pass

ip_count = Counter(r["client_ip"] for r in records)
path_count = Counter(r["uri"] for r in records)
status_count = Counter(str(r["status"]) for r in records)

groups = defaultdict(list)

for r in records:
    minute = r["time_obj"].strftime("%Y-%m-%dT%H:%M")
    groups[(r["client_ip"], minute)].append(r)

profiles = []

for (ip, minute), requests in groups.items():
    total = len(requests)
    unique_paths = len(set(r["uri"] for r in requests))
    denied = sum(r["status"] in (401, 403) for r in requests)
    not_found = sum(r["status"] == 404 for r in requests)
    rate_limited = sum(r["status"] == 429 for r in requests)
    server_errors = sum(500 <= r["status"] < 600 for r in requests)
    login_requests = sum(
        r["uri"].lower().startswith("/wp-login.php")
        for r in requests
    )
    suspicious_paths = sum(
        any(x in r["uri"].lower() for x in SUSPICIOUS_PATHS)
        for r in requests
    )

    avg_time = sum(r["request_time"] for r in requests) / total
    avg_bytes = sum(r["bytes"] for r in requests) / total

    profiles.append({
        "ip": ip,
        "minute": minute,
        "total_requests": total,
        "unique_paths": unique_paths,
        "denied": denied,
        "not_found": not_found,
        "rate_limited": rate_limited,
        "server_errors": server_errors,
        "login_requests": login_requests,
        "suspicious_paths": suspicious_paths,
        "avg_request_time": round(avg_time, 4),
        "avg_bytes": round(avg_bytes, 2)
    })

features = np.array([
    [
        p["total_requests"],
        p["unique_paths"],
        p["denied"],
        p["not_found"],
        p["rate_limited"],
        p["server_errors"],
        p["login_requests"],
        p["suspicious_paths"],
        p["avg_request_time"],
        p["avg_bytes"]
    ]
    for p in profiles
], dtype=float)

ai_enabled = len(profiles) >= MIN_AI_PROFILES
predictions = np.ones(len(profiles), dtype=int)
scores = np.zeros(len(profiles))

if ai_enabled:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    predictions = model.fit_predict(scaled)
    scores = model.decision_function(scaled)

events = []

for p, prediction, score in zip(profiles, predictions, scores):
    reasons = []

    if p["rate_limited"] > 0:
        reasons.append("RATE_LIMIT")

    if p["denied"] >= HIGH_DENIED_RATE:
        reasons.append("ACCESS_DENIED_BURST")

    if p["not_found"] >= HIGH_NOT_FOUND_RATE:
        reasons.append("PATH_SCANNING")

    if p["login_requests"] >= HIGH_LOGIN_RATE:
        reasons.append("LOGIN_BRUTE_FORCE")

    if p["total_requests"] >= HIGH_REQUEST_RATE:
        reasons.append("HIGH_REQUEST_RATE")

    if p["suspicious_paths"] > 0:
        reasons.append("SUSPICIOUS_PATH")

    ai_anomaly = bool(prediction == -1)

    if not reasons and not ai_anomaly:
        continue

    severity = "LOW"

    if any(x in reasons for x in [
        "RATE_LIMIT",
        "ACCESS_DENIED_BURST",
        "PATH_SCANNING",
        "LOGIN_BRUTE_FORCE",
        "HIGH_REQUEST_RATE"
    ]):
        severity = "MEDIUM"

    if "SUSPICIOUS_PATH" in reasons:
        severity = "HIGH"

    events.append({
        "minute": p["minute"],
        "ip": p["ip"],
        "total_requests": p["total_requests"],
        "unique_paths": p["unique_paths"],
        "login_requests": p["login_requests"],
        "denied": p["denied"],
        "not_found": p["not_found"],
        "rate_limited": p["rate_limited"],
        "reasons": reasons,
        "ai_anomaly": ai_anomaly,
        "ai_score": round(float(score), 4),
        "severity": severity
    })

severity_count = Counter(e["severity"] for e in events)

result = {
    "engine": "Hybrid Rules + IsolationForest",
    "window_minutes": WINDOW_MINUTES,
    "ai_enabled": ai_enabled,
    "minimum_ai_profiles": MIN_AI_PROFILES,
    "total_requests": len(records),
    "unique_ips": len(ip_count),
    "traffic_profiles": len(profiles),
    "security_events": len(events),
    "ai_anomalies": sum(e["ai_anomaly"] for e in events),
    "severity_count": dict(severity_count),
    "status_count": dict(status_count),
    "top_ips": ip_count.most_common(10),
    "top_paths": path_count.most_common(10),
    "events": sorted(
        events,
        key=lambda x: (
            {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[x["severity"]],
            x["minute"]
        ),
        reverse=True
    )[:100]
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Security analysis completed")
print("Window           :", WINDOW_MINUTES, "minutes")
print("Total requests   :", len(records))
print("Unique IPs       :", len(ip_count))
print("Traffic profiles :", len(profiles))
print("AI enabled       :", ai_enabled)
print("Security events  :", len(events))
print("AI anomalies     :", result["ai_anomalies"])
print("High severity    :", severity_count.get("HIGH", 0))
print("Medium severity  :", severity_count.get("MEDIUM", 0))
print("Low severity     :", severity_count.get("LOW", 0))
print("Output           :", OUTPUT_FILE)
