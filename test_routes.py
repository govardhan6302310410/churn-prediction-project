import requests

base_url = "http://127.0.0.1:8000"

print("Uploading data...")
with open("uploads/data.csv", "rb") as f:
    files = {"file": f}
    r = requests.post(f"{base_url}/upload", files=files)
    print(f"Upload status: {r.status_code}")

print("Training model...")
r = requests.post(f"{base_url}/model")
print(f"Model training status: {r.status_code}")

routes = [
    "/",
    "/upload",
    "/model",
    "/history",
    "/dashboard",
    "/segmentation",
    "/heatmap",
    "/model-comparison",
    "/feature-importance",
    "/risk-timeline",
    "/simulation",
    "/ai-explanation?id=1",
    "/retention-strategy?id=1",
    "/business-impact",
    "/customer-persona",
    "/risk-alerts",
    "/dataset-insights",
    "/confidence",
    "/interactive-charts",
    "/activity-monitor"
]

errors = False
for route in routes:
    try:
        res = requests.get(f"{base_url}{route}")
        print(f"GET {route:30} -> {res.status_code}")
        if res.status_code != 200:
            print(f"  --> ERROR details: {res.text[:200]}")
            errors = True
    except Exception as e:
        print(f"Failed to fetch {route}: {e}")
        errors = True

if not errors:
    print("ALL ROUTES LOADED SUCCESSFULLY!")
