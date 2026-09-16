import requests

url = "https://xeno-canto.org/api/2/recordings?query=Cinnyris+asiaticus"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
print("API v2 status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    print("Recordings count:", data.get("numRecordings"))
    for rec in data.get("recordings", [])[:5]:
        print(f"  ID: {rec.get('id')} | en: {rec.get('en')} | gen: {rec.get('gen')} | sp: {rec.get('sp')}")
else:
    print(r.text[:300])
