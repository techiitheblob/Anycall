import requests, json
url = "https://xeno-canto.org/api/3/recordings"
for sci_name in ["Elephas maximus", "Microhyla ornata", "Acheta domesticus", "Sus scrofa"]:
    r = requests.get(url, params={"query": sci_name}, timeout=15,
                     headers={"User-Agent": "AnyCall-Research/1.0"})
    print(f"\n{sci_name}: HTTP {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        recs = data.get("recordings", [])
        print(f"  numRecordings={data.get('numRecordings',0)}  numPages={data.get('numPages',0)}")
        for rec in recs[:5]:
            print(f"  ID={rec['id']}  en={rec.get('en','')}  cnt={rec.get('cnt','')}")
    else:
        print(r.text[:200])
