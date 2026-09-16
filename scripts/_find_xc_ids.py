"""Find Xeno-Canto recording IDs by scraping the public search page (no API key needed)."""
import requests, re, time, json

SPECIES = [
    ("hieroglyphus_banian",    "Hieroglyphus banian"),
    ("homorocoryphus_nitidulus","Homorocoryphus nitidulus"),
    ("conocephalus_maculatus", "Conocephalus maculatus"),
    ("acheta_domesticus",      "Acheta domesticus"),
    ("valanga_irregularis",    "Valanga irregularis"),
    ("schistocerca_gregaria",  "Schistocerca gregaria"),
    ("microhyla_ornata",       "Microhyla ornata"),
    ("fejervarya_limnocharis", "Fejervarya limnocharis"),
    ("kaloula_taprobanica",    "Kaloula taprobanica"),
    ("sphaerotheca_breviceps", "Sphaerotheca breviceps"),
    ("uperodon_systoma",       "Uperodon systoma"),
    ("nyctibatrachus_major",   "Nyctibatrachus major"),
    ("elephas_maximus",        "Elephas maximus"),
    ("panthera_tigris",        "Panthera tigris"),
    ("axis_axis",              "Axis axis"),
    ("vulpes_bengalensis",     "Vulpes bengalensis"),
    ("sus_scrofa",             "Sus scrofa"),
    ("melursus_ursinus",       "Melursus ursinus"),
]

HEADERS = {"User-Agent": "AnyCall-Research/1.0"}
results = {}

for sp_id, sci_name in SPECIES:
    query = sci_name.replace(" ", "+")
    url = f"https://xeno-canto.org/explore?query={query}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        # Extract recording IDs from href="/NNN" patterns in the HTML
        ids = re.findall(r'href="/(\d{4,7})"', r.text)
        # Deduplicate, keep first 10
        seen = []
        for i in ids:
            if i not in seen:
                seen.append(i)
            if len(seen) >= 10:
                break
        results[sp_id] = seen
        print(f"{sp_id}: {seen}")
    except Exception as e:
        results[sp_id] = []
        print(f"{sp_id}: ERROR {e}")
    time.sleep(1.0)

with open("data/xc_extra_ids.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to data/xc_extra_ids.json")
