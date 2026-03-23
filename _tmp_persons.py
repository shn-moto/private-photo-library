"""Query all persons from the API to match with family_history.txt data"""
import httpx, json

API = "http://192.168.1.107:8000"

resp = httpx.get(f"{API}/persons/family-tree", timeout=30)
data = resp.json()

persons = data["persons"]
persons.sort(key=lambda p: p["person_id"])

print(f"Total persons: {len(persons)}")
print(f"{'ID':>4} {'Name':<45} {'Birth':>12} {'Death':>12} {'Faces':>5} {'Photos':>6}")
print("-" * 90)
for p in persons:
    print(f"{p['person_id']:>4} {p['name']:<45} {p['birth_date'] or '':>12} {p['death_date'] or '':>12} {p['face_count']:>5} {p['photo_count']:>6}")
