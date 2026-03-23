import httpx

r = httpx.get('http://192.168.1.107:8000/persons/family-tree', timeout=15)
data = r.json()

# Check specific IDs (Misha=1, Aelita=2, Sasha=3, Tanya=5, Aurora=6, Veronika=25)
kids = [1, 2, 6, 25]
for p in data['persons']:
    if p['person_id'] in kids or p['person_id'] in [3, 5]:
        bd = p.get('birth_date', '') or ''
        print(f"id={p['person_id']:3d} birth={bd:12s} name={p['name']:20s} photo_count={p.get('photo_count',0)}")

print("\n--- Relations involving these IDs ---")
for rel in data['relations']:
    if rel['person_id_from'] in [3, 5, 1, 2, 6, 25] or rel['person_id_to'] in [3, 5, 1, 2, 6, 25]:
        print(f"  {rel['person_id_from']} --({rel['relation_type']})--> {rel['person_id_to']}")
