import urllib.request, json

# Get all persons with dates
resp = urllib.request.urlopen('http://192.168.1.107:8000/persons?limit=500')
data = json.loads(resp.read())
persons = data.get('persons', data) if isinstance(data, dict) else data

print(f"Total persons: {len(persons)}")
print(f"\n{'ID':>4} {'Name':<35} {'Birth':>12} {'Death':>12} {'Faces':>5} {'Photos':>6}")
print("-" * 90)
for p in persons:
    pid = p['person_id']
    name = p['name'][:34]
    bd = p.get('birth_date', '') or ''
    dd = p.get('death_date', '') or ''
    fc = p.get('face_count', 0)
    pc = p.get('photo_count', 0)
    print(f"{pid:>4} {name:<35} {bd:>12} {dd:>12} {fc:>5} {pc:>6}")

# Get family tree to check relations
resp2 = urllib.request.urlopen('http://192.168.1.107:8000/persons/family-tree')
tree = json.loads(resp2.read())
rels = tree['relations']
print(f"\nRelations ({len(rels)}):")
pmap = {p['person_id']: p['name'] for p in tree['persons']}
for r in rels:
    f = pmap.get(r['person_id_from'], '?')
    t = pmap.get(r['person_id_to'], '?')
    print(f"  {r['relation_id']:>3}: {f} --[{r['relation_type']}]--> {t}")
