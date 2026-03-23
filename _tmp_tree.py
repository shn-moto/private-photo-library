import httpx, json

# Fetch full family tree
r = httpx.get("http://192.168.1.107:8000/persons/family-tree", timeout=30)
tree = r.json()

persons = {p['person_id']: p for p in tree['persons']}
relations = tree['relations']

# Print all persons with key info
print("=== ALL PERSONS ===")
for p in sorted(tree['persons'], key=lambda x: x['person_id']):
    pid = p['person_id']
    name = p['name']
    bd = p.get('birth_date', '')
    cfid = p.get('cover_face_id')
    pc = p.get('photo_count', 0)
    fc = p.get('face_count', 0)
    print(f"  pid={pid:>3}  {name:40s}  born={bd or '-':12s}  cover={cfid}  photos={pc}")

print(f"\n=== ALL RELATIONS ({len(relations)}) ===")
for rel in relations:
    rid = rel['relation_id']
    pfrom = rel['person_id_from']
    pto = rel['person_id_to']
    rtype = rel['relation_type']
    name_from = persons[pfrom]['name'] if pfrom in persons else f"?{pfrom}"
    name_to = persons[pto]['name'] if pto in persons else f"?{pto}"
    print(f"  rel={rid:>3}  {name_from:30s} --[{rtype:8s}]--> {name_to}")

# Now build parent-child and spouse relationships for key persons
print("\n=== TREE STRUCTURE (parent -> children) ===")
# Find all parent relations
parent_of = {}  # person_id -> list of children ids
spouse_of = {}  # person_id -> list of spouse ids
for rel in relations:
    if rel['relation_type'] == 'parent':
        parent = rel['person_id_from']
        child = rel['person_id_to']
        parent_of.setdefault(parent, []).append(child)
    elif rel['relation_type'] == 'spouse':
        spouse_of.setdefault(rel['person_id_from'], []).append(rel['person_id_to'])
        spouse_of.setdefault(rel['person_id_to'], []).append(rel['person_id_from'])

# Print tree for important persons
for pid in sorted(parent_of.keys()):
    name = persons[pid]['name'] if pid in persons else f"?{pid}"
    children = [persons[c]['name'] if c in persons else f"?{c}" for c in parent_of[pid]]
    spouses = [persons[s]['name'] if s in persons else f"?{s}" for s in spouse_of.get(pid, [])]
    sp_str = f" (spouse: {', '.join(spouses)})" if spouses else ""
    print(f"  {name} (pid={pid}){sp_str}")
    for c in parent_of[pid]:
        cname = persons[c]['name'] if c in persons else f"?{c}"
        print(f"    └── {cname} (pid={c})")

# Key question: who are parents of Sasha (pid=3) and Tanya (pid=5)?
print("\n=== SASHA (pid=3) PARENTS ===")
for rel in relations:
    if rel['person_id_to'] == 3 and rel['relation_type'] == 'parent':
        pfrom = rel['person_id_from']
        print(f"  Parent: {persons[pfrom]['name']} (pid={pfrom})")
    if rel['person_id_from'] == 3:
        print(f"  Sasha --[{rel['relation_type']}]--> {persons[rel['person_id_to']]['name']} (pid={rel['person_id_to']})")

print("\n=== TANYA (pid=5) PARENTS ===")
for rel in relations:
    if rel['person_id_to'] == 5 and rel['relation_type'] == 'parent':
        pfrom = rel['person_id_from']
        print(f"  Parent: {persons[pfrom]['name']} (pid={pfrom})")
    if rel['person_id_from'] == 5:
        print(f"  Tanya --[{rel['relation_type']}]--> {persons[rel['person_id_to']]['name']} (pid={rel['person_id_to']})")

# Sasha-Tanya connection
print("\n=== SASHA-TANYA SPOUSE ===")
for rel in relations:
    if (rel['person_id_from'] == 3 and rel['person_id_to'] == 5) or \
       (rel['person_id_from'] == 5 and rel['person_id_to'] == 3):
        print(f"  {rel}")
