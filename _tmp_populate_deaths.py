"""Populate death dates from family_history.txt into DB via API.
Also sets birth_date_approx for year-only birth dates.
Run AFTER deploy with migrate_add_date_approx.sql applied."""
import httpx

API = "http://192.168.1.107:8000"

# Death dates from family_history.txt (all year-only → approx)
DEATH_DATES = {
    146: ("1905-01-01", True),   # Анищенко Игнат (1814-1905)
    116: ("1961-01-01", True),   # Анищенко Самуил Игнатович (1884-1961)
    140: ("1953-01-01", True),   # Францева Меланья Ивановна (1884-1953)
    111: ("1987-01-01", True),   # Анищенко Михаил Самуилович (1912-1987)
    110: ("1994-01-01", True),   # Анищенко Вера Терентьевна (1912-1994)
    124: ("2018-01-01", True),   # Анищенко Иван (1935-2018)
    149: ("1941-01-01", True),   # Анищенко Владимир (1939-1941)
    117: ("1968-01-01", True),   # Шаргаев Терентий Венедиктович (1883-1968)
    139: ("1968-01-01", True),   # Яцковская Елена Стефановна (1884-1968)
    136: ("2008-01-01", True),   # Шуньков Михаил (1929-2008)
    133: ("1993-01-01", True),   # Бабушка Настя / Коршунова Анастасия Егоровна (1923-1993)
    120: ("2010-01-01", True),   # Шуньков Владимир (1951-2010)
}

print(f"Setting {len(DEATH_DATES)} death dates...")
for pid, (death_date, approx) in DEATH_DATES.items():
    resp = httpx.put(f"{API}/persons/{pid}", json={
        "death_date": death_date,
        "death_date_approx": approx
    }, timeout=10)
    if resp.status_code == 200:
        print(f"  OK: person {pid} death_date={death_date} approx={approx}")
    else:
        print(f"  FAIL: person {pid}: {resp.status_code} {resp.text}")

print("\nDone!")
