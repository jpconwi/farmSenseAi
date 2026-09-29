"""
gen_more_data.py
----------------
Adds MORE synthetic reports (NOT real farmer data) to the dataset, with realistic
"hotspots": some towns and crops have many more reports, and each hotspot has a
mix of different problem types.

  Input : data/farmer_reports_v1.csv   (the dataset before this step)
  Output: data/farmer_reports.csv      (v1 + about 650 new rows, random seed 21)

Run from the project folder:  python gen_more_data.py
It is reproducible: it always starts from farmer_reports_v1.csv, so running it twice gives the same file.
"""
import csv, random

LOCATIONS = ["Tago", "Lianga", "Tandag", "San Miguel", "Bislig", "Cantilan", "Marihatag", "Barobo"]
CROPS = ["Rice", "Corn", "Coconut", "Banana", "Vegetables"]

# ---- report text pools: (weight, text). Every text matches a rule in utils/knowledge.py.
T = {
 "Rice": {
  "leaf":   [(3, "Brown spots with yellow rings are appearing on the rice leaves and many plants are dying."),
             (3, "Yellow spots are spreading on the rice leaves and the plants are dying.")],
  "sheath": [(3, "Brown patches with grey centers are spreading up the rice stems after continuous rain."),
             (2, "Brown patches on the rice stems are killing some tillers.")],
  "flood":  [(3, "The rice field is flooded after three days of heavy rain."),
             (2, "Rice seedlings were buried in mud after the flood.")],
  "water":  [(3, "Lack of water in the canal is drying the rice paddies."),
             (2, "The rice field has not received enough water this week.")],
  "rats":   [(2, "Rats cut many young rice tillers overnight.")],
  "insect": [(3, "Insects are eating the rice leaves and leaving white streaks.")],
  "hopper": [(3, "Many small insects are gathering at the base of the rice plants and the plants are drying in circles.")],
  "nutr":   [(3, "Rice plants are pale and growing slowly, the field needs fertilizer.")],
  "storm":  [(2, "Heavy rain caused the rice plants to fall over."),
             (2, "Strong wind knocked down rice plants before harvest.")],
 },
 "Corn": {
  "worm":   [(4, "Worms are chewing holes in the young corn leaves."),
             (3, "Fall armyworm larvae are hiding in the corn whorl and eating the leaves.")],
  "rot":    [(3, "Pink mold is growing on the corn cobs after rain."),
             (2, "White fungus is spreading on the corn ears.")],
  "nutr":   [(3, "Corn plants are pale and stunted, the soil needs fertilizer."),
             (2, "Corn leaves are turning yellow and the plants are growing slowly.")],
  "dry":    [(3, "The corn field is very dry and the leaves are rolling."),
             (2, "Corn plants are wilting because the soil is very dry.")],
  "wind":   [(3, "Strong wind flattened many corn stalks."),
             (2, "Wind knocked down corn plants near the road.")],
  "storm":  [(2, "Heavy rain damaged the corn plants near the river.")],
  "rats":   [(1, "Rats are gnawing the corn stalks.")],
 },
 "Coconut": {
  "beetle": [(4, "Rhinoceros beetles are boring into the coconut crown."),
             (3, "Beetles have left holes and sawdust at the base of the coconut fronds.")],
  "yellow": [(4, "Coconut fronds are turning yellow and drooping in several trees."),
             (3, "Old coconut palms have yellow drooping fronds and smaller nuts.")],
  "brown":  [(3, "The coconut leaves are turning brown and drying from the tips."),
             (2, "Many coconut leaves are becoming brown after the dry weeks.")],
  "yield":  [(3, "Coconut trees are giving fewer nuts this season."),
             (2, "The palms produce fewer nuts and many nuts fall early.")],
  "wind":   [(2, "Strong wind broke several coconut fronds.")],
 },
 "Banana": {
  "sig":    [(5, "Banana leaves have dark spots spreading quickly."),
             (3, "Banana leaves have dark spots spreading across the surface."),
             (3, "Older banana leaves have dark streaks and spots.")],
  "aphid":  [(4, "Aphids and ants are found at the base of the banana leaves."),
             (3, "Aphids are found on the underside of banana leaves.")],
  "drop":   [(4, "Banana leaves are continuously falling and hanging around the stem."),
             (3, "Older banana leaves are drooping and turning yellow."),
             (2, "The banana leaves are falling one after another and the stem is splitting.")],
  "storm":  [(3, "Banana plants toppled after strong rains."),
             (2, "Heavy rain damaged several banana plants.")],
  "nutr":   [(3, "Banana bunches are small and the leaves look pale, the soil needs fertilizer.")],
  "weak":   [(2, "Several banana plants have weak stems and poor growth.")],
  "insect": [(2, "Insects are feeding on young banana leaves.")],
  "water":  [(2, "The banana plants are wilting because there is not enough water.")],
 },
 "Vegetables": {
  "cater":  [(4, "Caterpillars are eating the cabbage leaves."),
             (3, "Worms are boring holes in the pechay leaves.")],
  "wilt":   [(4, "Tomato plants are wilting even after watering and the stems ooze when cut."),
             (2, "Eggplant plants keep wilting despite regular watering.")],
  "mildew": [(3, "Powdery mildew is covering the cucumber leaves."),
             (2, "White powder is spreading on the squash leaves, it looks like mildew.")],
  "nutr":   [(3, "Ampalaya seedlings are turning yellow and growing slowly.")],
  "storm":  [(3, "Heavy rain washed away the vegetable seedbeds.")],
  "dry":    [(2, "The vegetable beds are very dry and the plants are wilting.")],
  "wind":   [(1, "Strong wind knocked down the tomato stakes and plants.")],
 },
}

def pool(crop, kinds=None):
    out = []
    for k, items in T[crop].items():
        if kinds is None or k in kinds:
            out += items
    return out

# ---- hotspots: (location, crop, {problem kinds: extra weight}, rows to add)
HOTSPOTS = [
    ("Tandag",     "Banana",     {"sig": 5, "aphid": 3, "drop": 4, "storm": 2, "nutr": 1}, 80),
    ("Tandag",     "Rice",       {"sheath": 4, "leaf": 3, "flood": 3, "hopper": 2},          55),
    ("Tago",       "Coconut",    {"beetle": 4, "yellow": 4, "brown": 3, "yield": 2},         60),
    ("Bislig",     "Corn",       {"worm": 5, "wind": 3, "rot": 3, "dry": 1},                 55),
    ("Lianga",     "Rice",       {"flood": 5, "water": 3, "hopper": 3, "insect": 2},         50),
    ("Barobo",     "Vegetables", {"wilt": 5, "mildew": 4, "cater": 3},                       50),
    ("Cantilan",   "Banana",     {"nutr": 3, "aphid": 3, "weak": 2, "sig": 2},               35),
    ("San Miguel", "Corn",       {"nutr": 4, "dry": 3, "worm": 2},                           30),
    ("Marihatag",  "Coconut",    {"yield": 4, "brown": 3, "yellow": 2},                      30),
]
CROP_W = {"Rice": 26, "Corn": 24, "Coconut": 16, "Banana": 20, "Vegetables": 14}
LOC_W  = [15, 12, 20, 9, 13, 11, 8, 12]   # Tago, Lianga, Tandag, San Miguel, Bislig, Cantilan, Marihatag, Barobo
N_GENERAL = 200

rng = random.Random(21)
with open("data/farmer_reports_v1.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
seen = {(r["date"], r["location"], r["crop"], r["report"]) for r in rows}

def add(loc, crop, text, sept_bias=False):
    for _ in range(20):                                   # avoid exact duplicate rows (the app drops them)
        if sept_bias:
            month, day = rng.choices([(8, rng.randint(1, 31)), (9, rng.randint(1, 29))], weights=[3, 7])[0]
        else:
            month, day = rng.choice([(8, rng.randint(1, 31)), (9, rng.randint(1, 29))])
        key = (f"2026-{month:02d}-{day:02d}", loc, crop, text)
        if key not in seen:
            seen.add(key)
            rows.append(dict(zip(("date", "location", "crop", "report"), key)))
            return

for loc, crop, kinds, n in HOTSPOTS:
    items = [(w * kinds[k], t) for k, its in T[crop].items() if k in kinds for w, t in its]
    ws, ts = zip(*items)
    for _ in range(n):
        add(loc, crop, rng.choices(ts, weights=ws)[0], sept_bias=True)

for _ in range(N_GENERAL):
    crop = rng.choices(CROPS, weights=[CROP_W[c] for c in CROPS])[0]
    ws, ts = zip(*pool(crop))
    add(rng.choices(LOCATIONS, weights=LOC_W)[0], crop, rng.choices(ts, weights=ws)[0])

rng.shuffle(rows)
with open("data/farmer_reports.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["date", "location", "crop", "report"])
    w.writeheader(); w.writerows(rows)
print(f"Wrote {len(rows)} rows")
