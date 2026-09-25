import csv, random

random.seed(7)

locations = ["Tago", "Lianga", "Tandag", "San Miguel", "Bislig", "Cantilan", "Marihatag", "Barobo"]
crops = ["Rice", "Corn", "Coconut", "Banana", "Vegetables"]

reports = [
    ("Rice", "Many insects are eating the rice leaves."),
    ("Rice", "There are many small insects around the plants."),
    ("Rice", "The rice leaves have yellow spots and several plants are dying."),
    ("Rice", "Heavy rain flooded part of the rice field."),
    ("Rice", "The field has not received enough water."),
    ("Rice", "Some rice plants have brown patches on the stems."),
    ("Rice", "Rats have been damaging the young rice stalks."),
    ("Corn", "The corn plants are turning yellow and growing slowly."),
    ("Corn", "Several corn plants have damaged leaves from worms."),
    ("Corn", "The soil is very dry and the corn plants are wilting."),
    ("Corn", "Corn leaves are curling and look pale, maybe lacking nutrients."),
    ("Corn", "Strong winds knocked down several corn stalks last week."),
    ("Corn", "White fungus is appearing on some corn cobs."),
    ("Coconut", "Many coconut leaves are becoming brown."),
    ("Coconut", "Coconut trees are producing fewer nuts than usual."),
    ("Coconut", "Beetles have been boring into the coconut trunks."),
    ("Coconut", "Coconut fronds are drooping and turning yellow."),
    ("Banana", "Banana leaves have dark spots spreading quickly."),
    ("Banana", "Banana plants are not growing well due to lack of fertilizer."),
    ("Banana", "Strong rains caused some banana plants to fall over."),
    ("Banana", "Aphids are found on the underside of banana leaves."),
    ("Vegetables", "Cabbage leaves are being eaten by caterpillars."),
    ("Vegetables", "Tomato plants are wilting despite regular watering."),
    ("Vegetables", "Eggplant leaves show signs of powdery mildew."),
    ("Vegetables", "Vegetable seedlings are turning yellow, possibly nutrient deficiency."),
    ("Vegetables", "Heavy storm damaged the vegetable garden beds."),
]

rows = []
report_id = 1
start_month = 8
for i in range(44):
    crop, text = random.choice(reports)
    loc = random.choice(locations)
    month = random.choice([8, 9, 10])
    day = random.randint(1, 28)
    date = f"2026-{month:02d}-{day:02d}"
    rows.append({"date": date, "location": loc, "crop": crop, "report": text})
    report_id += 1

# Inject a few messy rows for cleaning demo
rows.append({"date": "2026-09-10", "location": "Tago", "crop": "Rice", "report": ""})  # empty report
rows.append(dict(rows[3]))  # duplicate
rows.append({"date": "", "location": "Lianga", "crop": "Corn", "report": "The corn field looks unhealthy."})  # missing date
rows.append({"date": "2026-09-15", "location": "  Bislig  ", "crop": "Coconut", "report": "  Coconut leaves are turning brown.  "})  # whitespace

random.shuffle(rows)

with open("data/farmer_reports.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "location", "crop", "report"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows")
