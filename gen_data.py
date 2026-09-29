"""
gen_data.py
-----------
Builds data/farmer_reports.csv (synthetic data, NOT real farmer reports):

  1. Reads the original 50-row sample from data/farmer_reports_base.csv
     (it includes a few deliberately messy rows for the cleaning demo).
  2. Adds 500 more synthetic reports generated with random seed 11.

Run from the project folder:  python gen_data.py
"""
import csv, random

locations = ["Tago", "Lianga", "Tandag", "San Miguel", "Bislig", "Cantilan", "Marihatag", "Barobo"]
crops = ["Rice", "Corn", "Coconut", "Banana", "Vegetables"]

with open("data/farmer_reports_base.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# ---------------------------------------------------------------------------
# 500 additional synthetic reports (random seed 11)
# Each entry is (weight, text): higher weight = reported more often.
# ---------------------------------------------------------------------------
rng = random.Random(11)

more = {
    "Rice": [
        (5, "Many insects are eating the rice leaves."),
        (3, "Insects are chewing the leaves of my rice plants."),
        (2, "Insects are folding the rice leaves and leaving white streaks."),
        (4, "There are many small insects around the plants."),
        (3, "Small insects are jumping around the base of the rice plants."),
        (5, "The rice leaves have yellow spots and several plants are dying."),
        (3, "Yellow spots are showing on the rice leaves and the tips are drying."),
        (3, "Some rice plants have brown patches on the stems."),
        (2, "Brown patches are spreading on the rice stems near the water line."),
        (3, "Heavy rain flooded part of the rice field."),
        (2, "The rice field was flooded after the river overflowed."),
        (3, "The field has not received enough water."),
        (2, "There is not enough water in the irrigation canal for the rice."),
        (2, "Rats have been damaging the young rice stalks."),
    ],
    "Corn": [
        (5, "The corn plants are turning yellow and growing slowly."),
        (3, "Corn leaves are pale and the plants look weak, maybe lacking nutrients."),
        (2, "Lower corn leaves are yellow, maybe the soil needs fertilizer."),
        (5, "Several corn plants have damaged leaves from worms."),
        (4, "Worms are eating the corn leaves and there are droppings in the whorl."),
        (3, "The soil is very dry and the corn plants are wilting."),
        (2, "Corn leaves are rolling because of dry weather."),
        (4, "Strong winds knocked down several corn stalks last week."),
        (2, "Wind flattened part of the corn field."),
        (3, "White fungus is appearing on some corn cobs."),
        (2, "Moldy cobs found after the rainy days and the kernels are rotting."),
        (1, "Rats are eating the corn cobs."),
    ],
    "Coconut": [
        (4, "Many coconut leaves are becoming brown."),
        (2, "The leaves of several coconut trees are turning brown at the tips."),
        (4, "Coconut trees are producing fewer nuts than usual."),
        (2, "The palms produce fewer nuts and many nuts fall early."),
        (4, "Beetles have been boring into the coconut trunks."),
        (2, "There are holes in the coconut crown and beetles are boring in."),
        (4, "Coconut fronds are drooping and turning yellow."),
        (2, "Several coconut fronds are yellow and hanging down."),
        (1, "Strong winds broke several coconut fronds."),
    ],
    "Banana": [
        (5, "Banana leaves have dark spots spreading quickly."),
        (3, "Dark spots and streaks are showing on the older banana leaves."),
        (3, "Banana plants are not growing well due to lack of fertilizer."),
        (2, "Banana leaves look pale and the bunches are small; the soil needs fertilizer."),
        (4, "Strong rains caused some banana plants to fall over."),
        (2, "Heavy rain toppled several banana plants."),
        (4, "Aphids are found on the underside of banana leaves."),
        (2, "Small black aphids with ants are on the banana plants."),
        (2, "The banana plants are wilting because there is not enough water."),
    ],
    "Vegetables": [
        (5, "Cabbage leaves are being eaten by caterpillars."),
        (3, "Caterpillars are eating holes in the mustard leaves."),
        (4, "Tomato plants are wilting despite regular watering."),
        (2, "Tomato plants are wilting despite regular watering, and the stems ooze when cut."),
        (4, "Eggplant leaves show signs of powdery mildew."),
        (2, "White powder is covering the squash leaves, it looks like powdery mildew."),
        (3, "Vegetable seedlings are turning yellow, possibly nutrient deficiency."),
        (3, "Heavy storm damaged the vegetable garden beds."),
        (2, "Strong rain washed out the vegetable seedbeds."),
        (2, "The vegetable plants are drying and the soil is very dry."),
    ],
}
crop_weights = {"Rice": 24, "Corn": 24, "Coconut": 16, "Banana": 18, "Vegetables": 18}
loc_weights = [16, 14, 18, 10, 12, 10, 8, 12]  # Tago, Lianga, Tandag, San Miguel, Bislig, Cantilan, Marihatag, Barobo

for _ in range(500):
    crop = rng.choices(crops, weights=[crop_weights[c] for c in crops])[0]
    weights, texts = zip(*more[crop])
    text = rng.choices(texts, weights=weights)[0]
    loc = rng.choices(locations, weights=loc_weights)[0]
    month = rng.choices([8, 9, 10], weights=[3, 4, 3])[0]
    rows.append({"date": f"2026-{month:02d}-{rng.randint(1, 28):02d}", "location": loc, "crop": crop, "report": text})

rng.shuffle(rows)

with open("data/farmer_reports.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "location", "crop", "report"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows")
