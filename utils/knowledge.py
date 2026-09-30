"""
utils/knowledge.py
------------------
A small agronomy knowledge base. It turns a farmer's free-text report into a
SPECIFIC likely disease / pest / stress (e.g. "Black Sigatoka") and holds the
cause, symptoms, treatment and prevention for it.

It is rule-based on purpose: the same report always gets the same answer,
with no AI guessing. It is general agronomy guidance (NOT from the dataset)
and is based on the farmer's words only, so it says "likely" / "possible".
"""
import re

TYPE_COLORS = {
    "Pest": "#EF6C00", "Disease": "#C62828", "Water": "#1E88E5",
    "Weather": "#5E35B1", "Nutrient": "#43A047", "Other": "#757575",
}

DISCLAIMER = ("ℹ️ *General agronomy guidance, not taken from the dataset. The diagnosis is based only on the "
              "farmer's written description. Confirm with your municipal agriculturist / DA technician before spraying.*")


def _e(key, name, typ, pattern, crops, agent, symptoms, causes, treatment, prevention):
    return dict(key=key, name=name, type=typ, pattern=re.compile(pattern, re.I), crops=crops, agent=agent,
                symptoms=symptoms, causes=causes, treatment=treatment, prevention=prevention)


# ORDER MATTERS: the first matching entry wins, so specific rules come before general ones.
KB = [
    _e("rice_leaf", "Rice brown spot / bacterial leaf blight (possible)", "Disease",
       r"yellow spots|spots.*dying|dying.*spots", ["Rice"],
       "Fungus *Bipolaris oryzae* (brown spot) or bacterium *Xanthomonas oryzae* (bacterial leaf blight). Tungro virus is also possible.",
       "Brown spot: oval brown spots with a yellow halo. Bacterial leaf blight: yellow-white streaks from the leaf edges, then drying and wilting. Tungro: yellow-orange leaves and stunting.",
       "Poor or nutrient-deficient soil (low nitrogen, potassium, silicon) and water stress favor brown spot. Blight spreads through rain splash, flood water and wounds from wind. Tungro is carried by green leafhoppers.",
       "Remove and destroy badly infected plants. Drain the field if blight is suspected. Avoid extra nitrogen. For brown spot use a registered fungicide (e.g., mancozeb or propiconazole, follow the label). Blight and tungro have no effective spray.",
       "Plant resistant varieties, use certified seed, fertilize in a balanced way, keep the field weed-free, control leafhoppers."),
    _e("rice_sheath", "Sheath blight / stem rot", "Disease",
       r"brown patches", ["Rice"],
       "Fungus *Rhizoctonia solani* (sheath blight).",
       "Grey-green to brown oval patches on the stem sheath near the water line that spread upward and can kill tillers.",
       "Dense planting, too much nitrogen, and hot, humid weather. The fungus survives in crop residue and bunds.",
       "Drain the field for a few days, remove weeds and infected debris, and use a registered fungicide (e.g., validamycin) as labeled.",
       "Wider spacing, balanced fertilizer, clean the bunds, avoid leaving infected stubble."),
    _e("rice_flood", "Flood / submergence damage", "Weather",
       r"flood", None,
       "Excess water from heavy rain (not a pathogen).",
       "Parts of the field sit underwater; plants are starved of oxygen and can be buried in silt.",
       "Heavy or prolonged rainfall combined with poor drainage or a low-lying field.",
       "Drain the water quickly, rinse silt off leaves, and re-apply nitrogen once water recedes. Replant gaps if needed.",
       "Improve drainage canals, plant flood-tolerant varieties, adjust planting time to avoid peak rains."),
    _e("water_shortage", "Water shortage (drought stress)", "Water",
       r"not.*enough water|lack of water|not received", None,
       "Insufficient water (not a pathogen).",
       "Leaves roll, wilt or turn pale and growth slows.",
       "Dry spell, delayed rain, or broken irrigation.",
       "Irrigate as soon as possible, mulch to hold moisture, and irrigate early morning or evening.",
       "Build small water reserves, use drought-tolerant varieties, plan planting around the rainy season."),
    _e("rodent", "Rodent (rat) damage", "Pest",
       r"\brats?\b|rodent", None,
       "Rats.",
       "Cut or gnawed young stalks and tillers.",
       "Rats live in nearby weeds, bunds and grass, especially near planting time.",
       "Community rat control (traps, baiting per DA guidance) and clean the field edges.",
       "Keep bunds short and weed-free, synchronize planting with neighbors, protect natural predators such as owls."),
    _e("fall_armyworm", "Fall armyworm", "Pest",
       r"worms?", ["Corn"],
       "Larvae of the moth *Spodoptera frugiperda*.",
       "Ragged, chewed leaves, windowed leaf tissue and sawdust-like droppings in the whorl.",
       "Adult moths lay eggs on leaves; warm weather and continuous corn planting increase outbreaks.",
       "Scout early. Hand-pick or crush egg masses. Use a registered insecticide (e.g., Bt or spinosad) in the whorl, in the early morning or evening.",
       "Plant on time with neighbors, rotate crops, encourage natural enemies, use Bt or tolerant hybrids where available."),
    _e("corn_ear_rot", "Corn ear rot (fungal)", "Disease",
       r"fungus|cobs?.*mold|mold.*cobs?", ["Corn"],
       "Fungi such as *Diplodia*, *Fusarium* or *Aspergillus*.",
       "White, pink or grey mold on the cob and kernels; kernels rot or shrivel.",
       "Rain that soaks the ear, insect or bird damage on the husk, and late harvesting.",
       "Harvest early, discard moldy cobs (they can carry harmful toxins, so do not feed them to animals), and dry grain fully before storage.",
       "Rotate crops, choose hybrids with tight husks, control ear-boring insects, do not delay harvest."),
    _e("wind", "Wind damage (lodging)", "Weather",
       r"wind|knocked down", None,
       "Strong wind (not a pathogen).",
       "Stalks bend or fall over.",
       "Strong wind before harvest, especially in shallow-rooted or over-fertilized fields.",
       "Prop or re-hill lightly-leaning plants and harvest badly lodged plants early.",
       "Hill up soil around the base, avoid excess nitrogen, plant shorter or sturdier varieties, use windbreaks."),
    _e("black_sigatoka", "Black Sigatoka (banana leaf spot)", "Disease",
       r"dark (spots|streaks)", ["Banana"],
       "Fungus *Pseudocercospora fijiensis*.",
       "Small dark streaks that grow into dark lesions with a yellow halo; leaves dry out early and yield drops.",
       "Warm, humid, rainy weather; spores spread by wind and rain; crowded plants and poor drainage.",
       "Cut off and remove infected leaves, then use a rotation of registered fungicides (e.g., mancozeb, propiconazole) as labeled.",
       "Wider spacing, good drainage, weed control, tolerant varieties."),
    _e("banana_aphid", "Banana aphid (carries bunchy top virus)", "Pest",
       r"aphids?", ["Banana"],
       "Insect *Pentalonia nigronervosa*.",
       "Small dark insects under the leaf base or leaves; ants nearby; if plants show stunted, bunched leaves it may be bunchy top.",
       "Aphids spread fast in warm weather and pass banana bunchy top virus from plant to plant.",
       "Remove and destroy plants with bunchy top, wash aphids off, and control them per DA guidance. There is no cure for the virus.",
       "Use clean, certified planting material, control ants and weeds, and inspect regularly."),
    _e("veg_caterpillar", "Caterpillars (diamondback moth / cabbage worm)", "Pest",
       r"caterpillars?|worms?", ["Vegetables"],
       "Larvae of the diamondback moth or cabbage looper.",
       "Holes chewed in leaves and green droppings.",
       "Moths lay eggs on leaves; warm, dry weather boosts numbers.",
       "Hand-pick larvae, spray Bt or neem, rotate products to avoid resistance.",
       "Rotate with non-cabbage crops, use nets, encourage natural enemies."),
    _e("bacterial_wilt", "Bacterial wilt / Fusarium wilt", "Disease",
       r"wilting despite|wilting.*watering", None,
       "Bacterium *Ralstonia solanacearum* or *Fusarium* fungus.",
       "Plants wilt suddenly and do not recover even with water. A cut stem in clear water may release milky ooze (bacterial wilt).",
       "Infected soil or water, warm weather, waterlogging, and root injury.",
       "Uproot and destroy affected plants (do not compost) and avoid moving soil or water from the affected area. There is no effective cure.",
       "Rotate with non-solanaceous crops (e.g., corn), use grafted or resistant varieties, improve drainage, disinfect tools."),
    _e("powdery_mildew", "Powdery mildew", "Disease",
       r"mildew", None,
       "Fungus (e.g., *Leveillula taurica*).",
       "White powdery patches on leaves that later turn yellow and dry.",
       "Warm days and humid nights, crowded plants and poor airflow.",
       "Remove badly affected leaves and spray sulfur, neem oil or a registered fungicide as labeled.",
       "Give plants space, avoid late-day overhead watering, choose tolerant varieties."),
    _e("coconut_beetle", "Rhinoceros beetle / palm weevil", "Pest",
       r"beetles?|boring", ["Coconut"],
       "Beetles such as *Oryctes rhinoceros* or the red palm weevil.",
       "Holes in the trunk or crown, chewed V-shaped cuts in fronds, and sawdust or oozing.",
       "Beetles breed in rotting logs, dung and dead palms and attack weakened palms.",
       "Remove and destroy breeding sites and beetles, and consult PCA for control (traps, biocontrol).",
       "Keep the farm clean of rotting trunks and debris, avoid wounding palms."),
    _e("coconut_yellow", "Coconut frond yellowing (needs field check)", "Other",
       r"drooping|fronds?.*yellow", ["Coconut"],
       "Possible causes: nutrient deficiency (potassium, magnesium), water stress, or **cadang-cadang** (a viroid disease).",
       "Yellow, drooping fronds. Cadang-cadang shows spreading translucent yellow spots, smaller round nuts, and a shrinking crown.",
       "Poor soil, drought and unbalanced fertilization are the most common. Cadang-cadang spreads slowly and can only be confirmed by a lab test.",
       "Apply balanced fertilizer (including potassium and chloride/salt as advised) and water in dry spells. If yellowing spreads palm by palm, report it to the Philippine Coconut Authority (PCA). Cadang-cadang has no cure.",
       "Regular fertilizer, good drainage, use certified seedlings."),
    _e("coconut_brown", "Coconut leaf browning (needs field check)", "Other",
       r"leaves.*brown|brown.*leaves", ["Coconut"],
       "Possible causes: leaf spot fungi, scale insects, or drought or nutrient stress.",
       "Fronds turn brown from the tips or in patches and dry out.",
       "Long dry spells, insect infestation under the fronds, fungal leaf spot in humid weather, or low fertilizer.",
       "Inspect the underside of leaves for scale insects, prune and burn badly dried fronds, and fertilize. Ask PCA or your municipal agriculturist if it spreads.",
       "Fertilize regularly, keep the crown clean, watch for insects, avoid water stress."),
    _e("coconut_yield", "Low nut yield (stress or nutrition problem)", "Other",
       r"fewer nuts|less nuts", ["Coconut"],
       "Usually stress or nutrition, sometimes disease (e.g., cadang-cadang).",
       "Fewer or smaller nuts per bunch, and premature nut drop.",
       "Potassium or chloride deficiency, drought, old palms, poor pollination, pests or disease.",
       "Fertilize per soil needs, water in dry months, and check palms for pests or yellowing.",
       "Regular fertilization, replace very old palms, control pests early."),
    _e("storm", "Storm / heavy-rain damage", "Weather",
       r"storm|heavy rain|strong rains?|fall over|fell over", None,
       "Storm winds and heavy rain (not a pathogen).",
       "Broken, toppled or waterlogged plants and damaged beds.",
       "Typhoon or heavy rainfall events.",
       "Remove broken parts, prop up leaning plants, drain standing water, and spray a protective fungicide if leaves are wet or torn (disease risk rises after storms).",
       "Use windbreaks, raised beds and drainage, stake tall crops, and buy crop insurance if available."),
    _e("insects_leaf", "Leaf-feeding insects (leaffolder / armyworm)", "Pest",
       r"insects.*(eating|leaves)|eating.*leaves", ["Rice"],
       "Leaf-feeding insects such as the rice leaffolder or armyworm.",
       "Chewed, scraped or folded leaves; white streaks on leaves.",
       "Heavy nitrogen use, dense planting, and warm weather attract these insects.",
       "Scout the field; spray only when damage passes the economic threshold, using a registered insecticide or Bt as labeled.",
       "Avoid excess nitrogen, keep the field clean, protect spiders and other natural enemies."),
    _e("planthopper", "Planthoppers / leafhoppers (virus carriers)", "Pest",
       r"small insects|insects around", ["Rice"],
       "Brown planthopper or green leafhopper.",
       "Small insects at the base of the plants; yellowing or drying in circular patches ('hopperburn'); can spread tungro virus.",
       "Excess nitrogen, continuous planting, and misuse of insecticide that kills natural enemies.",
       "Drain the field for a few days, apply a registered insecticide only when numbers are high, following the label.",
       "Resistant varieties, synchronized planting, balanced fertilizer, protect natural enemies."),
    _e("banana_leaf_drop", "Panama disease (Fusarium wilt) or Sigatoka — possible", "Disease",
       r"leaves?\b.*\b(falling|fall|drooping|droop|hanging|collaps\w*|breaking|dropping)|\b(falling|drooping|collapsing)\b.*leaves?", ["Banana"],
       "Possible causes: Panama disease (soil fungus *Fusarium oxysporum* f. sp. *cubense*), Black Sigatoka leaf spot, or banana bunchy top virus.",
       "Older leaves turn yellow, droop and hang down around the stem like a skirt, then dry and fall; in Panama disease the stem may split near the base and the cut stem shows reddish-brown streaks.",
       "Panama disease lives in the soil for many years and spreads through infected suckers, tools, soil and water. Black Sigatoka thrives in warm, humid, rainy weather.",
       "Do not replant suckers from affected plants. Remove and destroy badly affected plants (do not compost), cut off dying leaves, keep tools and boots clean, and ask your municipal agriculturist / DA technician to inspect the field. Panama disease has no cure.",
       "Use clean, certified planting material, improve drainage, avoid moving soil or water from affected areas, disinfect tools, and consider resistant varieties."),
    _e("banana_yellow_edges", "Banana leaf yellowing (needs field check)", "Other",
       r"yellow(ing)?.*edges|edges.*yellow", ["Banana"],
       "Possible causes: potassium deficiency, early Panama disease, or water stress.",
       "Leaf edges turn yellow then brown and dry, usually starting on the older leaves.",
       "Low potassium in the soil is the most common cause; yellowing that spreads plant by plant may be a soil disease.",
       "Apply balanced fertilizer with potassium (ideally after a soil test). If plants keep yellowing and the stem splits, ask a technician to check for Panama disease.",
       "Regular fertilizing with potassium, compost or mulch, good drainage and clean planting material."),
    _e("banana_insects", "Leaf-feeding insects (banana skipper / caterpillars)", "Pest",
       r"insects?.*(feeding|eating)|(feeding|eating).*leaves", ["Banana"],
       "Leaf-feeding insects such as the banana skipper caterpillar, or beetles.",
       "Chewed or rolled young leaves, ragged leaf edges and droppings on the leaves.",
       "Warm weather, weedy fields and lack of natural enemies allow numbers to build up.",
       "Scout the field, pick off and destroy caterpillars, cut heavily damaged leaves, and use a registered or biological insecticide (e.g., Bt) only if damage is heavy.",
       "Keep the field clean and weed-free, protect natural enemies, and inspect young leaves every week."),
    _e("banana_weak", "Weak stems and poor growth (needs field check)", "Other",
       r"weak stems?|poor growth", ["Banana"],
       "Possible causes: banana weevil borer, nutrient shortage, or root damage.",
       "Thin or soft pseudostems, slow growth and small bunches; weevil damage shows tunnels at the base of the plant.",
       "Weevil larvae boring in the corm, poor or depleted soil, or too little fertilizer or water.",
       "Dig around the base to check for weevil tunnels, remove and destroy infested corms, fertilize, and ask a technician to inspect the field.",
       "Use clean suckers, remove old stumps and trash near plants, and fertilize regularly."),
    _e("nutrient", "Nutrient deficiency (mainly nitrogen)", "Nutrient",
       r"nutrient|fertilizer|pale|yellow.*(slow|growing)|turning yellow", None,
       "Lack of nutrients in the soil (not a pathogen). Damping-off can look similar in seedlings.",
       "Pale or yellow older leaves, curling, and slow growth.",
       "Poor or depleted soil, not enough fertilizer, leaching after heavy rain, or waterlogged roots.",
       "Apply balanced fertilizer (nitrogen first for yellow older leaves), or compost or manure, ideally after a soil test.",
       "Soil testing, regular fertilizing, compost, and rotating with legumes."),
    _e("drought", "Drought / water stress", "Water",
       r"very dry|dry|wilting", None,
       "Dry soil (not a pathogen).",
       "Wilting, leaf rolling and stunted growth.",
       "Not enough rain or irrigation.",
       "Irrigate right away and mulch to reduce evaporation.",
       "Water on schedule, drought-tolerant varieties."),
    _e("unspecified", "Unspecified crop health problem", "Other",
       r"unhealthy|not (very |so )?healthy|not (doing )?well|not good|\\bsick\\b|something wrong|has a problem|have a problem", None,
       "Not enough detail in the report.",
       "General poor plant health.",
       "Cannot tell from the description.",
       "Inspect leaves, stems, roots and soil, and ask for a field visit.",
       "Ask farmers to describe symptoms (color, spots, insects, when it started)."),
]

FALLBACK = dict(key="unclassified", name="Unclassified problem", type="Other", crops=None,
                agent="Not enough detail to identify.", symptoms="—", causes="—",
                treatment="Ask a technician to inspect the field.", prevention="—")
_BY_KEY = {e["key"]: e for e in KB}


GENERIC = {
    "Disease": ("Possible plant disease (needs field check)",
                "A disease is suspected from the wording, but the report does not have enough detail to name it.",
                "Leaf spots, yellowing, wilting, rotting or leaves that dry and fall.",
                "Fungi, bacteria or viruses, usually spread by wet weather, infected soil, tools or planting material.",
                "Cut off and remove affected leaves or plants, avoid spraying blindly, and ask a technician to inspect the field and confirm the disease.",
                "Use clean planting material, keep good drainage and spacing, and clean tools."),
    "Pest": ("Possible pest damage (needs field check)",
             "An insect or animal pest is suspected from the wording, but the report does not have enough detail to name it.",
             "Chewed leaves, holes, insects on the plant, or damaged stems.",
             "Warm weather, weedy fields and lack of natural enemies.",
             "Look under the leaves and at the base of the plant, remove the pests you find, and ask a technician before spraying.",
             "Scout weekly, keep the field clean and protect natural enemies."),
    "Water": ("Possible water problem (needs field check)",
              "Too little or too much water is suspected from the wording.",
              "Wilting, leaf rolling, or yellowing with soggy soil.",
              "Dry spell, broken irrigation, or poor drainage.",
              "Check soil moisture, irrigate or drain the field, and mulch.",
              "Plan irrigation and drainage, use mulch."),
    "Weather": ("Possible weather damage (needs field check)",
                "Damage from rain, wind or heat is suspected from the wording.",
                "Broken, toppled or waterlogged plants.",
                "Typhoon, heavy rain, strong wind or heat.",
                "Remove broken parts, prop up leaning plants and drain standing water.",
                "Use windbreaks, drainage and staking."),
    "Nutrient": ("Possible nutrient problem (needs field check)",
                 "A shortage of nutrients is suspected from the wording.",
                 "Pale or yellow leaves and slow growth.",
                 "Poor soil or too little fertilizer.",
                 "Apply balanced fertilizer or compost, ideally after a soil test.",
                 "Test the soil and fertilize regularly."),
}
_CROP_WORDS = {"Banana": r"\bbanana", "Rice": r"\brice\b|\bpalay", "Corn": r"\bcorn\b|\bmais",
               "Coconut": r"\bcoconut|\bniyog", "Vegetables": r"\bvegetable"}


def _infer_crop(text):
    """Guess the crop from words in the report (used when the user did not pick one)."""
    for crop, pat in _CROP_WORDS.items():
        if re.search(pat, text, re.I):
            return crop
    return None


def _generic(category):
    name, agent, symptoms, causes, treatment, prevention = GENERIC[category]
    return dict(key=f"generic_{category.lower()}", name=name, type=category, crops=None, agent=agent,
                symptoms=symptoms, causes=causes, treatment=treatment, prevention=prevention)


def _matches(crop, text):
    """Every knowledge entry that fits this crop + report text, in priority order."""
    return [e for e in KB
            if not (e["crops"] and crop not in e["crops"]) and e["pattern"].search(text)]


def diagnose(crop, text, category=None):
    """Return the best-matching knowledge entry for one report.

    crop      : crop name, or None (then it is guessed from the text)
    category  : optional AI category (Disease, Pest, Water, Weather, Nutrient, Other).
                The diagnosis is kept consistent with it:
                  1. if several entries fit the text, the one whose type equals the AI category wins;
                  2. an ambiguous "needs field check" entry (type Other) takes the AI category as its type;
                  3. if nothing specific fits, a general entry for that category is returned.
    """
    t = str(text or "")
    crop = crop or _infer_crop(t)
    hits = _matches(crop, t)
    if hits:
        if category:
            same = next((e for e in hits if e["type"] == category), None)
            if same:
                return same
            first = hits[0]
            if first["type"] == "Other" and category in TYPE_COLORS and category != "Other":
                return {**first, "type": category}
            return first
        return hits[0]
    if category in GENERIC:
        return _generic(category)
    return FALLBACK


def _first_sentence(text):
    text = str(text).strip()
    cut = re.search(r"(?<=[a-z\)])\.\s", text)
    return text[:cut.start() + 1] if cut else text


def general_guidance(crop=None):
    """Symptoms / causes / prevention for a report that is too vague to identify (e.g. "my rice is not healthy").

    Nothing is invented: it lists the common problems of that crop taken from this knowledge base, clearly labelled
    as things to CHECK, not as a diagnosis. Without a crop it lists the general problem types instead.
    """
    if crop:
        own = [e for e in KB if e["crops"] and crop in e["crops"]]
        rest = [e for e in KB if not e["crops"] and e["key"] != "unspecified" and e["type"] != "Other"]
        rows = [(e["name"], e["symptoms"], e["causes"], e["prevention"]) for e in (own + rest)[:6]]
        title = f"Needs more detail (common {crop} problems to check)"
    else:
        rows = [(f"{c}: {v[0]}", v[2], v[3], v[5]) for c, v in GENERIC.items() if c != "Other"]
        title = "Needs more detail (common crop problems to check)"
    bullets = lambda i: "\n\n" + "\n".join(f"- **{r[0]}:** {_first_sentence(r[i])}" for r in rows)
    what = crop.lower() if crop else "crop"
    return dict(
        key="unspecified", name=title, type="Other", crops=None,
        agent=(f"The report does not say what is wrong with the {what}, so the exact problem cannot be identified. "
               "Below are the common problems to check."),
        symptoms="Common signs to look for:" + bullets(1),
        causes="Possible causes (not confirmed for this report):" + bullets(2),
        treatment=("Inspect the leaves, stems, roots and soil, then ask the farmer to describe: which part is affected, "
                   "the colour or spots, insects seen, and when it started. Ask a technician to inspect the field."),
        prevention="General prevention:" + bullets(3),
    )


def match_in_text(text, crop=None):
    """All entries whose symptoms appear in free text (used when a user types symptoms into the chat)."""
    out = []
    for e in KB:
        if crop and e["crops"] and crop not in e["crops"]:
            continue
        if e["pattern"].search(text) and e["key"] != "unspecified":
            out.append(e)
    return out


def add_diagnosis(df):
    """Add likely_issue, issue_type and issue_key columns to a reports DataFrame.

    If the frame has an AI `category` column, each diagnosis is made consistent with it.
    """
    if df is None or df.empty or "likely_issue" in df.columns:
        return df
    df = df.copy()
    cats = df["category"] if "category" in df.columns else [None] * len(df)
    ds = [diagnose(c, r, None if str(k) in ("Not analyzed", "nan", "None", "—") else k)
          for c, r, k in zip(df["crop"], df["report"], cats)]
    df["likely_issue"] = [d["name"] for d in ds]
    df["issue_type"] = [d["type"] for d in ds]
    df["issue_key"] = [d["key"] for d in ds]
    return df


def entry_by_key(key):
    if str(key).startswith("generic_") and str(key)[8:].title() in GENERIC:
        return _generic(str(key)[8:].title())
    return _BY_KEY.get(key, FALLBACK)