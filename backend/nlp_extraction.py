"""
nlp_extraction.py
-----------------
NLP pipeline jo OCR se nikle hue raw text ko prediction form ke fields
par map karta hai.

Ye module ICD_Project.ipynb ke "Section 16: NLP-Based OCR Text Extraction"
mein develop kiya gaya hai, aur bilkul wahi logic yahan reusable form mein
rakha gaya hai taake backend (main.py) aur notebook dono ek hi source of
truth use karein.

IMPORTANT: Is module ka deficiency-prediction model se koi taluq nahi hai.
Ye sirf text parhta hai - koi ML model train/load/change nahi karta.

Pipeline ke 6 steps:
    1. Text Normalization   - OCR ki common ghaltiyan theek karna
    2. Line Segmentation    - text ko lines/tokens mein todna
    3. Lexicon Matching     - har field ke synonyms dhoondhna (fuzzy)
    4. Value Parsing        - number + unit nikalna
    5. Unit Conversion      - ml->L, mg->g, lbs->kg, feet->cm waghera
    6. Confidence Scoring   - har value ko score dena + validate karna

Beginner-friendly: sirf Python standard library (re, difflib, unicodedata)
use hoti hai. spaCy agar installed ho to tokenization behtar ho jati hai,
warna module bilkul theek chalta rehta hai.
"""

import re
import unicodedata
from difflib import SequenceMatcher

# --- Optional spaCy support -------------------------------------------------
# spaCy sirf behtar tokenization/lemmatization ke liye use hoti hai. Agar
# installed nahi hai to module apne simple tokenizer par fallback kar leta
# hai - kuch bhi crash nahi hota.
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = spacy.blank("en")  # model download nahi hua, blank English
    SPACY_AVAILABLE = True
except ImportError:
    _NLP = None
    SPACY_AVAILABLE = False


# ===========================================================================
# STEP 0: Field configuration
# ===========================================================================

# HARD_RANGE  = is se bahar value = OCR ne ghalat parha -> reject kar do
# TYPICAL_RANGE = normal insaani range -> is se bahar ho to "confirm karo"
FIELD_RANGES = {
    #                  hard_min, hard_max, typical_min, typical_max
    "Age":              (1,   120,   1,    100),
    "Calories":         (0,  10000, 500,   4500),
    "Protein":          (0,   500,   10,    250),
    "Vegetable_Intake": (0,    20,    0,     10),
    "Fruit_Intake":     (0,    20,    0,     10),
    "Dairy_Intake":     (0,    20,    0,     10),
    "Water_Intake":     (0,    20,  0.5,      8),
    "Meal_Frequency":   (1,    15,    1,      8),
    "BMI":              (5,    80,   12,     50),
    "Height":           (50,  250,  120,    210),
    "Weight":           (10,  300,   25,    180),
}

# Har field ke liye trigger phrases (synonyms). Ye "lexicon" NLP ka dil hai -
# regex ke bajaye hum natural language ke mukhtalif tareeqon ko samajhte hain.
FIELD_LEXICON = {
    "Age": ["age", "aged", "years old", "yrs old", "umar", "patient age"],
    "Calories": ["calories", "calorie", "kcal", "caloric intake", "energy",
                 "total calories", "daily calories", "calories per day"],
    "Protein": ["protein", "proteins", "total protein", "protein intake"],
    "Vegetable_Intake": ["vegetable intake", "vegetables", "vegetable", "veggies",
                         "veg intake", "sabzi", "vegetable servings"],
    "Fruit_Intake": ["fruit intake", "fruits", "fruit", "fruit servings", "phal"],
    "Dairy_Intake": ["dairy intake", "dairy", "milk products", "dairy servings",
                     "milk intake", "doodh"],
    "Water_Intake": ["water intake", "water", "fluid intake", "fluids", "hydration",
                     "h2o", "paani", "water consumption"],
    "Meal_Frequency": ["meal frequency", "meals per day", "meals a day", "meals",
                       "number of meals", "meal count", "eating frequency"],
    "BMI": ["bmi", "body mass index", "b m i"],
    "Height": ["height", "ht", "stature", "qad"],
    "Weight": ["weight", "wt", "body weight", "wazan"],
}

CATEGORICAL_LEXICON = {
    "Gender": {
        "Female": ["female", "woman", "girl", "f/", "sex: f", "gender: f", "aurat", "khatoon"],
        "Male": ["male", "man", "boy", "m/", "sex: m", "gender: m", "mard", "aadmi"],
    },
    "Diet_Type": {
        "Vegan": ["vegan", "plant based", "plant-based", "purely vegetarian"],
        "Non-Vegetarian": ["non vegetarian", "non-vegetarian", "nonveg", "non veg",
                           "non-veg", "meat eater", "omnivore", "mixed diet"],
        "Vegetarian": ["vegetarian", "veg diet", "veg", "lacto vegetarian"],
    },
    "Physical_Activity": {
        "High": ["high activity", "very active", "highly active", "active lifestyle",
                 "vigorous", "athlete", "activity: high", "exercise: high"],
        "Moderate": ["moderate activity", "moderately active", "moderate",
                     "activity: moderate", "exercise: moderate", "average activity"],
        "Low": ["low activity", "sedentary", "inactive", "not active",
                "activity: low", "exercise: low", "minimal activity"],
    },
}

# Units -> conversion factor into the unit our form expects.
# Form units: Calories=kcal, Protein=g, Water=litres, Height=cm, Weight=kg
UNIT_CONVERSIONS = {
    "Water_Intake": {
        "ml": 0.001, "milliliter": 0.001, "millilitre": 0.001, "cc": 0.001,
        "l": 1.0, "litre": 1.0, "liter": 1.0, "litres": 1.0, "liters": 1.0,
        "glass": 0.25, "glasses": 0.25,   # 1 glass ~ 250 ml
        "cup": 0.25, "cups": 0.25,
    },
    "Protein": {
        "mg": 0.001, "milligram": 0.001,
        "g": 1.0, "gm": 1.0, "gram": 1.0, "grams": 1.0, "gms": 1.0,
    },
    "Calories": {
        "kcal": 1.0, "cal": 1.0, "calorie": 1.0, "calories": 1.0, "kj": 0.239,
    },
    "Height": {
        "cm": 1.0, "centimeter": 1.0, "centimetre": 1.0,
        "m": 100.0, "meter": 100.0, "metre": 100.0,
        "in": 2.54, "inch": 2.54, "inches": 2.54,
        "ft": 30.48, "feet": 30.48, "foot": 30.48,
    },
    "Weight": {
        "kg": 1.0, "kilogram": 1.0, "kgs": 1.0, "kilo": 1.0,
        "g": 0.001, "gram": 0.001,
        "lb": 0.4536, "lbs": 0.4536, "pound": 0.4536, "pounds": 0.4536,
    },
}

# Confidence thresholds
FUZZY_ACCEPT = 0.82      # is se upar ka fuzzy match qabool
CONFIDENCE_AUTOFILL = 0.70   # is se upar = field auto-fill kar do
# 0.45 - 0.70 = "needs confirmation" (field khaali, user ko dikhayenge)
CONFIDENCE_MIN = 0.45


# ===========================================================================
# STEP 1: Text Normalization
# ===========================================================================

# OCR aksar in characters ko confuse karta hai - lekin sirf NUMBER ke andar
# in ko theek karna safe hai (warna "Iron" -> "1ron" ho jayega).
OCR_DIGIT_FIXES = {
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "l": "1", "I": "1", "|": "1", "!": "1",
    "S": "5", "s": "5",
    "B": "8", "Z": "2", "z": "2", "g": "9",
}


def normalize_text(raw_text):
    """OCR text ko saaf karna: unicode fix, spacing fix, lowercase."""
    if not raw_text:
        return ""

    # Unicode normalize (smart quotes, accents waghera)
    text = unicodedata.normalize("NFKC", raw_text)

    # Common OCR punctuation noise
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # ":" ke aas paas ka random space normalize karo ("Age : 25" -> "Age: 25")
    text = re.sub(r"\s*:\s*", ": ", text)

    # Number ke andar ka ghalat space hatao ("1 800" -> "1800")
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)

    # Thousands separator hatao ("1,800" -> "1800")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    # Multiple spaces -> single space (lekin newlines bachao)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def fix_ocr_digits(token):
    """Sirf un tokens par chalao jo number hone chahiye thay."""
    fixed = "".join(OCR_DIGIT_FIXES.get(ch, ch) for ch in token)
    return fixed


# ===========================================================================
# STEP 2: Line Segmentation & Tokenization
# ===========================================================================

def segment_lines(text):
    """Text ko meaningful lines mein todna. Khaali lines hata dete hain."""
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]


def tokenize(line):
    """
    Line ko tokens mein todna. spaCy available ho to uska tokenizer use hota
    hai (behtar handling of punctuation/units), warna simple regex.
    """
    if SPACY_AVAILABLE and _NLP is not None:
        return [tok.text.lower() for tok in _NLP(line) if not tok.is_space]
    return re.findall(r"[A-Za-z]+|\d+\.?\d*|[^\w\s]", line.lower())


# ===========================================================================
# STEP 3: Lexicon Matching (fuzzy)
# ===========================================================================

def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_field_mention(line, phrases):
    """
    Line ke andar dhoondho ke koi trigger phrase mojood hai ya nahi.

    Returns: (match_score, char_index_where_phrase_ends) ya (0.0, -1)
    - Exact substring match  -> score 1.0
    - Fuzzy (OCR typo) match -> score = similarity ratio
    """
    low = line.lower()
    best_score, best_end = 0.0, -1

    def _consider(score, end):
        """Behtar score lo; barabar score par SAB SE PEHLE wali position lo.

        Ye zaroori hai kyunki kuch trigger phrases asal mein units hain jo
        number ke BAAD aate hain (misal "Energy: 2200 kcal" - yahan "kcal"
        bhi Calories ka trigger hai). Agar hum aakhri match uthayen to
        number search region se bahar reh jata hai.
        """
        nonlocal best_score, best_end
        if score > best_score or (score == best_score and 0 <= end < best_end):
            best_score, best_end = score, end

    for phrase in phrases:
        # (a) Exact substring - sab se strong signal
        idx = low.find(phrase)
        if idx != -1:
            _consider(1.0, idx + len(phrase))
            continue

        # (b) Fuzzy: phrase ko sliding window ke saath compare karo
        words = re.findall(r"[a-z]+", low)
        n = len(phrase.split())
        for i in range(len(words) - n + 1):
            window = " ".join(words[i:i + n])
            score = _similarity(phrase, window)
            if score >= FUZZY_ACCEPT:
                # window ka approx end position line mein
                pos = low.find(words[i + n - 1])
                end = (pos + len(words[i + n - 1])) if pos != -1 else 0
                _consider(score, end)

    return best_score, best_end


# ===========================================================================
# STEP 4 + 5: Value Parsing & Unit Conversion
# ===========================================================================

NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-zA-Z/']*)")
FEET_INCH_RE = re.compile(r"(\d+)\s*(?:'|ft|feet)\s*(\d+)?\s*(?:\"|in|inch)?")


def parse_value(line, field, keyword_end):
    """
    Line se number + unit nikalo aur form ke unit mein convert karo.

    keyword_end = jahan trigger phrase khatam hui - us ke BAAD wala number
    sab se relevant hota hai ("Water Intake: 2 L" mein 2).
    """
    # Height ke liye pehle feet'inch format check karo (5'8")
    if field == "Height":
        m = FEET_INCH_RE.search(line)
        if m:
            feet = float(m.group(1))
            inches = float(m.group(2)) if m.group(2) else 0.0
            return round(feet * 30.48 + inches * 2.54, 1), "ft/in"

    search_region = line[keyword_end:] if keyword_end > 0 else line

    candidates = list(NUMBER_RE.finditer(search_region))

    if not candidates:
        # Fallback 1: keyword ke BAAD koi number nahi mila - poori line dekho.
        # ("Calories 1800" jahan match "kcal" par hua ho, ya "25 years - age")
        candidates = list(NUMBER_RE.finditer(line))

    if not candidates:
        # Fallback 2: shayad OCR ne digit ko letter parh liya ("2S" -> "25")
        candidates = list(NUMBER_RE.finditer(fix_ocr_digits(line)))

    if not candidates:
        return None, None

    match = candidates[0]
    try:
        value = float(match.group(1))
    except ValueError:
        return None, None

    unit = (match.group(2) or "").lower().strip(".")

    # Unit conversion agar is field ke liye units defined hain
    conversions = UNIT_CONVERSIONS.get(field)
    if conversions and unit in conversions:
        value = value * conversions[unit]
        value = round(value, 2)

    return value, (unit or None)


# ===========================================================================
# STEP 6: Confidence Scoring & Validation
# ===========================================================================

def score_and_validate(field, value, match_score, had_unit):
    """
    Har extracted value ko confidence score do aur range ke against check karo.

    Returns: (final_value_or_None, confidence, status)
      status = "auto"    -> field bhar do
               "confirm" -> user se confirm karwao, field khaali rakho
               "reject"   -> bilkul ignore karo
    """
    if value is None:
        return None, 0.0, "reject"

    hard_min, hard_max, typ_min, typ_max = FIELD_RANGES[field]

    # Hard range se bahar = OCR ne yaqeenan ghalat parha
    if not (hard_min <= value <= hard_max):
        return None, 0.0, "reject"

    confidence = match_score

    # Unit mila to confidence barh jati hai ("2 L" > sirf "2")
    if had_unit:
        confidence = min(1.0, confidence + 0.10)

    # Typical range se bahar = ho sakta hai theek ho, magar shak hai
    # (jaise "Protein: 459" jo asal mein "45 g" tha)
    if not (typ_min <= value <= typ_max):
        confidence *= 0.55

    # Integer-only fields
    if field in ("Age", "Meal_Frequency"):
        value = int(round(value))

    if confidence >= CONFIDENCE_AUTOFILL:
        return value, round(confidence, 2), "auto"
    if confidence >= CONFIDENCE_MIN:
        return value, round(confidence, 2), "confirm"
    return None, round(confidence, 2), "reject"


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def extract_numeric_fields(text):
    """Saare numeric fields ke liye NLP pipeline chalao."""
    results = {}
    lines = segment_lines(text)

    for field, phrases in FIELD_LEXICON.items():
        best = None

        for line in lines:
            match_score, keyword_end = find_field_mention(line, phrases)
            if match_score <= 0:
                continue

            value, unit = parse_value(line, field, keyword_end)
            if value is None:
                continue

            final_value, confidence, status = score_and_validate(
                field, value, match_score, had_unit=bool(unit)
            )
            if status == "reject":
                continue

            candidate = {
                "value": final_value,
                "confidence": confidence,
                "status": status,
                "unit": unit,
                "evidence": line.strip()[:80],
            }
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate

        if best is not None:
            results[field] = best

    return results


def extract_categorical_fields(text):
    """Gender / Diet_Type / Physical_Activity nikalo."""
    results = {}
    low = normalize_text(text).lower()

    for field, options in CATEGORICAL_LEXICON.items():
        best_option, best_score, best_evidence = None, 0.0, ""

        for option, phrases in options.items():
            for phrase in phrases:
                if phrase in low:
                    # Lambi phrase = zyada specific = zyada confidence
                    # ("non-vegetarian" ko "vegetarian" par tarjeeh milegi)
                    score = 0.75 + min(0.25, len(phrase) / 60.0)
                    if score > best_score:
                        idx = low.find(phrase)
                        best_option = option
                        best_score = score
                        best_evidence = low[max(0, idx - 20): idx + len(phrase) + 12].strip()

        if best_option and best_score >= CONFIDENCE_MIN:
            results[field] = {
                "value": best_option,
                "confidence": round(min(best_score, 1.0), 2),
                "status": "auto" if best_score >= CONFIDENCE_AUTOFILL else "confirm",
                "unit": None,
                "evidence": best_evidence[:80],
            }

    return results


ALL_FIELDS = list(FIELD_LEXICON.keys()) + list(CATEGORICAL_LEXICON.keys())


def extract_fields_from_text(raw_text):
    """
    Poori NLP pipeline ka entry point.

    Input : OCR ka raw text (string)
    Output: dict -> {
        "auto":    {field: value}          # in fields ko bhar do
        "confirm": {field: value}          # user confirm kare, field khaali
        "details": {field: {...metadata}}  # confidence, unit, evidence
        "not_detected": [field, ...]
    }
    """
    text = normalize_text(raw_text)
    if not text:
        return {"auto": {}, "confirm": {}, "details": {}, "not_detected": list(ALL_FIELDS)}

    details = {}
    details.update(extract_numeric_fields(text))
    details.update(extract_categorical_fields(text))

    auto = {f: d["value"] for f, d in details.items() if d["status"] == "auto"}
    confirm = {f: d["value"] for f, d in details.items() if d["status"] == "confirm"}
    not_detected = [f for f in ALL_FIELDS if f not in details]

    return {
        "auto": auto,
        "confirm": confirm,
        "details": details,
        "not_detected": not_detected,
    }
