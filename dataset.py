"""
dataset.py
----------
Builds a realistic search-query log to feed the sketch.

Real search traffic is heavily skewed: a handful of queries ("weather today",
"ipl live score") are searched constantly, and a very long tail of queries is
searched once or twice. That shape is called a Zipf distribution - the query
at popularity rank r is searched roughly 1/r as often as the top one.

The log is also deliberately messy (mixed case, stray spaces) like real typed
input, so the program has to normalise each query before counting it.

Run directly to (re)create the input file:
    python dataset.py
"""

import random

DEFAULT_PATH = "search_queries.txt"
DEFAULT_SIZE = 200_000

# Popular "head" queries, most popular first.
HEAD_QUERIES = [
    "weather today", "ipl live score", "gold rate today", "youtube", "gmail login",
    "whatsapp web", "chatgpt", "google translate", "irctc train booking", "news today",
    "petrol price today", "instagram login", "cricket score", "amazon", "flipkart",
    "speed test", "google maps", "sensex today", "usd to inr", "netflix",
    "neet 2026 result", "jee main 2026", "aadhaar card download", "pan card status",
    "pnr status", "online compiler python", "leetcode", "github", "linkedin jobs",
    "resume format for freshers", "what is machine learning", "python tutorial",
    "data structures and algorithms", "count min sketch", "bloom filter",
    "binary search tree", "dijkstra algorithm", "time complexity of quicksort",
    "hash table vs hash map", "big o notation", "movies near me", "restaurants near me",
    "atm near me", "pharmacy near me", "petrol bunk near me", "bus timings",
    "metro train timings", "today tamil calendar", "tamil new movies", "trending songs",
    "sbi net banking", "upi payment failed", "income tax e filing", "passport status",
    "voter id download", "driving licence apply online", "covid vaccine certificate",
    "electricity bill payment", "mobile recharge offers", "flight ticket booking",
    "hotel booking", "amrita university", "anna university results", "gate 2027 syllabus",
    "internship for btech students", "hackathon 2026", "how to learn coding",
    "how to make resume", "how to tie a tie", "how to take screenshot in laptop",
    "how to reduce stress", "how to improve cgpa", "how to prepare for placements",
    "biryani recipe", "dosa batter recipe", "filter coffee recipe", "paneer butter masala recipe",
    "world cup schedule", "olympics medal tally", "stock market today",
]

CITIES = [
    "chennai", "coimbatore", "madurai", "trichy", "salem", "bengaluru", "mysuru",
    "hyderabad", "vizag", "kochi", "trivandrum", "mumbai", "pune", "nagpur", "delhi",
    "noida", "gurgaon", "jaipur", "lucknow", "kanpur", "patna", "kolkata",
    "bhubaneswar", "guwahati", "ahmedabad", "surat", "indore", "bhopal", "chandigarh",
    "amritsar", "dehradun", "goa", "pondicherry", "vellore", "tirupati", "mangalore",
]
TRAVEL_MODES = [
    "train", "flight", "bus", "distance", "cab fare", "train timings",
    "flight ticket price", "road trip",
]
CITY_SUFFIXES = [
    "weather", "news", "pincode", "hotels", "tourist places", "colleges",
    "hospitals", "it companies", "metro map", "rain update",
]
PRODUCTS = [
    "laptop", "smartphone", "earbuds", "smartwatch", "tablet", "power bank", "monitor",
    "mechanical keyboard", "gaming mouse", "headphones", "bluetooth speaker", "printer",
    "wifi router", "ssd", "pen drive", "webcam", "air conditioner", "washing machine",
    "refrigerator", "microwave oven", "mixer grinder", "air fryer", "water purifier",
    "led tv", "cycle", "electric scooter", "running shoes", "backpack", "trimmer",
    "study table",
]
BUDGETS = ["1000", "5000", "10000", "20000", "30000", "50000", "80000"]
PRODUCT_SUFFIXES = ["review", "price", "offers", "specifications", "vs comparison"]
SUBJECTS = [
    "data structures", "operating systems", "dbms", "computer networks", "machine learning",
    "deep learning", "linear algebra", "probability", "discrete mathematics", "python",
    "java", "c programming", "robotics", "signals and systems", "digital electronics",
    "artificial intelligence", "data science", "statistics", "calculus", "compiler design",
]
SUBJECT_SUFFIXES = [
    "notes pdf", "syllabus", "important questions", "previous year question paper",
    "interview questions", "mcq", "lab manual", "online course", "book pdf", "projects",
]
DISHES = [
    "sambar", "rasam", "idli", "pongal", "chapati", "poori", "veg pulao", "fried rice",
    "noodles", "pasta", "pizza", "sandwich", "omelette", "chicken curry", "fish fry",
    "payasam", "gulab jamun", "cake", "lemon rice", "curd rice",
]
DISH_SUFFIXES = ["recipe", "recipe in tamil", "calories", "without oven", "for beginners"]


def build_vocabulary(rng):
    """All distinct queries, ordered from most to least popular."""
    tail = []
    for a in CITIES:
        for b in CITIES:
            if a != b:
                tail.extend(f"{a} to {b} {mode}" for mode in TRAVEL_MODES)
        tail.extend(f"{a} {suffix}" for suffix in CITY_SUFFIXES)
    for product in PRODUCTS:
        tail.extend(f"best {product} under {budget}" for budget in BUDGETS)
        tail.extend(f"{product} {suffix}" for suffix in PRODUCT_SUFFIXES)
        for city in CITIES:
            tail.append(f"{product} showroom in {city}")
            tail.append(f"{product} service center in {city}")
            tail.append(f"second hand {product} in {city}")
    for subject in SUBJECTS:
        tail.extend(f"{subject} {suffix}" for suffix in SUBJECT_SUFFIXES)
        tail.extend(f"{subject} coaching in {city}" for city in CITIES)
    for dish in DISHES:
        tail.extend(f"{dish} {suffix}" for suffix in DISH_SUFFIXES)
        tail.extend(f"best {dish} in {city}" for city in CITIES)
    rng.shuffle(tail)                      # tail popularity order is arbitrary
    return HEAD_QUERIES + tail


def _make_messy(query, rng):
    """Imitate how people really type: random capitals and extra spaces."""
    roll = rng.random()
    if roll < 0.04:
        return query.upper()
    if roll < 0.10:
        return query.title()
    if roll < 0.13:
        return "  " + query.replace(" ", "  ") + " "
    return query


def generate_stream(n=DEFAULT_SIZE, seed=2026, zipf_s=1.0):
    """Return n raw queries drawn from a Zipf(s) popularity distribution."""
    rng = random.Random(seed)
    vocabulary = build_vocabulary(rng)
    weights = [1.0 / (rank ** zipf_s) for rank in range(1, len(vocabulary) + 1)]
    picks = rng.choices(vocabulary, weights=weights, k=n)
    return [_make_messy(q, rng) for q in picks]


def normalize(raw_query):
    """Lower-case and collapse whitespace so 'IPL  Live score ' == 'ipl live score'."""
    return " ".join(raw_query.lower().split())


def write_file(path=DEFAULT_PATH, n=DEFAULT_SIZE):
    stream = generate_stream(n)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(stream) + "\n")
    return len(stream)


def read_file(path=DEFAULT_PATH):
    """Yield normalised queries one at a time (streaming - file is never fully in memory)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            query = normalize(line)
            if query:
                yield query


if __name__ == "__main__":
    count = write_file()
    print(f"Wrote {count:,} queries to {DEFAULT_PATH}")
