"""
grader/src/data/preprocess.py

Text preprocessing pipeline for the vinyl condition grader.
Reads unified.jsonl, applies text normalization and abbreviation
expansion, detects unverified media signals, assigns train/val/test
splits using adaptive stratification, and writes output JSONL files.

Transformation order (strictly enforced):
  1. Detect unverified media signals  — on raw text
  2. Detect Generic sleeve signals    — on raw text
  3. Lowercase
  4. Normalize whitespace
  5. Strip listing promo / shipping boilerplate (markdown, brackets, regex
     templates, configured phrase chunks) — on lowercased collapsed text.
     When protected-term patterns are supplied, ``###…###``, ``[…]``, and
     promo-gated ``***…***`` spans whose inner text matches a protected
     whole-token pattern are left intact to avoid dropping real defects.
  6. Optionally strip leading catalog digit before condition words
     (``strip_stray_numeric_tokens``)
  7. Expand abbreviations             — after lowercase
  8. Verify protected terms survive   — sanity check

The original `text` field is preserved. Cleaned text is written
to a new `text_clean` field. Labels are never modified.

Usage:
    python -m grader.src.data.preprocess
    python -m grader.src.data.preprocess --dry-run
"""

import copy
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import mlflow
import yaml

from grader.src.mlflow_tracking import (
    mlflow_pipeline_step_run_ctx,
)
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Listing promo / shipping noise (shared with TF-IDF extract_texts)
# ---------------------------------------------------------------------------
_DEFAULT_PROMO_NOISE_PATTERNS: tuple[str, ...] = (
    "**all $1 & $2 items = buy 2 get 1 free !! note: price deduction will be "
    "made on the least priced items in the order post-invoice so please "
    "refrain from making payment until final subtotal is adjusted**",
    '/all sealed items are sold "one way / as is" and cannot be returned or '
    "exchanged",
    'all sealed items are sold "one way / as is" and cannot be returned or '
    "exchanged",
    "all items sent securely in a double padded mailer with the vinyl "
    "separated from the sleeve (unless sealed)",
    "if not completely satisfied send back for a full refund at our expense",
    "all the products that we sell are 100% guaranteed",
    "we have a warehouse full of new cd's, cassettes, lp's, 45's, 12'' singles",
    "that are 35 plus years old",
    "everything has been marked down 75% for another 24 hours ship up to "
    "20 records in usa for only $5!! all orders over $25 cleaned on vpi!",
    "summer sale! all vinyl marked down 20%+ unlimited $5--",
    "summer sale 10% off storewide 1 week only!",
    "packed safely, shipped promptly! lp's are shipped in custom boxes for "
    "reinforced protection",
    "customs friendly",
    "part of my personal collection",
    "warehouse back stock",
    "always shipped with domestic tracking",
    "full refund",
    "post x4 records for the same price as shipping one record",
    "100 000+ items in our shop in upminster essex (district line / m25)",
    "- 1000's more records & cds at our shop in upminster essex",
    "100 000+ items in our shop",
    "you can collect in store we buy records!",
    "cleaned in a degritter - the best ultrasonic record cleaning",
    "disc stored in anti static inner",
    "ultrasonic cleaned",
    "cleaned with ultrasonic",
    "vpi vacuumed",
    "shipped in sturdy whiplash mailer",
    "whiplash mailer",
    "international buyers message me for your shipping quote",
    "qualify for free shipping",
    "less than half of our inventory is posted here!",
    "we're a real record store near pittsburgh pa",
    "read seller terms!",
    "cds ship in cardboard!",
    "$9 flat shipping!",
    "scoop purchase limited time buy 6x12\" singles get 6x12\" singles free "
    "( cheapest free )",
    "buy this copy today",
    "uk post only",
    "uk mainland customers 1-7 lp/12\" or 30 7\" singles same p&p combine & "
    "save*",
    "* -accurate grading or refund*",
    "*buy 5 get cheapest free*",
    "jacksonville pressing",
    "japanese pressing",
    "without obi",
    "w/o obi",
    "no obi",
    "all fair offers accepted",
    "⭐",
    "$5 unlimited shipping in usa",
    "postal charges reflect quality care and services used",
    "we're marrs plectrum records official rsd real world indie shop in "
    "peterborough uk",
    "we\u2019re marrs plectrum records official rsd real world indie shop in "
    "peterborough uk",
    "all orders in uk sent first class",
    "europe/worldwide with tracking",
    "we only use quality mailers",
    "ship throughout the week",
    "we do this professionally",
    "recorded delivery",
    "still in shrink-wrap",
    "pics available upon request",
    "cheapest price",
    "accepting paypal credit",
    "pay in 3",
    "watch my cleaning process here",
    "orders over $60 ship for $6",
    "free shipping on usa orders over $30",
    "flatrate shipping rates to all 6 continents",
    "from our us-hub",
    "check my other black sabbath records and combine shipping !!!",
    "buy 12 records and get the cheapest for free",
    "you will receive 2 free records in the same style",
    "free shipping: above 145 euro in europe (eu)",
    "check out our big stock of house techno trance disco & more",
    "pick up in barcelona possible",
    "cheap worldwide shipping price",
    "regular 1-5lp is the same shipping cost (2lp count as 2) gatefold sleeves "
    "is as well",
    "*was £295 27th jun '25 reduced 8th jun '25 £282 7th aug '25 £275 22nd aug "
    "'25 £269 5th sep '25 reduced 10th oct '25*",
    "superlow shipping prices to the europe and the us",
    "label variation",
    "orders usually processed within 24-48 hours",
    "in business since 1979",
    "rsd flash sale extended to",
    "up to 90% off",
    "items over original prices",
    "the price you see is the last price only items with this promo text",
    "items under 3€",
    "items under 3 eur",
)

# Currency amount in seller promo/shipping boilerplate (already lowercased).
_MONEY_TOKEN = r"(?:[$£€]\s*\d+(?:[.,]\d+)?)"

_RE_DISCOGS_US_SHIP_TAIL = re.compile(
    r"(?:^|\s)"
    r"(?:/\s*)?"
    + _MONEY_TOKEN
    + r"\s*unlimited\s+us\s*-\s*shipping"
    r"(?:\s*/\s*|\s+)+"
    r"free\s+on\s*"
    + _MONEY_TOKEN
    + r"\s*orders\s+of\s*3\+\s*items"
    r"(?:\s*/\s*|\s+)*"
    r"read\s+seller\s+terms\s+before\s+paying",
    re.IGNORECASE,
)

_RE_UNLIMITED_USA_SHIP_BANNER = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s*shipping\s+for\s+unlimited\s+items\s+in\s+usa\s*!",
    re.IGNORECASE,
)

# ``6.50 shipping for unlimited items in usa`` without currency symbol.
_RE_DECIMAL_SHIPPING_UNLIMITED_ITEMS_USA = re.compile(
    r"(?:^|\s)\d+(?:[.,]\d+)\s*shipping\s+for\s+unlimited\s+items\s+in\s+usa\s*!*",
    re.IGNORECASE,
)

# ``30% off select items`` banner.
_RE_PERCENT_OFF_SELECT_ITEMS = re.compile(
    r"\b\d+\s*%\s*off\s+select\s+items\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# Format-only promo fragment often concatenated into seller boilerplate.
_RE_CD_LP_BRAND_NEW_FRAGMENT = re.compile(
    r"\bcd\s+lp\s+brand\s+new\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# Generic seller handling/storage boilerplate; not condition evidence.
_RE_ALL_RECORDS_PROFESSIONALLY_CLEANED = re.compile(
    r"\ball\s+records?\s+professionally\s+cleaned\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)
_RE_STORED_HIGH_QUALITY_ANTISTATIC_SLEEVES = re.compile(
    r"\bstored\s+in\s+high\s+quality\s+anti(?:-| )static\s+sleeves?\b"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)
_RE_STORED_HIGH_QUALITY_ANTISTATIC_SLEEVES_UNLESS_SEALED = re.compile(
    r"\bstored\s+in\s+high\s+quality\s+anti(?:-| )static\s+sleeves?\s+"
    r"unless\s+sealed\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``$N unlimited shipping in usa`` (amount parameterized; ``in the usa`` not matched).
_RE_MONEY_UNLIMITED_SHIPPING_IN_USA = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s*unlimited\s+shipping\s+in\s+usa\b",
    re.IGNORECASE,
)

# ``£N unlimited uk shipping`` (same money token; ``uk`` not ``us``).
_RE_MONEY_UNLIMITED_UK_SHIPPING = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s*unlimited\s+uk\s+shipping\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# BLACK STAR (U+2B50) / glowing-star emoji (optional U+FE0F) seller decoration.
_RE_BLACK_STAR_DECORATION = re.compile("\u2b50\ufe0f?")

# ``still in shrink-wrap`` variants, including common typo ``shriankwrap``.
_RE_STILL_IN_SHRINK_VARIANTS = re.compile(
    r"\bstill\s+in\s+(?:shrink|shriank)(?:(?:\s*-\s*|\s+)wrap)?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# UK bulk-shipping promo (Discogs-style; amounts parameterized).
_RE_UK_BULK_SHIP_PROMO = re.compile(
    r"\buk\s+(?:upto|up\s+to)\s+\d+\s+records\s+delivered\s+2nd\s+class\s+for\s*"
    + _MONEY_TOKEN
    + r"\s+or\s+\d+\s+1st\s+class\s+for\s*"
    + _MONEY_TOKEN
    + r"\s+free\s+uk\s+shipping\s+on\s*"
    + _MONEY_TOKEN
    + r"\s+or\s+over\s+orders?\b",
    re.IGNORECASE,
)

# Trailing seller date stamps like ``3/23]`` (month/day + bracket).
_RE_DATE_STAMP_BRACKET = re.compile(r"\b\d{1,2}/\d{1,2}\]", re.IGNORECASE)

# Discogs-style multi-record shipping promo (``post x4`` / ``post x 3``, etc.).
_RE_POST_N_RECORDS_SAME_SHIP_PRICE = re.compile(
    r"\bpost\s+x\s*\d+\s+records?\s+for\s+the\s+same\s+price\s+as\s+shipping\s+"
    r"one\s+record\b",
    re.IGNORECASE,
)

# Shop inventory brag (European-style thousands ``100 000+`` or compact digits).
# Longer ``… shop in upminster essex (…)`` must run first — a trailing ``\b``
# after ``shop`` would otherwise win before the optional location tail matches.
_RE_ITEMS_IN_OUR_SHOP_UPMINSTER_PROMO = re.compile(
    r"\b(?:\d{1,3}(?:\s+\d{3})+|\d{5,8})\s*\+?\s*items\s+in\s+our\s+shop\s+"
    r"in\s+upminster\s+essex\s*\(\s*district\s*line\s*/\s*m25\s*\)",
    re.IGNORECASE,
)
_RE_ITEMS_IN_OUR_SHOP_PROMO = re.compile(
    r"\b(?:\d{1,3}(?:\s+\d{3})+|\d{5,8})\s*\+?\s*items\s+in\s+our\s+shop\b",
    re.IGNORECASE,
)

# ``- 1000's more records & cds at our shop in upminster essex`` (leading dash).
_RE_DASH_1000S_MORE_RECORDS_UPMINSTER = re.compile(
    r"(?:^|\s)-\s*1000['\u2019]?s\s+more\s+records\s*&\s*cd(?:'|\u2019)?s\s+"
    r"at\s+our\s+shop\s+in\s+upminster\s+essex\b",
    re.IGNORECASE,
)

_RE_COLLECT_STORE_BUY_RECORDS = re.compile(
    r"\byou can collect in store we buy records[!?.]*(?=\s|$)",
    re.IGNORECASE,
)

# Degritter / ultrasonic cleaning shop boilerplate (hyphen or en/em dash).
_RE_DEGRITTER_ULTRASONIC_PROMO = re.compile(
    r"\bcleaned\s+in\s+a\s+degritter\s*[-–—]\s*the\s+best\s+ultrasonic\s+record\s+cleaning\b",
    re.IGNORECASE,
)

_RE_DEGRITTER_MK2_LISTENED_GAUGE_CONDITION = re.compile(
    r"\bcleaned\s+in\s+a\s+degritter\s+mk\.?\s*ii\s+ultrasonic\s+machine\s+"
    r"record\s+listened\s+to\s+all\s+the\s+way\s+through\s+to\s+gauge\s+condition\b",
    re.IGNORECASE,
)

# ``everything in our inventory is ultrasonically cleaned before shipment``.
_RE_INVENTORY_ULTRASONIC_CLEANED_BEFORE_SHIPMENT = re.compile(
    r"\beverything\s+in\s+our\s+inventory\s+is\s+ultrasonic(?:ally)?\s+cleaned\s+"
    r"(?:before|prior\s+to)\s+shipment\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``record has been ultrasonically cleaned`` boilerplate (non-condition process note).
_RE_RECORD_HAS_BEEN_ULTRASONICALLY_CLEANED = re.compile(
    r"\brecord\s+has\s+been\s+ultrasonic(?:ally)?\s+cleaned\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``please allow up to N week(s) before checking the status of your order``.
_RE_PLEASE_ALLOW_UP_TO_N_WEEKS_BEFORE_ORDER_STATUS = re.compile(
    r"\bplease\s+allow\s+up\s+to\s+\d+\s+weeks?\s+before\s+checking\s+the\s+status\s+"
    r"of\s+your\s+order\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_MONTH_NAME_EN = (
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)"
)

# ``everything is 10% off through december 31`` (month name + day; optional year).
_RE_EVERYTHING_IS_PCT_OFF_THROUGH_MONTH_DAY = re.compile(
    r"\beverything\s+is\s+\d+\s*%\s+off\s+through\s+"
    + _MONTH_NAME_EN
    + r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4})?(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``this includes sealed items if the condition is m/m then it is sealed``.
_RE_INCLUDES_SEALED_ITEMS_IF_MM_THEN_SEALED = re.compile(
    r"\bthis\s+includes\s+sealed\s+items\s+if\s+the\s+condition\s+is\s+"
    r"m\s*/\s*m\s+then\s+it\s+is\s+sealed\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_INTERNATIONAL_BUYERS_SHIPPING_QUOTE = re.compile(
    r"\binternational\s+buyers\s*,?\s*message\s+me\s+for\s+your\s+shipping\s+quote"
    r"[!.,]*(?=\s|$)",
    re.IGNORECASE,
)

# ``international shipping in N kg parcel(s)`` shop blurb.
_RE_INTERNATIONAL_SHIPPING_N_KG_PARCELS = re.compile(
    r"\binternational\s+shipping\s+in\s+\d+\s*kg\s+parcels?\b",
    re.IGNORECASE,
)

# ``fill up your parcel with N records / M cds`` bundle line.
_RE_FILL_UP_PARCEL_RECORDS_SLASH_CDS = re.compile(
    r"\bfill\s+up\s+your\s+parcel\s+with\s+\d+\s+records?\s*/\s*\d+\s+cd'?s?\b",
    re.IGNORECASE,
)

# ``ships in N business day(s)`` dispatch blurb.
_RE_SHIPS_IN_N_BUSINESS_DAYS = re.compile(
    r"\bships?\s+in\s+\d+\s+business\s+days?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_ALL_ORDERS_SHIP_N_BUSINESS_DAYS_LATER = re.compile(
    r"\ball\s+orders?\s+sold\s+will\s+ship\s+the\s+\d+\s*[-–]\s*\d+\s+"
    r"business\s+days?\s+later\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_YOU_PURCHASE_FRIDAY_LATEST_SHIP_TUESDAY = re.compile(
    r"\(\s*you\s+purchase\s+on\s+friday\s*,?\s*latest\s+ship\s+date\s+tuesday\s*\)"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``usps media mail`` (optional ``with tracking``).
_RE_USPS_MEDIA_MAIL = re.compile(
    r"\busps\s+media\s+mail(?:\s+with\s+tracking)?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``shipped/ships/sent with tracking`` (avoids stripping bare ``tracking`` in notes).
_RE_SHIP_VERB_WITH_TRACKING = re.compile(
    r"\b(?:shipped?|ships?|sent)\s+with\s+tracking\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``$N shipping within the u.s`` (amount parameterized).
_RE_MONEY_SHIPPING_WITHIN_US = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s+shipping\s+within\s+the\s+u\.?\s*s\.?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``$N unlimited items shipped in the u.s`` shop banner.
_RE_MONEY_UNLIMITED_ITEMS_SHIPPED_IN_US = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s+unlimited\s+items?\s+shipped\s+in\s+the\s+u\.?\s*s\.?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# Custom mailer / padding shop blurb.
_RE_CUSTOM_MAILERS_CORNER_PADDING = re.compile(
    r"\bi\s+use\s+custom\s+shipping\s+mailers\s+with\s+added\s+corner\s+"
    r"protection\s+and\s+cardboard\s+padding\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``low priced quick worldwide delivery`` shop blurb.
_RE_LOW_PRICED_QUICK_WORLDWIDE_DELIVERY = re.compile(
    r"\blow[-\s]?priced\s+quick\s+worldwide\s+delivery\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# Standalone ``low priced`` / ``worldwide delivery`` (run after the full blurb).
_RE_LOW_PRICED_STANDALONE = re.compile(
    r"\blow[-\s]?priced\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_WORLDWIDE_DELIVERY_STANDALONE = re.compile(
    r"\bworldwide\s+delivery\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_CHEAP_SAFE_SHIPPING_FROM_EURO_WORLDWIDE = re.compile(
    r"\bcheap\s*&\s*safe\s+shipping\s+from\s+\d+(?:[.,]\d+)?\s*€\s+worldwide\b"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``pick up in <city> possible`` where city is one or two words.
_RE_PICK_UP_IN_ONE_OR_TWO_WORDS_POSSIBLE = re.compile(
    r"\bpick\s+up\s+in\s+[a-z0-9][a-z0-9'&.\-]*(?:\s+[a-z0-9][a-z0-9'&.\-]*)?\s+"
    r"possible\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_PRICED_TO_MOVE = re.compile(
    r"\bpriced\s+to\s+move\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_NEED_MORE_STORAGE_SPACE = re.compile(
    r"\bneed\s+to\s+create\s+more\s+storage\s+space\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``ships quickly`` / ``ships quickly, same or next day`` dispatch blurbs.
_RE_SHIPS_QUICKLY_OPTIONAL_SAME_NEXT_DAY = re.compile(
    r"\bships?\s+quickly(?:\s*,?\s*same\s+or\s+next\s+day)?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_SAME_OR_NEXT_DAY = re.compile(
    r"\bsame\s+or\s+next\s+day\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_SECURE_VINYL_MAILER = re.compile(
    r"\bsecure\s+vinyl\s+mailers?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``pics available`` / ``pics available upon request``.
_RE_PICS_AVAILABLE = re.compile(
    r"\bpics?\s+available(?:\s+upon\s+request)?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``photos on request`` / ``photos upon request`` (not bare ``photos on`` —
# avoids ``photos on the sleeve``).
_RE_PHOTOS_ON_REQUEST = re.compile(
    r"\b(?:more\s+)?photos?\s+(?:on\s+(?:request|demand)|upon\s+request)\b"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_REACH_OUT_ANY_QUESTIONS = re.compile(
    r"\breach\s+out\s+(?:w/|with)\s+any\s+questions\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_ASK_IF_YOU_HAVE_ANY_QUESTIONS = re.compile(
    r"\bplease\s+ask\s+if\s+you\s+have\s+any\s+questions\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_CAN_SEND_PICS_IF_WANTED = re.compile(
    r"\bcan\s+send\s+pics?\s+if\s+wanted\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_ASK_BEFORE_COMMITTING_TO_PURCHASE = re.compile(
    r"\bplease\s+ask\s+before\s+committing\s+to\s+purchase\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_PLEASE_PAY_AT_CHECKOUT = re.compile(
    r"\bplease\s+pay\s+at\s+checkout\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_ORDERS_NO_PAYMENT_IN_N_HOURS_CANCELLED = re.compile(
    r"\borders?\s+with\s+no\s+payment\s+in\s+\d+\s+hours?\s+will\s+be\s+cancelled\b"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_FLAWED_STOCK_BRAND_NEW_SEALED_DISCLAIMER = re.compile(
    r"\bplease\s+note\s*,?\s*these\s+albums\s+are\s+brand\s+new\s+and\s+sealed\s*,?\s*"
    r"but\s+from\s+my\s+flawed\s+stock\s*[-–—]?\s*covers\s+may\s+have\s+"
    r"shelf\s*/\s*shipping\s+wear\s+that\s+includes\s+creases\s+and\s+corner\s+dings\s*"
    r"\(\s*the\s+records?\s+themselves\s+are\s+guaranteed\s+to\s+be\s+fine\s*\)\s*"
    r"sometimes\s+major\s*,?\s*usually\s+minor\s*"
    r"ask\s+first\s+if\s+big\s+creases\s+would\s+be\s+a\s+big\s+concern\s+for\s+you\s*"
    r"regardless\s*,?\s*save\s+a\s+little\s+money\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_I_HAVE_EXCELLENT_RATINGS = re.compile(
    r"\bi\s+have\s+excellent\s+ratings\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_WILL_SHIP_QUICKLY_AND_SECURELY = re.compile(
    r"\bwill\s+ship\s+quickly\s+and\s+securely\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_THANKS_STANDALONE = re.compile(
    r"\bthanks\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_WHITE_LABEL_PANDA_RARE_DISCLAIMER = re.compile(
    r"\bthis\s+is\s+black\s+vinyl\s*,?\s*white\s+label\s*"
    r"\(\s*not\s+the\s+white\s+vinyl\s+pressing\s+with\s+the\s+panda\s+logo\s*\)\s*"
    r"[-–—]\s*don['\u2019]?t\s+know\s+if\s+it['\u2019]?s\s+rare\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_US_ORDERS_ONLY = re.compile(
    r"\bus\s+orders\s+only\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_CONTINENTAL_US_ONLY = re.compile(
    r"\bi\s+ship\s+to\s+the\s+continental\s+u\.?\s*s\.?\s+only\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``offer for 2€ or less``-style lowball CTAs.
_RE_OFFER_FOR_EURO_OR_LESS = re.compile(
    r"\boffers?\s+for\s+\d+(?:[.,]\d+)?\s*(?:€|eur)\s+or\s+less\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``see my shipping policy for correct mail prices`` Discogs-style CTA.
_RE_SEE_MY_SHIPPING_POLICY_MAIL_PRICES = re.compile(
    r"\bsee\s+my\s+shipping\s+policy\s+for\s+correct\s+(?:mail|postal)\s+prices?\b"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``get 20% off final price`` checkout / bundle promos.
_RE_GET_PCT_OFF_FINAL_PRICE = re.compile(
    r"\bget\s+\d+\s*%\s+off\s+final\s+price\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_SHIPS_IN_PROTECTIVE_BOX = re.compile(
    r"\bships?\s+in\s+(?:a\s+)?protective\s+box(?:es)?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``ship(s) in (…) vinyl cardboard`` mailer wording (articles / ``new``).
_RE_SHIP_IN_NEW_VINYL_CARDBOARD = re.compile(
    r"\bships?\s+in\s+(?:a\s+new\s+|the\s+new\s+|(?:a|the|new)\s+)"
    r"?vinyl\s+cardboard\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_INDEPENDENT_STARTUP_WISCONSIN = re.compile(
    r"\bindependent\s+start(?:[-–]\s*)?\s*up\s+in\s+wisconsin\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``flat rate $N domestic shipping`` (money token; ``flat rate`` before amount).
_RE_FLAT_RATE_MONEY_DOMESTIC_SHIPPING = re.compile(
    r"(?:^|\s)flat\s+rate\s+"
    + _MONEY_TOKEN
    + r"\s+domestic\s+shipping\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_THANKS_SUPPORTING_SMALL_BUSINESS = re.compile(
    r"\bthanks\s+for\s+supporting\s+(?:a\s+)?small\s+business(?:es)?\b"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# Seller grading-policy blurb (Discogs-style).
_RE_GRADE_MINT_ONLY_WHEN_STILL_SEALED = re.compile(
    r"\bi\s+generally\s+only\s+grade\s+mint\s+when\s+records\s+are\s+still\s+"
    r"sealed\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``free over $N`` threshold blurbs (not bare ``over $N`` — collides with
# ``just over $80``-style condition prose).
_RE_FREE_OVER_MONEY = re.compile(
    r"\bfree\s+over\s+" + _MONEY_TOKEN + r"\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_USUALLY_SAME_DAY = re.compile(
    r"\busually\s+same\s+day\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_FREE_SAME_DAY_INTERNATIONAL_SHIPPING = re.compile(
    r"\bfree\s+same\s+day\s+international\s+shipping\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_AVERAGE_N_DASH_M_DAYS = re.compile(
    r"\baverage\s+\d+\s*[-–]\s*\d+\s+days?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_ALL_SHIPPING_INCLUDES_TRACKING = re.compile(
    r"\ball\s+shipping\s+includes\s+tracking\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_SAME_PRICE_BUY_1_OR_N_RECORDS = re.compile(
    r"\bthe\s+same\s+price\s+whether\s+you\s+buy\s+1\s+or\s+\d+\s+records?\b"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_SHIPPED_FROM_OUR_LA_STORE = re.compile(
    r"\bshipped\s+from\s+our\s+l\.?\s*a\.?\s*stores?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

_RE_QUALIFY_FREE_SHIPPING_PROMO = re.compile(
    r"\b(?:may\s+|will\s+)?qualif(?:y|ying|ies|ied)\s+for\s+free\s+shipping\b",
    re.IGNORECASE,
)

# ``orders over $X ship for $Y`` threshold shipping blurbs.
_RE_ORDERS_OVER_SHIP_FOR = re.compile(
    r"\borders\s+over\s+"
    + _MONEY_TOKEN
    + r"\s+ship\s+for\s*"
    + _MONEY_TOKEN
    + r"\b",
    re.IGNORECASE,
)

# ``free shipping on usa orders over $X`` shop banners.
_RE_FREE_SHIPPING_ON_USA_ORDERS_OVER = re.compile(
    r"\bfree\s+shipping\s+on\s+usa\s+orders\s+over\s+"
    + _MONEY_TOKEN
    + r"\b",
    re.IGNORECASE,
)

# ``buy N records and get the cheapest for free`` bundle promos.
_RE_BUY_N_RECORDS_GET_CHEAPEST_FOR_FREE = re.compile(
    r"\bbuy\s+\d+\s+records\s+and\s+get\s+the\s+cheapest\s+for\s+free\b",
    re.IGNORECASE,
)

# ``you will receive N free records in the same style`` bundle blurbs.
_RE_YOU_WILL_RECEIVE_N_FREE_RECORDS_SAME_STYLE = re.compile(
    r"\byou\s+will\s+receive\s+\d+\s+free\s+records\s+in\s+the\s+same\s+style\b",
    re.IGNORECASE,
)

# ``free shipping: above N euro in europe (eu)`` EU threshold blurbs.
_RE_FREE_SHIPPING_ABOVE_EURO_IN_EUROPE_EU = re.compile(
    r"\bfree\s+shipping:\s*above\s+\d+(?:[.,]\d+)?\s+euros?\s+in\s+europe\s*"
    r"\(\s*eu\s*\)",
    re.IGNORECASE,
)

# ``$N flat shipping!`` (amount parameterized; optional leading space / line start).
_RE_FLAT_SHIPPING_PROMO = re.compile(
    r"(?:^|\s)" + _MONEY_TOKEN + r"\s*flat\s+shipping!*",
    re.IGNORECASE,
)

# ``$N flat rate shipping on all orders`` (amount parameterized). Trailing
# ``!`` / ``.`` / ``,`` after ``orders`` is stripped when present but not required.
_RE_FLAT_RATE_SHIPPING_ALL_ORDERS = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s*flat\s+rate\s+shipping\s+on\s+all\s+orders\b(?:[!.,]+)?",
    re.IGNORECASE,
)

# ``$4 flat s&h within the US for unlimited items`` shorthand shipping blurb.
_RE_FLAT_SH_AND_H_WITHIN_US_UNLIMITED_ITEMS = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s*flat\s+s\s*(?:&|/)\s*h\s+within\s+the\s+u\.?\s*s\.?\s+for\s+"
    r"unlimited\s+items?\b(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``$N shipping in the u.s with tracking packed well and secure in a new …``
# Optional tail only for packaging nouns (avoid eating ``new corner``, etc.).
_RE_US_SHIPPING_TRACKING_PACKED_SECURE_NEW = re.compile(
    r"(?:^|\s)"
    + _MONEY_TOKEN
    + r"\s+shipping\s+in\s+the\s+u\.?\s*s\.?"
    + r"\s+with\s+tracking\s+packed\s+well\s+and\s+secure\s+in\s+a\s+new"
    + r"(?:\s+(?:mailer|boxes?|packages?|package|whiplash(?:\s+mailer)?))?"
    r"(?:\s*[!.,…]+)?",
    re.IGNORECASE,
)

# ``cd's ship …`` / ``cds ship …`` shop blurb.
_RE_CDS_SHIP_CARDBOARD_PROMO = re.compile(
    r"\bcd'?s\s+ship\s+in\s+cardboard!*",
    re.IGNORECASE,
)

_RE_READ_SELLER_TERMS_SHOUT = re.compile(
    r"\bread\s+seller\s+terms!+(?=\s|$|[,;.])",
    re.IGNORECASE,
)

_RE_REAL_RECORD_STORE_PITTSBURGH_PROMO = re.compile(
    r"\bwe['\u2019]re\s+a\s+real\s+record\s+store\s+near\s+pittsburgh\s+pa\b",
    re.IGNORECASE,
)

_RE_LESS_HALF_INVENTORY_POSTED_PROMO = re.compile(
    r"\bless\s+than\s+half\s+of\s+our\s+inventory\s+is\s+posted\s+here!*"
    r"(?=\s|$|[,;.])",
    re.IGNORECASE,
)

_RE_BUY_THIS_COPY_TODAY_PROMO = re.compile(
    r"\bbuy this copy today\b",
    re.IGNORECASE,
)

# ``check my other black sabbath records and combine shipping!!!`` shop CTA.
_RE_CHECK_OTHER_BLACK_SABBATH_COMBINE_SHIPPING = re.compile(
    r"\bcheck\s+my\s+other\s+black\s+sabbath\s+records\s+and\s+combine\s+shipping"
    r"(?:\s*!+)?(?=\s|$|[.!,?;])",
    re.IGNORECASE,
)

# Scoop ``6x12 singles`` bundle line (straight or curly inch marks before
# ``singles``).
_RE_SCOOP_6X12_BUNDLE_PROMO = re.compile(
    r"\bscoop\s+purchase\s+limited\s+time\s+buy\s+6x12\W*singles\s+get\s+6x12"
    r"\W*singles\s+free\s*\(\s*cheapest\s+free\s*\)",
    re.IGNORECASE,
)

_RE_UK_POST_ONLY_PROMO = re.compile(
    r"\buk\s+post\s+only!*(?=\s|$|[,.;])",
    re.IGNORECASE,
)

# ``uk p+p for N records`` / ``uk p&p for N records`` postage blurbs.
_RE_UK_PP_FOR_N_RECORDS = re.compile(
    r"\buk\s+p(?:\+|&)p\s+for\s+\d+\s+records?\b",
    re.IGNORECASE,
)

# ``*buy N get cheapest free*`` style promos (leading/trailing asterisks).
_RE_BUY_N_GET_CHEAPEST_FREE_PROMO = re.compile(
    r"\*+\s*buy\s+\d+\s+get\s+cheapest\s+free\s*\*+",
    re.IGNORECASE,
)

# ``*was £… 27th jun … reduced … *`` Discogs price-drop history (dates + amounts).
_RE_STAR_WAS_PRICE_DATED_REDUCTION_HISTORY = re.compile(
    r"\*+was\s+"
    + _MONEY_TOKEN
    + r"(?=[\s\S]*(?:\breduced\b|\d{1,2}(?:st|nd|rd|th)\s+[a-z]{3,9}))"
    r"[\s\S]{10,800}?"
    r"\*+",
    re.IGNORECASE,
)

# ``* -accurate grading or refund*`` shop guarantee blurbs.
_RE_ACCURATE_GRADING_OR_REFUND_PROMO = re.compile(
    r"\*+\s*-\s*accurate\s+grading\s+or\s+refund\s*\*+",
    re.IGNORECASE,
)

# UK mainland P&P combine line (``1-7 lp/12" or 30 7" singles …``).
_RE_UK_MAINLAND_COMBINE_SAVE_PROMO = re.compile(
    r"\buk\s+mainland\s+customers\s+"
    r"\d+-\d+\s+lp/\d+[\"'\u201d]?"
    r"\s+or\s+\d+\s+7[\"'\u201d]\s+singles\s+"
    r"same\s+p&p\s+combine\s*&\s*save\*+",
    re.IGNORECASE,
)

_RE_WITH_YOU_WITHIN = re.compile(
    r"\bwith\s+you\s+within\s+\d+"
    r"(?:\s*(?:day|days|working\s+days?|working\s+day))?\b",
    re.IGNORECASE,
)

_RE_PICKUP_SHOP_LINE = re.compile(
    r"(?:^|\s)\|\s*pick\s+up\s+order\s+over\s*"
    + _MONEY_TOKEN
    + r"(?:\s*\([^)]*\))?\s*welcome\s+at\s+our\s+shop\b[^.!?]*",
    re.IGNORECASE,
)

# Discogs-style shop mailer blurb (often wrapped in **…**). We do **not** strip
# arbitrary ``**…**`` spans — sellers bold real condition terms (**stain**,
# **seam split**), which would drop protected vocabulary.
_RE_STAR_MAILER_BLURB = re.compile(
    r"\*\*all items sent securely in a double padded mailer with the vinyl "
    r"separated from the sleeve \(unless sealed\)\*\*",
    re.IGNORECASE,
)

# ``**all $A & $B items = buy 2 get 1 free !! note: … post-invoice …**`` shop promo.
_RE_STAR_BUY2_GET1_LEAST_PRICED_NOTE = re.compile(
    r"\*\*all\s+"
    + _MONEY_TOKEN
    + r"\s*&\s*"
    + _MONEY_TOKEN
    + r"\s*items\s*=\s*buy\s+2\s+get\s+1\s+free\s*!+\s*"
    r"note:\s*price\s+deduction\s+will\s+be\s+made\s+on\s+the\s+least\s+priced\s+"
    r"items\s+in\s+the\s+order\s+post-invoice\s+so\s+please\s+refrain\s+from\s+"
    r"making\s+payment\s+until\s+final\s+subtotal\s+is\s+adjusted\*\*",
    re.IGNORECASE,
)

# Only remove ``***…***`` when the inner text looks like shipping/promo, not
# short condition emphasis.
_RE_TRIPLE_STAR_PROMOISH = re.compile(
    r"(?:free\s+shipping|free\s+uk\s+shipping|orders\s+over|unlimited\s+us|"
    r"marked\s+down|summer\s+sale|inside\s+eu|buy\s+\d|\d+\s*%\s*off|"
    r"discount\s+on|shipping\s+to|euro\s+sale|records\s+at)",
    re.IGNORECASE,
)

# ``### … ###`` shop banners (same non-greedy semantics as legacy single-pass sub).
_RE_HASH_SHOP_BLOCK = re.compile(r"(#{3,})([\s\S]*?)(#{3,})", re.IGNORECASE)


def protected_terms_from_grades(grades: dict[str, Any]) -> set[str]:
    """
    Collect lowercase strings from every ``*signal*`` list on each grade
    definition (same rule as protected-term harvesting for cleaning).
    """
    terms: set[str] = set()
    for grade_def in grades.values():
        if not isinstance(grade_def, dict):
            continue
        for key, value in grade_def.items():
            if "signal" not in key.lower():
                continue
            if not isinstance(value, list):
                continue
            for signal in value:
                if isinstance(signal, str):
                    terms.add(signal.lower())
    return terms


def build_protected_term_token_patterns(
    guidelines: dict[str, Any],
) -> dict[str, re.Pattern[str]]:
    """
    Build whole-token regex patterns (``\\b`` + ``re.escape``) for every
    guideline-derived protected term — same semantics as ``Preprocessor``
    uses for ``_verify_protected_terms`` / structural promo gating.
    """
    grades = guidelines.get("grades", {})
    if not isinstance(grades, dict):
        grades = {}
    terms = protected_terms_from_grades(grades)
    return {
        t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
        for t in terms
        if str(t).strip()
    }


def _inner_matches_any_protected(
    inner: str,
    patterns: dict[str, re.Pattern[str]],
) -> bool:
    return any(p.search(inner) for p in patterns.values())


def _triple_star_inner_is_promo(inner: str) -> bool:
    t = inner.strip()
    if len(t) < 14:
        return False
    return bool(_RE_TRIPLE_STAR_PROMOISH.search(t))


def load_promo_noise_patterns(pp_cfg: dict[str, Any]) -> tuple[str, ...]:
    """
    Return promo phrase substrings (lowercased), longest first, for substring
    removal. Empty or missing ``promo_noise_patterns`` falls back to defaults
    so offline tests stay aligned with production grader.yaml.
    """
    raw = pp_cfg.get("promo_noise_patterns")
    if not raw:
        seq: Sequence[str] = _DEFAULT_PROMO_NOISE_PATTERNS
    else:
        seq = raw
    phrases = [str(p).strip().lower() for p in seq if str(p).strip()]
    return tuple(sorted(phrases, key=len, reverse=True))


def strip_listing_promo_noise(
    text: str,
    promo_phrases: tuple[str, ...],
    *,
    protected_term_patterns: dict[str, re.Pattern[str]] | None = None,
) -> str:
    """
    Remove shop promo / shipping boilerplate spans. Caller must pass text
    that is already lowercased with whitespace collapsed to single spaces.

    When ``protected_term_patterns`` is provided (same whole-token patterns
    as ``Preprocessor._protected_term_token_patterns``), structural removals
    for ``###…###``, ``[…]``, and promo-gated ``***…***`` are **skipped** if
    the span's inner text matches any protected term, so real defect wording
    co-located with seller templates is not dropped.

    Arbitrary ``**…**`` markdown is **not** removed — sellers use it to bold
    real defects (e.g. **stain**, **seam split**), which would otherwise delete
    grading vocabulary. Triple-star blocks are removed only when the inner
    text matches shipping/promo heuristics. UK bulk-shipping lines,
    ``mm/dd]`` date stamps, ``post xN records for the same price as shipping
    one record`` promos, ``NNN NNN+ items in our shop`` shop brags (optional
    Upminster / district line tail), ``you can collect in store we buy
    records`` CTAs, ``- 1000's more records & cds … upminster essex`` dash promos,
    Degritter ultrasonic-cleaning blurbs,
    ``everything in our inventory is ultrasonic / ultrasonically cleaned before
    or prior to shipment`` shop lines,     ``please allow up to N week(s) before
    checking the status of your order``,
    ``everything is N% off through <month> <day>`` (optional ``st`` / ``nd`` /
    year), ``this includes sealed items if the condition is m/m then it is
    sealed``, international
    buyer shipping-quote CTAs,     ``international shipping in N kg parcel(s)``,
    ``fill up your parcel with N records / M cds``,
    ``ships in N business day(s)``,
    ``all orders sold will ship the N-M business days later``,
    ``(you purchase on friday, latest ship date tuesday)``,
    ``usps media mail`` (optional ``with tracking``),
    ``shipped/ships/sent with tracking``, ``$N shipping within the u.s``,
    ``$N unlimited item(s) shipped in the u.s``,
    custom-mailer/corner-padding blurbs, ``low priced quick worldwide delivery``,
    standalone ``low priced`` / ``low-priced`` and ``worldwide delivery``,
    ``cheap & safe shipping from N € worldwide``,
    ``pick up in <one/two words> possible``,
    ``priced to move``, ``need to create more storage space``,
    ``ships quickly`` (optional comma and ``same or next day``), standalone
    ``same or next day``,
    ``secure vinyl mailer``, ``pics available`` (optional ``upon request``),
    ``photo(s) on request`` / ``upon request`` (optional ``more``;
    ``on demand`` variant),
    ``reach out w/ or with any questions``,
    ``please ask if you have any questions``,
    ``can send pic(s) if wanted``,
    ``please ask before committing to purchase``,
    ``please pay at checkout``,
    ``orders with no payment in N hour(s) will be cancelled``,
    flawed-stock brand-new/sealed disclaimer blocks,
    ``i have excellent ratings``, ``will ship quickly and securely``,
    white-label/panda-logo rarity disclaimers,
    standalone ``thanks``, ``us orders only``,
    ``i ship to the continental us only``,
    ``offer for N€ or less`` (``eur`` spelling allowed),
    ``see my shipping policy for correct mail / postal prices``,
    ``get N% off final price`` bundle/checkout blurbs,
    ``ships in (a) protective box`` / ``boxes``, ``ship(s) in (new/a/the) vinyl cardboard``,
    ``independent start-up in wisconsin``,
    ``flat rate $N domestic shipping``, ``thanks for supporting (a) small business``,
    ``i generally only grade mint when records are still sealed``,
    ``free over $N`` (not bare ``over $N``), ``usually same day``,
    ``free same day international shipping``, ``average N - M days``,
    ``all shipping includes tracking``,
    ``the same price whether you buy 1 or N records``,
    ``shipped from our l.a / la store``,
    ``qualify for free shipping``-style lines
    (optional may/will; qualify / qualifies / qualifying / qualified),
    ``orders over $X ship for $Y`` threshold blurbs,
    ``free shipping on usa orders over $X`` banners,
    ``buy N records and get the cheapest for free`` / ``you will receive N free
    records in the same style`` / ``free shipping: above N euro in europe (eu)``
    promos, ``$N flat shipping!`` / ``$N flat rate shipping on all orders`` /
    ``$N flat s&h within the u.s for unlimited items`` /
    ``$N shipping in the u.s with tracking packed well and secure in a new``,
    ``cds ship in cardboard!``, ``read seller terms!``,
    Pittsburgh real-record-store blurbs, ``less than half … posted here``
    inventory promos, ``buy this copy today``, Black Sabbath ``check my other
    … combine shipping`` CTAs, Scoop ``6x12`` bundle blurbs,
    ``uk post only`` / ``uk p+p for N records`` shipping notes,
    ``*buy N get cheapest free*`` promos,
    ``*was £… / $… … reduced / dated price history*`` blocks,
    ``* -accurate grading or refund*`` blurbs, UK mainland ``combine & save``
    P&P lines, and ``**all $A & $B items = buy 2 get 1 free !! note: …**`` shop
    promos are also removed via dedicated regexes. ``$N unlimited shipping in
    usa`` and ``£N unlimited uk shipping`` banners, U+2B50 decorative star glyphs
    (optional emoji VS16), and
    substring phrases such as ``jacksonville pressing`` / ``all fair offers
    accepted`` follow the configured ``promo_noise_patterns`` list.
    """
    s = text.strip()

    # Gated ``*** … ***`` — promo-ish inner only; skip strip if inner matches
    # a protected whole token (mixed seller templates).
    pos = 0
    while True:
        m = re.search(r"\*\*\*([\s\S]*?)\*\*\*", s[pos:])
        if not m:
            break
        abs_start = pos + m.start()
        abs_end = pos + m.end()
        inner = m.group(1)
        if _triple_star_inner_is_promo(inner):
            if protected_term_patterns and _inner_matches_any_protected(
                inner, protected_term_patterns
            ):
                pos = abs_end
                continue
            s = s[:abs_start] + " " + s[abs_end:]
            pos = max(0, abs_start)
            continue
        pos = abs_end

    while True:
        n = _RE_STAR_MAILER_BLURB.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_STAR_BUY2_GET1_LEAST_PRICED_NOTE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    # ``### … ###`` — skip removal when inner matches a protected token.
    pos = 0
    chunks: list[str] = []
    while True:
        m = _RE_HASH_SHOP_BLOCK.search(s, pos)
        if not m:
            chunks.append(s[pos:])
            break
        chunks.append(s[pos : m.start()])
        inner = m.group(2)
        if protected_term_patterns and _inner_matches_any_protected(
            inner, protected_term_patterns
        ):
            chunks.append(m.group(0))
        else:
            chunks.append(" ")
        pos = m.end()
    s = "".join(chunks)

    # ``[ … ]`` — skip removal when inner matches a protected token.
    search_at = 0
    while True:
        m = re.search(r"\[([^\]]*)\]", s[search_at:])
        if not m:
            break
        abs_start = search_at + m.start()
        abs_end = search_at + m.end()
        inner = m.group(1)
        if protected_term_patterns and _inner_matches_any_protected(
            inner, protected_term_patterns
        ):
            search_at = abs_end
            continue
        s = s[:abs_start] + " " + s[abs_end:]
        search_at = max(0, abs_start)

    while True:
        n = _RE_DISCOGS_US_SHIP_TAIL.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_UNLIMITED_USA_SHIP_BANNER.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_DECIMAL_SHIPPING_UNLIMITED_ITEMS_USA.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PERCENT_OFF_SELECT_ITEMS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_CD_LP_BRAND_NEW_FRAGMENT.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ALL_RECORDS_PROFESSIONALLY_CLEANED.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_STORED_HIGH_QUALITY_ANTISTATIC_SLEEVES_UNLESS_SEALED.sub(
            " ", s, count=1
        )
        if n == s:
            break
        s = n

    while True:
        n = _RE_STORED_HIGH_QUALITY_ANTISTATIC_SLEEVES.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_MONEY_UNLIMITED_SHIPPING_IN_USA.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_MONEY_UNLIMITED_UK_SHIPPING.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_UK_BULK_SHIP_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PICKUP_SHOP_LINE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    s = _RE_WITH_YOU_WITHIN.sub(" ", s)

    s = _RE_DATE_STAMP_BRACKET.sub(" ", s)

    while True:
        n = _RE_POST_N_RECORDS_SAME_SHIP_PRICE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ITEMS_IN_OUR_SHOP_UPMINSTER_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ITEMS_IN_OUR_SHOP_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_DASH_1000S_MORE_RECORDS_UPMINSTER.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_COLLECT_STORE_BUY_RECORDS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_DEGRITTER_ULTRASONIC_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_DEGRITTER_MK2_LISTENED_GAUGE_CONDITION.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_INVENTORY_ULTRASONIC_CLEANED_BEFORE_SHIPMENT.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_RECORD_HAS_BEEN_ULTRASONICALLY_CLEANED.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PLEASE_ALLOW_UP_TO_N_WEEKS_BEFORE_ORDER_STATUS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_EVERYTHING_IS_PCT_OFF_THROUGH_MONTH_DAY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_INCLUDES_SEALED_ITEMS_IF_MM_THEN_SEALED.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_INTERNATIONAL_BUYERS_SHIPPING_QUOTE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_INTERNATIONAL_SHIPPING_N_KG_PARCELS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FILL_UP_PARCEL_RECORDS_SLASH_CDS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SHIPS_IN_N_BUSINESS_DAYS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ALL_ORDERS_SHIP_N_BUSINESS_DAYS_LATER.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_YOU_PURCHASE_FRIDAY_LATEST_SHIP_TUESDAY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_USPS_MEDIA_MAIL.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SHIP_VERB_WITH_TRACKING.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_MONEY_SHIPPING_WITHIN_US.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_MONEY_UNLIMITED_ITEMS_SHIPPED_IN_US.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_CUSTOM_MAILERS_CORNER_PADDING.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_LOW_PRICED_QUICK_WORLDWIDE_DELIVERY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_LOW_PRICED_STANDALONE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_WORLDWIDE_DELIVERY_STANDALONE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_CHEAP_SAFE_SHIPPING_FROM_EURO_WORLDWIDE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PICK_UP_IN_ONE_OR_TWO_WORDS_POSSIBLE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PRICED_TO_MOVE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_NEED_MORE_STORAGE_SPACE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SHIPS_QUICKLY_OPTIONAL_SAME_NEXT_DAY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SAME_OR_NEXT_DAY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SECURE_VINYL_MAILER.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PICS_AVAILABLE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PHOTOS_ON_REQUEST.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_REACH_OUT_ANY_QUESTIONS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ASK_IF_YOU_HAVE_ANY_QUESTIONS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_CAN_SEND_PICS_IF_WANTED.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ASK_BEFORE_COMMITTING_TO_PURCHASE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_PLEASE_PAY_AT_CHECKOUT.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ORDERS_NO_PAYMENT_IN_N_HOURS_CANCELLED.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FLAWED_STOCK_BRAND_NEW_SEALED_DISCLAIMER.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_I_HAVE_EXCELLENT_RATINGS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_WILL_SHIP_QUICKLY_AND_SECURELY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_THANKS_STANDALONE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_WHITE_LABEL_PANDA_RARE_DISCLAIMER.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_US_ORDERS_ONLY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_CONTINENTAL_US_ONLY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_OFFER_FOR_EURO_OR_LESS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SEE_MY_SHIPPING_POLICY_MAIL_PRICES.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_GET_PCT_OFF_FINAL_PRICE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SHIPS_IN_PROTECTIVE_BOX.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SHIP_IN_NEW_VINYL_CARDBOARD.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_INDEPENDENT_STARTUP_WISCONSIN.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FLAT_RATE_MONEY_DOMESTIC_SHIPPING.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_THANKS_SUPPORTING_SMALL_BUSINESS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_GRADE_MINT_ONLY_WHEN_STILL_SEALED.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FREE_OVER_MONEY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_USUALLY_SAME_DAY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FREE_SAME_DAY_INTERNATIONAL_SHIPPING.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_AVERAGE_N_DASH_M_DAYS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ALL_SHIPPING_INCLUDES_TRACKING.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SAME_PRICE_BUY_1_OR_N_RECORDS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SHIPPED_FROM_OUR_LA_STORE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_QUALIFY_FREE_SHIPPING_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ORDERS_OVER_SHIP_FOR.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FREE_SHIPPING_ON_USA_ORDERS_OVER.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_BUY_N_RECORDS_GET_CHEAPEST_FOR_FREE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_YOU_WILL_RECEIVE_N_FREE_RECORDS_SAME_STYLE.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FREE_SHIPPING_ABOVE_EURO_IN_EUROPE_EU.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FLAT_SHIPPING_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FLAT_RATE_SHIPPING_ALL_ORDERS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_FLAT_SH_AND_H_WITHIN_US_UNLIMITED_ITEMS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_US_SHIPPING_TRACKING_PACKED_SECURE_NEW.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_CDS_SHIP_CARDBOARD_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_READ_SELLER_TERMS_SHOUT.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_REAL_RECORD_STORE_PITTSBURGH_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_LESS_HALF_INVENTORY_POSTED_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_SCOOP_6X12_BUNDLE_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_BUY_THIS_COPY_TODAY_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_CHECK_OTHER_BLACK_SABBATH_COMBINE_SHIPPING.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_UK_POST_ONLY_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_UK_PP_FOR_N_RECORDS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_UK_MAINLAND_COMBINE_SAVE_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_ACCURATE_GRADING_OR_REFUND_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_BUY_N_GET_CHEAPEST_FREE_PROMO.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_STAR_WAS_PRICE_DATED_REDUCTION_HISTORY.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_BLACK_STAR_DECORATION.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    while True:
        n = _RE_STILL_IN_SHRINK_VARIANTS.sub(" ", s, count=1)
        if n == s:
            break
        s = n

    for phrase in promo_phrases:
        if not phrase:
            continue
        while phrase in s:
            s = s.replace(phrase, " ")

    # Collapse punctuation glue left between removed phrase chunks (commas, etc.)
    s = re.sub(r"(?:\s*[,;]\s*)+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" ,;")
    return s


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Text preprocessing pipeline for vinyl condition grader.

    Config keys read from grader.yaml:
        preprocessing.lowercase
        preprocessing.normalize_whitespace
        preprocessing.strip_stray_numeric_tokens
        preprocessing.promo_noise_patterns
        preprocessing.abbreviation_map
        preprocessing.min_text_length_discogs
        data.splits.train / val / test
        data.splits.random_seed
        paths.processed
        paths.splits
        mlflow.tracking_uri
        mlflow.experiment_name

    Config keys read from grading_guidelines.yaml:
        grades.Mint.hard_signals          — for unverified media detection
        grades.Generic.hard_signals*      — for Generic sleeve detection
            (aggregated across legacy ``hard_signals`` plus the
            strict/cosignal variants introduced in §13/§13b; see
            :func:`_collect_hard_signals`)
        grades[*].*signal* lists
            — protected terms for cleaning / gating (all keys whose names
              contain ``signal``)
    """

    def __init__(
        self,
        config_path: str,
        guidelines_path: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        else:
            self.config = self._load_yaml(config_path)
        self.guidelines = self._load_yaml(guidelines_path)

        pp_cfg = self.config["preprocessing"]
        self.do_lowercase: bool = pp_cfg.get("lowercase", True)
        self.do_normalize_whitespace: bool = pp_cfg.get(
            "normalize_whitespace", True
        )
        self.strip_stray_numeric_tokens: bool = bool(
            pp_cfg.get("strip_stray_numeric_tokens", True)
        )
        self.promo_noise_patterns: tuple[str, ...] = load_promo_noise_patterns(
            pp_cfg
        )

        # Build ordered abbreviation list — order from config is preserved.
        # Using list of tuples, not dict, to guarantee expansion order.
        # Longer/more specific patterns must come before shorter ones
        # (e.g. "vg++" before "vg+") — enforced in grader.yaml ordering.
        self.abbreviation_pairs: list[tuple[str, str]] = [
            (abbr.lower(), expansion)
            for abbr, expansion in pp_cfg.get("abbreviation_map", {}).items()
        ]

        # Replace the entire abbreviation_patterns block with:
        self.abbreviation_patterns: list[tuple[re.Pattern, str]] = []
        for abbr, expansion in self.abbreviation_pairs:
            escaped = re.escape(abbr.lower())
            if abbr.endswith("+"):
                # Prevent vg+ from matching inside vg++
                # by requiring the next char is not also +
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\+)",
                    re.IGNORECASE,
                )
            else:
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\w)",
                    re.IGNORECASE,
                )
            self.abbreviation_patterns.append((pattern, expansion))

        # Unverified media signals — from config
        self.unverified_signals: list[str] = self.config.get(
            "preprocessing", {}
        ).get(
            "unverified_media_signals",
            [
                "untested",
                "unplayed",
                "sold as seen",
                "haven't played",
                "not played",
                "unable to test",
                "no turntable",
            ],
        )

        # Generic sleeve hard signals — aggregated from every hard-signal
        # variant (legacy ``hard_signals`` plus the strict / cosignal /
        # per-target keys introduced in §13/§13b). Detection here uses
        # substring match, so tier distinctions are irrelevant; callers
        # only care whether *any* Generic hard phrase appears.
        generic_def = self.guidelines.get("grades", {}).get("Generic", {})
        self.generic_signals: list[str] = self._collect_hard_signals(generic_def)

        # Media verifiability cues — used to mark media as unverified when the
        # comment does not include any playback-related language.
        # This is intentionally conservative: we only treat "playback" cues
        # as verifiable, not cosmetic cover wording.
        self._mint_grade_def: dict[str, Any] = (
            self.guidelines.get("grades", {}).get("Mint", {}) or {}
        )
        self.mint_hard_signals: list[str] = self._collect_hard_signals(
            self._mint_grade_def
        )

        media_cue_substrings = (
            "play",
            "played",
            "plays",
            "skip",
            "skipping",
            "surface noise",
            "crackle",
            "crackling",
            "noise",
            "sound",
            "tested",
            "won't play",
            "cannot play",
            "can't play",
        )

        self.media_verifiable_cues: list[str] = []
        # Legacy signal keys (strict/cosignal hard variants are harvested
        # via ``_collect_hard_signals`` below so the §13b migration does
        # not drop Poor's playback cues from the verifiable set).
        _legacy_signal_keys = (
            "supporting_signals",
            "forbidden_signals",
            "supporting_signals_media",
            "forbidden_signals_media",
        )
        for grade_def in self.guidelines.get("grades", {}).values():
            applies_to = grade_def.get("applies_to", [])
            if "media" not in applies_to:
                continue
            candidate_signals: list[str] = list(
                self._collect_hard_signals(grade_def)
            )
            for signal_list_key in _legacy_signal_keys:
                for signal in grade_def.get(signal_list_key, []) or []:
                    if isinstance(signal, str):
                        candidate_signals.append(signal.lower())
            for s in candidate_signals:
                if any(sub in s for sub in media_cue_substrings):
                    self.media_verifiable_cues.append(s)

        # De-duplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for cue in self.media_verifiable_cues:
            if cue in seen:
                continue
            seen.add(cue)
            deduped.append(cue)
        self.media_verifiable_cues = deduped

        # Additional heuristic for comments that explicitly reference the
        # record/media object plus condition defects (not just sleeve).
        self.media_subject_terms: tuple[str, ...] = (
            "vinyl",
            "record",
            "disc",
            "lp",
            "wax",
            "pressing",
            "labels",
            "label",
        )
        self.media_condition_terms: tuple[str, ...] = (
            "mark",
            "marks",
            "scratch",
            "scratches",
            "scuff",
            "scuffs",
            "wear",
            "play wear",
            "surface",
            "dimple",
            "dimples",
            "bubble",
            "bubbling",
            "press",
            "pressed",
        )

        # Protected terms — derived from all hard_signals and
        # supporting_signals across all grades. These must survive
        # all text transformations unchanged.
        self.protected_terms: set[str] = self._build_protected_terms()
        self._protected_term_token_patterns: dict[str, re.Pattern[str]] = {
            t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
            for t in self.protected_terms
            if str(t).strip()
        }

        # Split config
        split_cfg = self.config["data"]["splits"]
        self.train_ratio: float = split_cfg["train"]
        self.val_ratio: float = split_cfg["val"]
        self.test_ratio: float = split_cfg["test"]
        self.random_seed: int = split_cfg.get("random_seed", 42)

        self._harmonization_min_samples: int = int(
            self.config.get("data", {})
            .get("harmonization", {})
            .get("min_samples_per_class", 50)
        )

        # Description adequacy (thin notes — training filter + inference hints)
        da_cfg = pp_cfg.get("description_adequacy") or {}
        self.description_adequacy_enabled: bool = bool(
            da_cfg.get("enabled", False)
        )
        self.drop_insufficient_from_training: bool = bool(
            da_cfg.get("drop_insufficient_from_training", False)
        )
        self.require_both_for_training: bool = bool(
            da_cfg.get("require_both_for_training", True)
        )
        self.min_chars_sleeve_fallback: int = int(
            da_cfg.get("min_chars_sleeve_fallback", 56)
        )
        self.prompt_sleeve: str = str(
            da_cfg.get(
                "user_prompt_sleeve",
                "Add jacket/sleeve condition details.",
            )
        ).strip()
        self.prompt_media: str = str(
            da_cfg.get(
                "user_prompt_media",
                "Describe disc/playable condition or sealed/unplayed.",
            )
        ).strip()
        configured_sleeve_terms = da_cfg.get("sleeve_evidence_terms") or []
        self.sleeve_evidence_terms: tuple[str, ...] = tuple(
            str(t).lower() for t in configured_sleeve_terms
        ) or (
            "jacket",
            "sleeve",
            "cover",
            "gatefold",
            "obi",
            "insert",
            "spine",
            "corner",
            "corners",
            "ringwear",
            "ring wear",
            "seam",
            "split",
            "crease",
            "stain",
            "shrink",
        )
        # Longer phrases first for grade-token detection on cleaned text
        self._grade_phrases: tuple[str, ...] = (
            "very good plus",
            "near mint",
            "mint minus",
            "excellent plus",
            "excellent minus",
            "very good",
            "good plus",
            "excellent",
            "good",
            "mint",
            "poor",
        )

        # Mint sleeve listings often have very short notes ("still sealed", …).
        # When enabled, treat sleeve note as adequate if sleeve_label is Mint and
        # any Mint-ish phrase matches (media label unrestricted).
        self.mint_sleeve_label_relax_sleeve_note: bool = bool(
            da_cfg.get(
                "mint_sleeve_label_relax_sleeve_note",
                da_cfg.get("mint_both_labels_relax_sleeve_note", True),
            )
        )
        _mint_relax: list[str] = list(
            self._collect_hard_signals(self._mint_grade_def)
        )
        _mint_relax_seen: set[str] = set(_mint_relax)
        for _sig in self._mint_grade_def.get("supporting_signals", []) or []:
            if isinstance(_sig, str):
                _ls = _sig.lower().strip()
                if _ls and _ls not in _mint_relax_seen:
                    _mint_relax.append(_ls)
                    _mint_relax_seen.add(_ls)
        for _sig in da_cfg.get("mint_sleeve_note_relax_extra_terms", []) or []:
            if isinstance(_sig, str):
                _ls = _sig.lower().strip()
                if _ls and _ls not in _mint_relax_seen:
                    _mint_relax.append(_ls)
                    _mint_relax_seen.add(_ls)
        for _extra in ("brand new", "like new", "new copy"):
            if _extra not in _mint_relax_seen:
                _mint_relax.append(_extra)
                _mint_relax_seen.add(_extra)
        self.mint_sleeve_relax_substrings: tuple[str, ...] = tuple(_mint_relax)

        # Paths
        processed_dir = Path(self.config["paths"]["processed"])
        splits_dir = Path(self.config["paths"]["splits"])
        self.reports_dir = Path(self.config["paths"]["reports"])
        self.input_path = processed_dir / "unified.jsonl"
        self.output_path = processed_dir / "preprocessed.jsonl"
        self.split_paths = {
            "train": splits_dir / "train.jsonl",
            "val": splits_dir / "val.jsonl",
            "test": splits_dir / "test.jsonl",
            # All inadequate-for-training rows (written when thin-note filter is on)
            "test_thin": splits_dir / "test_thin.jsonl",
        }
        splits_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # MLflow: ``run()`` uses ``mlflow_pipeline_step_run_ctx`` — configure only
        # when a nested step run is actually opened (``log_pipeline_step_runs``).

        # Stats
        self._stats: dict = {}

    # -----------------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------------
    # Protected terms
    # -----------------------------------------------------------------------
    @staticmethod
    def _collect_hard_signals(grade_def: dict[str, Any]) -> list[str]:
        """
        Aggregate every hard-signal phrase declared on a grade, across the
        legacy ``hard_signals`` list and the §13/§13b strict/cosignal
        variants (untargeted and per-target ``_sleeve`` / ``_media``
        keys). Deduplicated while preserving first-seen order.

        Used by preprocess-time detectors that care about "is any Generic
        / Mint hard phrase present?" and do not distinguish tiers.
        """
        if not isinstance(grade_def, dict):
            return []
        keys = (
            "hard_signals",
            "hard_signals_strict",
            "hard_signals_cosignal",
            "hard_signals_strict_sleeve",
            "hard_signals_strict_media",
            "hard_signals_cosignal_sleeve",
            "hard_signals_cosignal_media",
        )
        seen: set[str] = set()
        out: list[str] = []
        for key in keys:
            values = grade_def.get(key)
            if not isinstance(values, list):
                continue
            for signal in values:
                if not isinstance(signal, str):
                    continue
                s = signal.lower()
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)
        return out

    def _build_protected_terms(self) -> set[str]:
        """
        Derive protected terms from all grade signal lists in guidelines.
        These terms carry grading signal and must survive normalization.
        Stored in lowercase for case-insensitive comparison.
        """
        grades = self.guidelines.get("grades", {})
        if not isinstance(grades, dict):
            return set()
        return protected_terms_from_grades(grades)

    # -----------------------------------------------------------------------
    # Detection — must run on RAW text
    # -----------------------------------------------------------------------
    def detect_unverified_media(self, text: str) -> bool:
        """
        Returns False (unverifiable) if any unverified media signal
        is found in the raw text. Case-insensitive.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        # 1) Hard unverified signals always win.
        if any(signal in text_lower for signal in self.unverified_signals):
            return False

        # 2) Sealed/Mint exemption: sealed implies Mint media by convention
        # in this project, even if playback isn't described.
        if any(sig in text_lower for sig in self.mint_hard_signals):
            return True

        # 3) If no playback/media cue exists in the comment, mark unverified.
        if any(cue in text_lower for cue in self.media_verifiable_cues):
            return True

        # 4) Comments that mention the media object + defect language
        # (e.g., "Vinyl has light surface marks", "labels bubbling")
        # count as verifiable media descriptions.
        has_media_subject = any(
            term in text_lower for term in self.media_subject_terms
        )
        has_media_condition = any(
            term in text_lower for term in self.media_condition_terms
        )
        if has_media_subject and has_media_condition:
            return True

        # 5) Explicit play-testing language (may not appear in guideline-derived cues).
        play_markers = (
            "plays perfectly",
            "plays great",
            "plays fine",
            "plays well",
            "plays cleanly",
            "plays through",
            "tested",
            "spin tested",
        )
        if any(m in text_lower for m in play_markers):
            return True

        return False

    def detect_media_evidence_strength(self, text: str) -> str:
        """
        Estimate how directly the comment describes playable media condition.

        Returns one of:
          - "none": no usable media evidence
          - "weak": indirect/limited media evidence
          - "strong": clear playback/media-condition evidence
        """
        text_lower = text.lower()
        if any(signal in text_lower for signal in self.unverified_signals):
            return "none"

        playback_hits = sum(
            1 for cue in self.media_verifiable_cues if cue in text_lower
        )
        has_media_subject = any(
            term in text_lower for term in self.media_subject_terms
        )
        has_media_condition = any(
            term in text_lower for term in self.media_condition_terms
        )

        if playback_hits >= 2:
            return "strong"
        if playback_hits >= 1 and (has_media_subject or has_media_condition):
            return "strong"
        if has_media_subject and has_media_condition:
            return "weak"
        if any(sig in text_lower for sig in self.mint_hard_signals):
            # Sealed/unopened gives a convention-based high media label,
            # but textual evidence for actual playback condition is limited.
            return "weak"
        return "none"

    def detect_generic_sleeve(self, text: str) -> bool:
        """
        Returns True if any Generic hard signal is found in raw text.
        Used to re-confirm Generic sleeve labels from text alone.

        Note: sleeve_label may already be Generic from ingestion.
        This is a secondary text-based detection for records where
        the API condition field was ambiguous.

        Must be called before any text transformation.
        """
        text_lower = text.lower()
        return any(signal in text_lower for signal in self.generic_signals)

    def _count_distinct_grade_phrases(self, text_clean_lower: str) -> int:
        """How many canonical grade phrases appear (after abbreviation expansion)."""
        return sum(1 for phrase in self._grade_phrases if phrase in text_clean_lower)

    def sleeve_note_adequate(self, text_clean: str) -> bool:
        """
        True if the note plausibly describes jacket/sleeve/packaging,
        or uses multi-grade shorthand (e.g. NM/VG), or is long free text.
        """
        t = text_clean.strip().lower()
        if not t:
            return False
        if any(hint in t for hint in self.sleeve_evidence_terms):
            return True
        if self._count_distinct_grade_phrases(t) >= 2:
            return True
        if len(text_clean.strip()) >= self.min_chars_sleeve_fallback:
            return True
        return False

    def media_note_adequate(self, raw_text: str) -> bool:
        """True when media_evidence_strength is not ``none`` (see detect_*)."""
        return self.detect_media_evidence_strength(raw_text) != "none"

    def _mint_listing_sleeve_relaxed_ok(
        self, raw_text: str, text_clean: str
    ) -> bool:
        """Short sealed / Mint-ish copy counts as sleeve evidence."""
        blob = f"{raw_text} {text_clean}".lower()
        return any(h in blob for h in self.mint_sleeve_relax_substrings)

    def compute_description_quality(
        self,
        raw_text: str,
        text_clean: str,
        *,
        sleeve_label: str | None = None,
        media_label: str | None = None,
    ) -> dict:
        """
        Fields for training filter and inference UX.

        Returns keys: sleeve_note_adequate, media_note_adequate,
        adequate_for_training, description_quality_gaps (list[str]),
        description_quality_prompts (list[str]), needs_richer_note (bool).

        When ``mint_sleeve_label_relax_sleeve_note`` is enabled in config and
        ``sleeve_label`` is ``Mint``, sleeve adequacy also passes if the note
        contains Mint listing phrases (sealed, shrink, brand new, …).
        """
        if not self.description_adequacy_enabled:
            return {
                "sleeve_note_adequate": True,
                "media_note_adequate": True,
                "adequate_for_training": True,
                "description_quality_gaps": [],
                "description_quality_prompts": [],
                "needs_richer_note": False,
            }

        sleeve_ok = self.sleeve_note_adequate(text_clean)
        if (
            not sleeve_ok
            and self.mint_sleeve_label_relax_sleeve_note
            and str(sleeve_label or "").strip() == "Mint"
            and self._mint_listing_sleeve_relaxed_ok(raw_text, text_clean)
        ):
            sleeve_ok = True
        media_ok = self.media_note_adequate(raw_text)
        gaps: list[str] = []
        prompts: list[str] = []
        if not sleeve_ok:
            gaps.append("sleeve")
            prompts.append(self.prompt_sleeve)
        if not media_ok:
            gaps.append("media")
            prompts.append(self.prompt_media)

        if self.require_both_for_training:
            train_ok = sleeve_ok and media_ok
        else:
            train_ok = sleeve_ok or media_ok

        return {
            "sleeve_note_adequate": sleeve_ok,
            "media_note_adequate": media_ok,
            "adequate_for_training": train_ok,
            "description_quality_gaps": gaps,
            "description_quality_prompts": prompts,
            "needs_richer_note": bool(gaps),
        }

    # -----------------------------------------------------------------------
    # Text normalization
    # -----------------------------------------------------------------------
    def _lowercase(self, text: str) -> str:
        return text.lower() if self.do_lowercase else text

    def _normalize_whitespace(self, text: str) -> str:
        if not self.do_normalize_whitespace:
            return text
        # Collapse multiple whitespace characters into a single space
        # and strip leading/trailing whitespace
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _strip_leading_numeric_boilerplate(text: str) -> str:
        """
        Drop a lone leading catalog/index digit before obvious condition
        boilerplate (e.g. ``6 sealed, new hype sticker`` → ``sealed, …``).
        Does not strip counts that are part of a format description
        (``2 lp``, ``7\"``, ``disk 2 of 3``, ``6 inch``, ``3 inches``).
        """
        s = text.strip()
        m = re.match(
            r"^(\d{1,2})\s+",
            s,
            flags=re.IGNORECASE,
        )
        if not m:
            return text
        rest = s[m.end() :].lstrip()
        rest_lower = rest.lower()
        if re.match(r'^(?:\d{1,2}\s*["\u201d]|\d+\s*/\d+)', rest_lower):
            return text
        if re.match(
            r"^(?:\d+\s+lp\b|\d+\s+inch\b|\d+\s+inches\b|disk\s+\d+\s+of\b)",
            rest_lower,
        ):
            return text
        if re.match(
            r"^(?:sealed|new\s+hype|new\b|nm\b|mint\b|vg\+|vg\b|ex\b|poor\b|good\b)",
            rest_lower,
        ):
            return rest
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand abbreviations using ordered regex patterns.
        Order is preserved from grader.yaml abbreviation_map.
        Longer patterns (vg++) are applied before shorter ones (vg+)
        to prevent partial match corruption.

        When an abbreviation is immediately followed by a letter (e.g.
        ``vg+original`` with no space), append a single space so the
        expansion does not glue to the next word (``… plus original``).
        """
        for pattern, expansion in self.abbreviation_patterns:

            def _repl(m: re.Match[str], exp: str = expansion) -> str:
                s = m.string
                j = m.end()
                if j < len(s) and s[j].isalpha():
                    return f"{exp} "
                return exp

            text = pattern.sub(_repl, text)
        return text

    def _verify_protected_terms(
        self, original: str, cleaned: str
    ) -> list[str]:
        """
        Sanity check — verify that protected terms present in the
        original text as **whole tokens** still appear that way in the cleaned
        text (``\\b`` word boundaries + ``re.escape`` per term).

        Returns list of terms that were lost during transformation.
        A non-empty list indicates a preprocessing bug or an aggressive strip
        that removed a real defect token.
        """
        lost: list[str] = []
        for term, pat in self._protected_term_token_patterns.items():
            if pat.search(original) and not pat.search(cleaned):
                lost.append(term)
        return lost

    def clean_text(self, text: str) -> str:
        """
        Apply full text normalization pipeline in correct order.
        Runs on text AFTER detection steps have completed.

        Steps:
          1. Lowercase
          2. Normalize whitespace
          3. Strip listing promo / shipping boilerplate
          4. Optionally strip leading catalog digit (``strip_stray_numeric_tokens``)
          5. Expand abbreviations
        """
        cleaned = self._lowercase(text)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = strip_listing_promo_noise(
            cleaned,
            self.promo_noise_patterns,
            protected_term_patterns=self._protected_term_token_patterns,
        )
        if self.strip_stray_numeric_tokens:
            cleaned = self._strip_leading_numeric_boilerplate(cleaned)
        cleaned = self._expand_abbreviations(cleaned)
        return cleaned

    @classmethod
    def normalize_text_for_tfidf(
        cls,
        text: str,
        *,
        preprocessing_cfg: dict[str, Any],
        protected_term_patterns: dict[str, re.Pattern[str]] | None = None,
    ) -> str:
        """
        Match ``clean_text`` through promo stripping and leading-digit cleanup,
        but **omit** abbreviation expansion so TF-IDF sees the same tokens as
        ``text_clean`` from preprocess (which already expanded abbrevs).

        Pass the same ``protected_term_patterns`` as ``Preprocessor`` uses
        (from ``build_protected_term_token_patterns(guidelines)``) so TF-IDF
        skips the same gated ``###`` / ``[]`` / ``***`` spans as training.
        """
        pp = preprocessing_cfg
        s = text.strip()
        if pp.get("lowercase", True):
            s = s.lower()
        if pp.get("normalize_whitespace", True):
            s = re.sub(r"\s+", " ", s).strip()
        phrases = load_promo_noise_patterns(pp)
        s = strip_listing_promo_noise(
            s, phrases, protected_term_patterns=protected_term_patterns
        )
        if bool(pp.get("strip_stray_numeric_tokens", True)):
            s = cls._strip_leading_numeric_boilerplate(s)
        return s

    # -----------------------------------------------------------------------
    # Record processing
    # -----------------------------------------------------------------------
    def process_record(self, record: dict) -> dict:
        """
        Process a single unified record. Returns a new dict with:
          - text_clean:       normalized, expanded text
          - media_verifiable: re-detected from raw text
          - All original fields preserved unchanged

        Detection runs on original text.
        Cleaning runs after detection.
        """
        raw_text = record.get("text", "")

        # Step 1 & 2: detection on raw text
        media_verifiable = self.detect_unverified_media(raw_text)
        media_evidence_strength = self.detect_media_evidence_strength(raw_text)
        text_based_generic = self.detect_generic_sleeve(raw_text)

        # Step 3-5: text normalization
        text_clean = self.clean_text(raw_text)

        # Step 6: protected term sanity check
        lost_terms = self._verify_protected_terms(raw_text, text_clean)
        if lost_terms:
            logger.warning(
                "Protected terms lost during cleaning for item_id=%s: %s",
                record.get("item_id", "?"),
                lost_terms,
            )
            self._stats["protected_terms_lost"] += 1

        # Build output record — original fields preserved, new fields appended
        processed = {**record}
        processed["text_clean"] = text_clean
        processed["media_verifiable"] = media_verifiable
        processed["media_evidence_strength"] = media_evidence_strength

        dq = self.compute_description_quality(
            raw_text,
            text_clean,
            sleeve_label=str(record.get("sleeve_label") or ""),
            media_label=str(record.get("media_label") or ""),
        )
        processed.update(dq)

        # If text-based Generic detection fires but sleeve_label is not
        # already Generic, log for review — do not silently override label.
        # Label integrity is paramount; discrepancies are flagged, not fixed.
        if text_based_generic and record.get("sleeve_label") != "Generic":
            logger.debug(
                "Generic signal in text but sleeve_label=%r for item_id=%s. "
                "Label preserved — review may be needed.",
                record.get("sleeve_label"),
                record.get("item_id", "?"),
            )
            self._stats["generic_text_label_mismatch"] += 1

        return processed

    # -----------------------------------------------------------------------
    # Adaptive stratification
    # -----------------------------------------------------------------------
    def _compute_imbalance(self, labels: list[str]) -> float:
        """
        Compute imbalance ratio for a label list.
        Ratio = max_class_count / min_class_count.
        Higher ratio = more imbalanced.
        """
        counts = Counter(labels)
        if len(counts) < 2:
            return 1.0
        return max(counts.values()) / min(counts.values())

    def select_stratify_key(self, records: list[dict]) -> str:
        """
        Adaptively select the stratification key based on which
        target has higher class imbalance.

        Logs the decision and both imbalance ratios to MLflow.
        """
        sleeve_labels = [r["sleeve_label"] for r in records]
        media_labels = [r["media_label"] for r in records]

        sleeve_imbalance = self._compute_imbalance(sleeve_labels)
        media_imbalance = self._compute_imbalance(media_labels)

        stratify_key = (
            "sleeve_label"
            if sleeve_imbalance >= media_imbalance
            else "media_label"
        )

        logger.info(
            "Adaptive stratification — sleeve imbalance: %.2f | "
            "media imbalance: %.2f | stratifying on: %s",
            sleeve_imbalance,
            media_imbalance,
            stratify_key,
        )

        self._stats["sleeve_imbalance_ratio"] = sleeve_imbalance
        self._stats["media_imbalance_ratio"] = media_imbalance
        self._stats["stratify_key"] = stratify_key

        return stratify_key

    # -----------------------------------------------------------------------
    # Train/val/test split
    # -----------------------------------------------------------------------
    def split_records(self, records: list[dict]) -> dict[str, list[dict]]:
        """
        Assign train/val/test split to each record using adaptive
        stratified sampling.

        Strategy:
          1. Select stratification key based on imbalance ratio
          2. Attempt stratified split using StratifiedShuffleSplit
          3. If any stratum has < 2 samples, fall back to random split
             for affected records and log a warning

        Returns dict mapping split name → list of records.
        Each record has a "split" field added.
        """
        stratify_key = self.select_stratify_key(records)
        labels = [r[stratify_key] for r in records]

        # Identify strata with fewer than 2 samples — cannot be stratified
        label_counts = Counter(labels)
        too_rare = {
            label for label, count in label_counts.items() if count < 2
        }

        if too_rare:
            logger.warning(
                "Strata with < 2 samples — falling back to random split "
                "for these classes: %s",
                too_rare,
            )
            self._stats["rare_strata_fallback"] = list(too_rare)

        # Separate rare and stratifiable records
        rare_records = [r for r in records if r[stratify_key] in too_rare]
        strat_records = [r for r in records if r[stratify_key] not in too_rare]
        strat_labels = [r[stratify_key] for r in strat_records]

        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

        if strat_records:
            # First split: train vs (val + test)
            val_test_ratio = self.val_ratio + self.test_ratio
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_test_ratio,
                random_state=self.random_seed,
            )
            train_idx, val_test_idx = next(
                splitter.split(strat_records, strat_labels)
            )

            train_records = [strat_records[i] for i in train_idx]
            val_test_records = [strat_records[i] for i in val_test_idx]
            val_test_labels = [strat_labels[i] for i in val_test_idx]

            # Second split: val vs test from the val_test pool
            val_ratio_adjusted = self.val_ratio / val_test_ratio
            splitter2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=1.0 - val_ratio_adjusted,
                random_state=self.random_seed,
            )
            val_idx, test_idx = next(
                splitter2.split(val_test_records, val_test_labels)
            )

            splits["train"] = train_records
            splits["val"] = [val_test_records[i] for i in val_idx]
            splits["test"] = [val_test_records[i] for i in test_idx]

        # Distribute rare records proportionally using random assignment
        if rare_records:
            import random

            rng = random.Random(self.random_seed)
            for record in rare_records:
                split_name = rng.choices(
                    ["train", "val", "test"],
                    weights=[
                        self.train_ratio,
                        self.val_ratio,
                        self.test_ratio,
                    ],
                )[0]
                splits[split_name].append(record)

        # Tag each record with its split name
        for split_name, split_records in splits.items():
            for record in split_records:
                record["split"] = split_name

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

        return splits

    # -----------------------------------------------------------------------
    # Loading and saving
    # -----------------------------------------------------------------------
    def load_unified(self) -> list[dict]:
        """Load the unified JSONL file produced by harmonize_labels.py."""
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Unified dataset not found at {self.input_path}. "
                "Run harmonize_labels.py first."
            )
        records = []
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d records from %s", len(records), self.input_path)
        return records

    def save_preprocessed(self, records: list[dict]) -> None:
        """Write full preprocessed dataset with split field to JSONL."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Saved %d preprocessed records to %s",
            len(records),
            self.output_path,
        )

    def save_splits(self, splits: dict[str, list[dict]]) -> None:
        """Write individual train/val/test JSONL files."""
        for split_name, split_records in splits.items():
            path = self.split_paths[split_name]
            with open(path, "w", encoding="utf-8") as f:
                for record in split_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Saved %d records to %s", len(split_records), path)

    def _write_test_thin_jsonl(self, thin_records: list[dict]) -> None:
        """Eval-only split: rows not eligible for train/val/test adequacy pool."""
        path = self.split_paths["test_thin"]
        with open(path, "w", encoding="utf-8") as f:
            for record in thin_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Saved %d thin-note eval records to %s",
            len(thin_records),
            path,
        )

    def _remove_stale_test_thin_file(self) -> None:
        path = self.split_paths["test_thin"]
        if path.exists():
            path.unlink()
            logger.info(
                "Removed %s — thin-note split only when "
                "description_adequacy + drop_insufficient_from_training are on",
                path,
            )

    # -----------------------------------------------------------------------
    # Class distribution report (post-preprocess, by split)
    # -----------------------------------------------------------------------
    @staticmethod
    def _label_distribution(
        records: list[dict],
    ) -> dict[str, dict[str, int]]:
        sleeve: Counter[str] = Counter()
        media: Counter[str] = Counter()
        for record in records:
            sleeve[str(record["sleeve_label"])] += 1
            media[str(record["media_label"])] += 1
        return {"sleeve": dict(sleeve), "media": dict(media)}

    def _rare_class_warnings_for_dist(
        self,
        distribution: dict[str, dict[str, int]],
        *,
        scope: str,
    ) -> list[str]:
        warnings: list[str] = []
        threshold = self._harmonization_min_samples
        for target, grade_counts in distribution.items():
            for grade, count in grade_counts.items():
                if count < threshold:
                    warnings.append(
                        f"RARE CLASS — scope: {scope}, target: {target}, "
                        f"grade: {grade}, count: {count} "
                        f"(threshold: {threshold})"
                    )
        return warnings

    def _format_grade_table_lines(
        self,
        distribution: dict[str, dict[str, int]],
    ) -> list[str]:
        sleeve_order = self.guidelines["sleeve_grades"]
        sleeve_dist = distribution["sleeve"]
        media_dist = distribution["media"]
        lines = [
            "-" * 60,
            f"{'Grade':<20} {'Sleeve':>8} {'Media':>8}",
            "-" * 60,
        ]
        for grade in sleeve_order:
            sleeve_count = sleeve_dist.get(grade, 0)
            media_count = (
                "-" if grade == "Generic" else media_dist.get(grade, 0)
            )
            lines.append(
                f"{grade:<20} {sleeve_count:>8} {str(media_count):>8}"
            )
        sleeve_total = sum(sleeve_dist.values())
        media_total = sum(media_dist.values())
        lines += [
            "-" * 60,
            f"{'Total':<20} {sleeve_total:>8} {media_total:>8}",
            "",
        ]
        return lines

    def _format_class_distribution_splits_report(
        self,
        processed: list[dict],
        out_splits: dict[str, list[dict]],
    ) -> str:
        lines: list[str] = [
            "=" * 60,
            "VINYL GRADER — CLASS DISTRIBUTION BY SPLIT (AFTER PREPROCESS)",
            "=" * 60,
            "",
            f"Total preprocessed rows (full pool): {len(processed):>10}",
        ]
        if (
            self.description_adequacy_enabled
            and self.drop_insufficient_from_training
        ):
            eligible = self._stats.get("n_adequate_for_training", 0)
            excl = self._stats.get("n_excluded_from_splits", 0)
            lines += [
                f"Eligible for train/val/test:       {eligible:>10}",
                f"Excluded from splits (thin):     {excl:>10}",
                "",
            ]

        by_source: Counter[str] = Counter(
            str(r.get("source") or "?") for r in processed
        )
        lines.append("By source (full pool):")
        for src in sorted(by_source.keys()):
            lines.append(f"  {src + ':':<40} {by_source[src]:>10}")
        lines.append("")
        lines.append("Split sizes:")
        for name in ("train", "val", "test", "test_thin"):
            if name not in out_splits:
                continue
            lines.append(f"  {name + ':':<40} {len(out_splits[name]):>10}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("Full pool (all rows written to preprocessed.jsonl)")
        lines.append("-" * 60)
        full_dist = self._label_distribution(processed)
        lines.extend(self._format_grade_table_lines(full_dist))
        all_warnings = self._rare_class_warnings_for_dist(
            full_dist, scope="full_pool"
        )

        for split_name in ("train", "val", "test", "test_thin"):
            if split_name not in out_splits:
                continue
            rows = out_splits[split_name]
            lines.append("-" * 60)
            lines.append(f"Split: {split_name} ({len(rows)} rows)")
            lines.append("-" * 60)
            dist = self._label_distribution(rows)
            lines.extend(self._format_grade_table_lines(dist))
            all_warnings.extend(
                self._rare_class_warnings_for_dist(
                    dist, scope=f"split:{split_name}"
                )
            )

        if all_warnings:
            lines += [
                "=" * 60,
                "RARE CLASS WARNINGS",
                "=" * 60,
            ]
            for w in all_warnings:
                lines.append(f"  {w}")
            lines.append("")

        lines += [
            "=" * 60,
            "Note: Poor and Generic are expected to be rare.",
            "Rule engine owns these grades — low sample count",
            "does not prevent grading of these conditions.",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _save_class_distribution_splits_report(
        self,
        processed: list[dict],
        out_splits: dict[str, list[dict]],
    ) -> None:
        path = self.reports_dir / "class_distribution_splits.txt"
        text = self._format_class_distribution_splits_report(
            processed, out_splits
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved class distribution (splits) to %s", path)

    # -----------------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------------
    def _log_mlflow(self, splits: dict[str, list[dict]]) -> None:
        mlflow.log_params(
            {
                "lowercase": self.do_lowercase,
                "normalize_whitespace": self.do_normalize_whitespace,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "random_seed": self.random_seed,
                "stratify_key": self._stats.get("stratify_key", "unknown"),
                "n_abbreviations": len(self.abbreviation_pairs),
            }
        )
        mlflow.log_metrics(
            {
                "total_processed": self._stats["total_processed"],
                "protected_terms_lost": self._stats["protected_terms_lost"],
                "generic_text_label_mismatch": self._stats[
                    "generic_text_label_mismatch"
                ],
                "sleeve_imbalance_ratio": self._stats[
                    "sleeve_imbalance_ratio"
                ],
                "media_imbalance_ratio": self._stats["media_imbalance_ratio"],
                "n_train": len(splits["train"]),
                "n_val": len(splits["val"]),
                "n_test": len(splits["test"]),
                "n_adequate_for_training": self._stats.get(
                    "n_adequate_for_training", 0
                ),
                "n_excluded_from_splits": self._stats.get(
                    "n_excluded_from_splits", 0
                ),
                "n_test_thin": self._stats.get("n_test_thin", 0),
            }
        )

    def _save_description_adequacy_report(
        self,
        all_processed: list[dict],
        split_pool: list[dict],
    ) -> None:
        path = self.reports_dir / "description_adequacy_summary.txt"
        excl = [r for r in all_processed if not r["adequate_for_training"]]
        lines = [
            "Description adequacy (preprocessing)",
            "=" * 60,
            f"Total records:           {len(all_processed)}",
            f"Eligible for splits:     {len(split_pool)}",
            f"Excluded (thin notes):   {len(excl)}",
            "",
            "Excluded rows lack sleeve cues and/or playable-media cues "
            "(see preprocessing.description_adequacy in grader.yaml).",
            "They remain in preprocessed.jsonl with adequacy flags for audit.",
            "Eval-only split: grader/data/splits/test_thin.jsonl (same rows).",
            "",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Wrote %s", path)

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> dict[str, list[dict]]:
        """
        Full preprocessing pipeline:
          1. Load unified.jsonl
          2. Process each record (detect + clean)
          3. Adaptive stratified split (adequate rows only when thin filter on)
          4. Save preprocessed.jsonl, train/val/test, and optional test_thin.jsonl
          5. Save ``class_distribution_splits.txt`` under ``paths.reports``
          6. Log metrics to MLflow

        Args:
            dry_run: process and split but do not write files
                     or log to MLflow.

        Returns:
            Dict mapping split name → list of processed records.
        """
        self._stats = {
            "total_processed": 0,
            "protected_terms_lost": 0,
            "generic_text_label_mismatch": 0,
            "sleeve_imbalance_ratio": 0.0,
            "media_imbalance_ratio": 0.0,
            "stratify_key": "",
            "rare_strata_fallback": [],
            "n_adequate_for_training": 0,
            "n_excluded_from_splits": 0,
            "n_test_thin": 0,
        }

        with mlflow_pipeline_step_run_ctx(self.config, "preprocess") as mlf:
            records = self.load_unified()

            # Process each record
            processed: list[dict] = []
            for record in records:
                processed.append(self.process_record(record))
                self._stats["total_processed"] += 1

            logger.info(
                "Processed %d records. Protected term losses: %d. "
                "Generic/label mismatches: %d.",
                self._stats["total_processed"],
                self._stats["protected_terms_lost"],
                self._stats["generic_text_label_mismatch"],
            )

            split_pool = processed
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                split_pool = [r for r in processed if r["adequate_for_training"]]
                self._stats["n_excluded_from_splits"] = len(processed) - len(
                    split_pool
                )
                self._stats["n_adequate_for_training"] = len(split_pool)
                logger.info(
                    "Description adequacy — eligible for splits: %d | "
                    "excluded (thin notes): %d",
                    len(split_pool),
                    self._stats["n_excluded_from_splits"],
                )
                self._save_description_adequacy_report(processed, split_pool)
            else:
                self._stats["n_adequate_for_training"] = len(processed)

            if not split_pool:
                raise ValueError(
                    "No records left for train/val/test splits after "
                    "description_adequacy filtering. Relax "
                    "preprocessing.description_adequacy or set "
                    "drop_insufficient_from_training: false."
                )

            # Adaptive stratified split (training-eligible rows only when filtering)
            splits = self.split_records(split_pool)

            thin_records: list[dict] = []
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                thin_records = [
                    r for r in processed if not r["adequate_for_training"]
                ]
                for r in thin_records:
                    r["split"] = "test_thin"
                self._stats["n_test_thin"] = len(thin_records)
            else:
                self._remove_stale_test_thin_file()

            out_splits = dict(splits)
            if (
                self.description_adequacy_enabled
                and self.drop_insufficient_from_training
            ):
                out_splits["test_thin"] = thin_records

            if dry_run:
                logger.info(
                    "Dry run — skipping file writes and MLflow logging."
                )
                return out_splits

            # Save outputs
            self.save_preprocessed(processed)
            self.save_splits(splits)
            self._write_test_thin_jsonl(thin_records)
            self._save_class_distribution_splits_report(processed, out_splits)
            if mlf:
                self._log_mlflow(splits)

        return out_splits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess unified vinyl grader dataset"
    )
    parser.add_argument(
        "--config",
        default="grader/configs/grader.yaml",
        help="Path to grader config YAML",
    )
    parser.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
        help="Path to grading guidelines YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and split without writing output files",
    )
    args = parser.parse_args()

    preprocessor = Preprocessor(
        config_path=args.config,
        guidelines_path=args.guidelines,
    )
    preprocessor.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
