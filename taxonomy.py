"""The 'client' category structure: a fashion & lifestyle retailer that groups products its own way.
Ground truth = Flipkart's category tree mapped to these client categories. The tree is NEVER a model input."""
import re
L1_MAP = {
    "Footwear": "Footwear", "Jewellery": "Jewellery",
    "Watches": "Watches & Eyewear", "Sunglasses": "Watches & Eyewear", "Eyewear": "Watches & Eyewear",
    "Bags, Wallets & Belts": "Bags & Accessories",
    "Mobiles & Accessories": "Electronics", "Computers": "Electronics", "Cameras & Accessories": "Electronics",
    "Gaming": "Electronics", "Home Entertainment": "Electronics", "Health & Personal Care Appliances": "Electronics",
    "Home Decor & Festive Needs": "Home Decor & Furnishing", "Home Furnishing": "Home Decor & Furnishing",
    "Furniture": "Home Decor & Furnishing",
    "Kitchen & Dining": "Kitchen & Dining", "Home & Kitchen": "Kitchen & Dining",
    "Beauty and Personal Care": "Beauty & Personal Care",
    "Baby Care": "Baby & Toys", "Toys & School Supplies": "Baby & Toys",
    "Automotive": "Automotive", "Pens & Stationery": "Stationery",
    "Tools & Hardware": "Other", "Home Improvement": "Other", "Sports & Fitness": "Other",
    "Pet Supplies": "Other", "eBooks": "Other",
}
CLOTHING_L2 = {"Men's Clothing": "Men's Apparel", "Women's Clothing": "Women's Apparel", "Kids' Clothing": "Kids' Apparel"}

def parse_tree(s):
    parts = s.strip('[]"').split(" >> ")
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")

def client_label(tree):
    l1, l2 = parse_tree(tree)
    if l1 == "Clothing": return CLOTHING_L2.get(l2)      # other clothing (e.g. fabric) -> unlabeled
    return L1_MAP.get(l1)                                  # unknown / broken tree -> None (excluded, reported)

# High-precision rules written from domain knowledge before looking at test results.
APPAREL = r"(t-shirt|shirt|kurta|kurti|jeans|trousers?|shorts|top|dress|saree|sari|leggings?|track pants|sweatshirt|jacket|bra|briefs?|trunks?|nightsuit|pyjama|capri|skirt|blazer|vest|night dress|salwar|dupatta)"
RULES = [  # (name, regex on lowercased product name, client category)
    ("watch", r"\bwatch(es)?\b(?!.*\b(strap|charger|box)\b)", "Watches & Eyewear"),
    ("sunglasses", r"\bsunglass(es)?\b", "Watches & Eyewear"),
    ("jewellery", r"\b(bangles?|necklaces?|earrings?|pendants?|anklets?|mangalsutra|jewellery set|nose pin|bracelets?)\b", "Jewellery"),
    ("footwear", r"\b(sandals?|slippers?|flip flops?|sneakers?|shoes|bellies|loafers?|boots|heels|moccasins?|wedges)\b(?!.*\b(rack|polish|cleaner|stand)\b)", "Footwear"),
    ("mobile acc", r"\b(back cover|screen guard|tempered glass|power bank|mobile)\b", "Electronics"),
    ("mens apparel", r"\bmen'?s\b.*\b" + APPAREL + r"\b", "Men's Apparel"),
    ("womens apparel", r"\bwomen'?s\b.*\b" + APPAREL + r"\b", "Women's Apparel"),
    ("kids apparel", r"\b(boy'?s|girl'?s|baby boy'?s|baby girl'?s)\b.*\b" + APPAREL + r"\b", "Kids' Apparel"),
    ("stationery", r"\b(pens?|notebook|diary|pencils?)\b", "Stationery"),
    ("kitchen", r"\b(mug|lunch box|pressure cooker|kadhai|cookware|dinner set|tumbler|casserole)\b", "Kitchen & Dining"),
    ("furnishing", r"\b(bedsheet|curtains?|cushion covers?|pillow covers?|blankets?|bath towel|door mat)\b", "Home Decor & Furnishing"),
]
COMPILED = [(n, re.compile(p), c) for n, p, c in RULES]

def apply_rules(name):
    """Return (category, rule) for the FIRST matching rule, but only if all matching rules agree."""
    hits = [(c, n) for n, p, c in COMPILED if p.search(name.lower())]
    if not hits: return None, None
    cats = {c for c, _ in hits}
    if len(cats) > 1: return None, "conflict"        # rules disagree -> let the model / reviewer decide
    return hits[0]
