"""Controlled topic taxonomy and a lexical topic classifier.

Free-text interests are matched with embeddings, but a small shared taxonomy gives the
router a second, explainable signal ("matched on: festive_season, discount_offers") and
an inverted index that is cheap to query at any scale.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

TOPICS: Dict[str, dict] = {
    "payments": {
        "label": "Payments & UPI",
        "terms": ["upi", "payment", "payments", "pay", "paying", "paid", "wallet", "checkout",
                  "bnpl", "pay later", "qr code", "autopay", "transaction", "transactions"],
    },
    "cards": {
        "label": "Credit & debit cards",
        "terms": ["credit card", "credit cards", "debit card", "debit cards", "card", "cards",
                  "cardholders", "card holders", "rupay", "visa", "mastercard", "reward points",
                  "no-cost emi", "no cost emi"],
    },
    "banking_lending": {
        "label": "Banking & lending",
        "terms": ["loan", "loans", "emi", "emis", "lending", "interest rate", "interest rates",
                  "fixed deposit", "fd", "savings account", "credit line", "bank", "banks",
                  "personal loan", "home loan", "auto loan", "vehicle loan", "financing", "finance"],
    },
    "discount_offers": {
        "label": "Discounts, cashback & offers",
        "terms": ["discount", "discounts", "cashback", "offer", "offers", "deal", "deals", "coupon",
                  "coupons", "promo", "promo code", "promotion", "promotions", "sale", "flat off", "free delivery",
                  "price drop", "bonus", "rewards"],
        "patterns": [r"\d+\s*%\s*(off|discount|cashback)", r"(₹|rs\.?|inr)\s?\d[\d,]*\s*(off|cashback)",
                     r"\boff on\b", r"\boff all\b"],
    },
    "festive_season": {
        "label": "Festive season",
        "terms": ["diwali", "deepavali", "dussehra", "navratri", "holi", "eid", "christmas",
                  "new year", "pongal", "onam", "raksha bandhan", "durga puja", "ganesh chaturthi",
                  "festive", "festival", "festivals", "big billion days", "great indian festival",
                  "dhanteras"],
    },
    "food_delivery": {
        "label": "Food & grocery delivery",
        "terms": ["food delivery", "food", "restaurant", "restaurants", "meal", "meals", "dining",
                  "grocery", "groceries", "quick commerce", "instamart", "dineout", "cloud kitchen",
                  "swiggy", "zomato", "blinkit"],
    },
    "ecommerce": {
        "label": "E-commerce",
        "terms": ["e-commerce", "ecommerce", "online shopping", "marketplace", "sellers", "seller",
                  "cart", "flipkart", "amazon", "myntra", "shoppers", "fashion", "apparel"],
    },
    "electronics": {
        "label": "Consumer electronics",
        "terms": ["smartphone", "smartphones", "phone", "phones", "laptop", "laptops", "tv",
                  "television", "electronics", "gadget", "gadgets", "appliance", "appliances",
                  "headphones", "iphone"],
    },
    "automotive": {
        "label": "Automotive & mobility",
        "terms": ["car", "cars", "bike", "bikes", "motorcycle", "motorcycles", "scooter", "scooters",
                  "two-wheeler", "two-wheelers", "two wheeler", "vehicle", "vehicles", "ev",
                  "electric vehicle", "dealership", "dealerships", "test ride", "pulsar", "chetak",
                  "showroom", "automobile", "fuel", "petrol"],
    },
    "travel": {
        "label": "Travel & hospitality",
        "terms": ["flight", "flights", "hotel", "hotels", "travel", "holiday", "holidays", "trip",
                  "trips", "airline", "airlines", "vacation", "bus tickets", "train tickets", "visa on arrival"],
    },
    "insurance": {
        "label": "Insurance",
        "terms": ["insurance", "insure", "insurance policy", "policyholders", "premium", "premiums", "claim",
                  "claims", "health cover", "motor insurance", "travel insurance"],
    },
    "markets_macro": {
        "label": "Markets & macro updates",
        "terms": ["repo rate", "inflation", "gdp", "rbi", "monetary policy", "sensex", "nifty",
                  "stock market", "bond yields", "rupee", "economy", "consumer spending",
                  "market update", "quarterly results", "earnings"],
    },
    "regulation": {
        "label": "Regulation & compliance",
        "terms": ["regulation", "regulations", "regulatory", "compliance", "kyc", "guidelines",
                  "circular", "npci", "sebi", "irdai", "mandate", "data protection", "dpdp"],
    },
    "logistics": {
        "label": "Logistics & supply chain",
        "terms": ["logistics", "shipping", "shipment", "shipments", "warehouse", "warehousing",
                  "courier", "last-mile", "last mile", "fulfilment", "fulfillment", "supply chain",
                  "pin codes", "pincodes"],
    },
    "partnerships": {
        "label": "Partnerships & co-marketing",
        "terms": ["partnership", "partnerships", "partner", "partners", "co-branded", "tie-up",
                  "tie up", "collaboration", "co-marketing", "alliance", "merchant", "merchants",
                  "brands"],
    },
}


def topic_keys() -> List[str]:
    return list(TOPICS)


def validate_topics(topics: Iterable[str]) -> List[str]:
    unknown = [t for t in topics if t not in TOPICS]
    if unknown:
        raise ValueError(f"unknown topics: {', '.join(unknown)}; see GET /v1/topics")
    return list(topics)


class TopicClassifier:
    """Counts whole-word term hits per topic; score = 1 - exp(-hits) (1 hit = 0.63)."""

    def __init__(self, topics: Dict[str, dict] = TOPICS):
        self._compiled = {}
        for key, spec in topics.items():
            terms = sorted((re.escape(t) for t in spec["terms"]), key=len, reverse=True)
            pats = [re.compile(r"(?<![a-z0-9])(?:" + "|".join(terms) + r")(?![a-z0-9])")]
            pats += [re.compile(p) for p in spec.get("patterns", [])]
            self._compiled[key] = pats

    def classify(self, text: str) -> Dict[str, float]:
        lowered = text.lower()
        out: Dict[str, float] = {}
        for key, pats in self._compiled.items():
            hits = sum(len(p.findall(lowered)) for p in pats)
            if hits:
                out[key] = round(1.0 - math.exp(-hits), 3)
        return out
