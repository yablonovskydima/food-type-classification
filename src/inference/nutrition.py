"""Illustrative food references; these are not nutrition predictions from an image.

USDA FoodData Central SR Legacy values per 100 g, verified via its API on 2026-11-17.
Nutrient numbers: 208 (kcal), 203 (protein), 205 (carbohydrate), 204 (fat).
Each broad Food-11 category maps to one explicitly named example, not a class average.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NutritionReference:
    example: str
    fdc_id: int
    kcal: float
    protein_g: float
    carbohydrate_g: float
    fat_g: float
    culinary_note: str

    @property
    def source_url(self) -> str:
        return f"https://fdc.nal.usda.gov/food-details/{self.fdc_id}/nutrients"


NUTRITION_REFERENCES = {
    "bread": NutritionReference(
        "White bread, commercially prepared",
        174924,
        266,
        8.85,
        49.4,
        3.33,
        "Bread includes many recipes; fillings, spreads and toppings change its composition.",
    ),
    "dairy_product": NutritionReference(
        "Whole milk, 3.25% fat, with vitamin D",
        171265,
        61,
        3.15,
        4.8,
        3.25,
        "Milk, cheese and yogurt share this category but have very different compositions.",
    ),
    "dessert": NutritionReference(
        "Vanilla ice cream",
        167575,
        207,
        3.5,
        23.6,
        11,
        "Desserts vary with their base, sugar, cream and toppings.",
    ),
    "egg": NutritionReference(
        "Whole egg, hard-boiled",
        173424,
        155,
        12.6,
        1.12,
        10.6,
        "This example is boiled; frying oil, butter and fillings change the dish.",
    ),
    "fried_food": NutritionReference(
        "French fries in vegetable oil",
        170698,
        312,
        3.43,
        41.4,
        14.7,
        "The base ingredient, batter and absorbed oil vary between fried dishes.",
    ),
    "meat": NutritionReference(
        "Chicken breast, meat only, roasted",
        171477,
        165,
        31,
        0,
        3.57,
        "The animal, cut, skin, cooking method and sauce affect the finished dish.",
    ),
    "noodles_pasta": NutritionReference(
        "Pasta, cooked, unenriched, without added salt",
        168928,
        158,
        5.8,
        30.9,
        0.93,
        "This is cooked plain pasta; sauces, cheese and other additions are excluded.",
    ),
    "rice": NutritionReference(
        "White long-grain rice, enriched, cooked",
        168878,
        130,
        2.69,
        28.2,
        0.28,
        "This is cooked plain rice; oil, sauces and other additions are excluded.",
    ),
    "seafood": NutritionReference(
        "Atlantic farmed salmon, cooked with dry heat",
        175168,
        206,
        22.1,
        0,
        12.4,
        "Fish and shellfish differ by species; batter, oil and sauces also matter.",
    ),
    "soup": NutritionReference(
        "Canned tomato soup, prepared with equal volume water",
        171176,
        32,
        0.71,
        7.45,
        0.21,
        "Broth, cream, dilution and solid ingredients vary widely between soups.",
    ),
    "vegetable_fruit": NutritionReference(
        "Raw apple with skin",
        171688,
        52,
        0.26,
        13.8,
        0.17,
        "This category covers many fruits and vegetables; dressings and preparation vary.",
    ),
}
