"""
LangChain Agent: E-commerce Product Recommendation Agent
===============================

This module implements a LangChain agent for e-commerce product recommendations.
The agent can:
- Search product catalog by various criteria
- Recommend products based on user preferences
- Compare similar products
- Generate personalized explanations for recommendations
- Handle natural language shopping queries

This demonstrates creating an AI shopping assistant that can help users find products.
"""

import json
import os
import random
import re

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize LLMs
llm = ChatOpenAI(temperature=0.7)
recommendation_llm = ChatOpenAI(temperature=0.2)  # More precise for recommendations

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Create a mock product catalog
def create_product_catalog():
    """Create a mock product catalog for demonstration purposes."""
    categories = [
        "Electronics",
        "Clothing",
        "Home & Kitchen",
        "Books",
        "Sports & Outdoors",
    ]
    brands = {
        "Electronics": [
            "TechGiant",
            "SoundMaster",
            "VisualPro",
            "SmartLife",
            "PowerTech",
        ],
        "Clothing": [
            "UrbanStyle",
            "ComfortFit",
            "LuxuryThreads",
            "ActiveWear",
            "ClassicApparel",
        ],
        "Home & Kitchen": [
            "HomeCraft",
            "KitchenWiz",
            "CozyLiving",
            "ModernHome",
            "ChefChoice",
        ],
        "Books": [
            "KnowledgePress",
            "StoryWeaver",
            "AcademicMinds",
            "FictionHouse",
            "WisdomBooks",
        ],
        "Sports & Outdoors": [
            "AthleteChoice",
            "OutdoorAdventure",
            "FitnessPro",
            "SportElite",
            "NatureGear",
        ],
    }

    products = []
    product_id = 1000

    for category in categories:
        category_brands = brands[category]

        for _ in range(random.randint(15, 25)):
            brand = random.choice(category_brands)

            # Generate different attributes based on category
            if category == "Electronics":
                name = f"{brand} {random.choice(['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch', 'Speaker', 'Camera'])}"
                attributes = {
                    "screen_size": f'{random.choice([5.5, 6.1, 6.7, 13.3, 15.6, 27])}"',
                    "storage": f"{random.choice([64, 128, 256, 512, 1024])} GB",
                    "battery_life": f"{random.randint(4, 24)} hours",
                    "color": random.choice(["Black", "Silver", "White", "Blue", "Red"]),
                    "wireless": random.choice([True, False]),
                }

            elif category == "Clothing":
                name = f"{brand} {random.choice(['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sweater', 'Shoes', 'Hat'])}"
                attributes = {
                    "size": random.choice(["XS", "S", "M", "L", "XL", "XXL"]),
                    "color": random.choice(
                        ["Black", "White", "Blue", "Red", "Green", "Yellow", "Purple"]
                    ),
                    "material": random.choice(
                        ["Cotton", "Polyester", "Wool", "Leather", "Denim"]
                    ),
                    "gender": random.choice(["Men", "Women", "Unisex"]),
                    "season": random.choice(
                        ["Summer", "Winter", "Spring", "Fall", "All Season"]
                    ),
                }

            elif category == "Home & Kitchen":
                name = f"{brand} {random.choice(['Blender', 'Coffee Maker', 'Toaster', 'Cookware Set', 'Knife Set', 'Bedding Set', 'Table Lamp'])}"
                attributes = {
                    "color": random.choice(["Black", "White", "Silver", "Red", "Blue"]),
                    "material": random.choice(
                        ["Plastic", "Metal", "Glass", "Ceramic", "Wood"]
                    ),
                    "dishwasher_safe": random.choice([True, False]),
                    "warranty": f"{random.choice([1, 2, 5, 10])} years",
                    "weight": f"{random.uniform(0.5, 15):.1f} lbs",
                }

            elif category == "Books":
                genres = [
                    "Fiction",
                    "Non-fiction",
                    "Science Fiction",
                    "Mystery",
                    "Romance",
                    "Biography",
                    "Self-help",
                    "History",
                ]
                genre = random.choice(genres)
                name = f"{random.choice(['The', 'A', ''])} {random.choice(['Great', 'Hidden', 'Lost', 'Secret', 'Ultimate', 'Complete', 'Essential'])} {random.choice(['Guide to', 'Story of', 'History of', 'Journey through', 'Exploration of', 'Handbook of', ''])} {genre}"
                attributes = {
                    "author": f"{random.choice(['John', 'Jane', 'David', 'Sarah', 'Michael', 'Emily', 'Robert', 'Lisa'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis'])}",
                    "pages": random.randint(100, 800),
                    "language": "English",
                    "format": random.choice(
                        ["Hardcover", "Paperback", "E-book", "Audiobook"]
                    ),
                    "publication_year": random.randint(1990, 2023),
                    "genre": genre,
                }

            else:  # Sports & Outdoors
                name = f"{brand} {random.choice(['Tennis Racket', 'Running Shoes', 'Yoga Mat', 'Camping Tent', 'Bicycle', 'Basketball', 'Fishing Rod'])}"
                attributes = {
                    "color": random.choice(
                        ["Black", "White", "Blue", "Red", "Green", "Yellow", "Orange"]
                    ),
                    "size": random.choice(["XS", "S", "M", "L", "XL", "One Size"]),
                    "weight": f"{random.uniform(0.2, 20):.1f} lbs",
                    "material": random.choice(
                        ["Nylon", "Polyester", "Rubber", "Metal", "Carbon Fiber"]
                    ),
                    "skill_level": random.choice(
                        [
                            "Beginner",
                            "Intermediate",
                            "Advanced",
                            "Professional",
                            "All Levels",
                        ]
                    ),
                }

            # Common product fields
            product = {
                "id": product_id,
                "name": name,
                "category": category,
                "brand": brand,
                "price": round(random.uniform(9.99, 999.99), 2),
                "rating": round(random.uniform(3.0, 5.0), 1),
                "reviews": random.randint(5, 1000),
                "in_stock": random.choice([True, False]),
                "attributes": attributes,
            }

            products.append(product)
            product_id += 1

    # Create directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save the catalog
    with open("data/product_catalog.json", "w") as f:
        json.dump(products, f, indent=2)

    return "data/product_catalog.json"


# Create product catalog if it doesn't exist
CATALOG_PATH = "data/product_catalog.json"
if not os.path.exists(CATALOG_PATH):
    create_product_catalog()

# Load product catalog
with open(CATALOG_PATH, "r") as f:
    PRODUCT_CATALOG = json.load(f)


# Create recommendation explanation prompt
recommendation_explanation_template = """
You are a helpful e-commerce shopping assistant. Your task is to explain why the following product 
would be a good match for the customer based on their preferences and needs.

Customer query: {customer_query}

Product details:
{product_details}

Write a personalized explanation of why this product is a good recommendation for the customer.
Keep your explanation conversational, helpful, and highlight the product features that match their needs.
Do not invent product features that aren't mentioned in the product details.

Explanation:
"""
recommendation_explanation_prompt = PromptTemplate(
    input_variables=["customer_query", "product_details"],
    template=recommendation_explanation_template,
)
recommendation_explanation_chain = LLMChain(
    llm=llm, prompt=recommendation_explanation_prompt
)


# Helper functions for product recommendations
def search_products(query: str) -> str:
    """
    Search for products in the catalog based on various criteria.

    Args:
        query: A string containing search terms and optional filters in the format:
              "search terms [category:X] [brand:Y] [min_price:N] [max_price:M] [min_rating:R]"

    Returns:
        str: JSON string of matching products or error message
    """
    try:
        # Extract filters from query
        filters = {}

        # Extract category filter
        category_match = re.search(r"\[category:([^\]]+)\]", query)
        if category_match:
            filters["category"] = category_match.group(1).strip()
            query = query.replace(category_match.group(0), "").strip()

        # Extract brand filter
        brand_match = re.search(r"\[brand:([^\]]+)\]", query)
        if brand_match:
            filters["brand"] = brand_match.group(1).strip()
            query = query.replace(brand_match.group(0), "").strip()

        # Extract price filters
        min_price_match = re.search(r"\[min_price:(\d+(?:\.\d+)?)\]", query)
        if min_price_match:
            filters["min_price"] = float(min_price_match.group(1))
            query = query.replace(min_price_match.group(0), "").strip()

        max_price_match = re.search(r"\[max_price:(\d+(?:\.\d+)?)\]", query)
        if max_price_match:
            filters["max_price"] = float(max_price_match.group(1))
            query = query.replace(max_price_match.group(0), "").strip()

        # Extract rating filter
        min_rating_match = re.search(r"\[min_rating:(\d+(?:\.\d+)?)\]", query)
        if min_rating_match:
            filters["min_rating"] = float(min_rating_match.group(1))
            query = query.replace(min_rating_match.group(0), "").strip()

        # The remaining text is the search terms
        search_terms = query.strip().lower()

        # Filter products
        filtered_products = PRODUCT_CATALOG.copy()

        # Apply category filter
        if "category" in filters:
            filtered_products = [
                p
                for p in filtered_products
                if filters["category"].lower() in p["category"].lower()
            ]

        # Apply brand filter
        if "brand" in filters:
            filtered_products = [
                p
                for p in filtered_products
                if filters["brand"].lower() in p["brand"].lower()
            ]

        # Apply price filters
        if "min_price" in filters:
            filtered_products = [
                p for p in filtered_products if p["price"] >= filters["min_price"]
            ]

        if "max_price" in filters:
            filtered_products = [
                p for p in filtered_products if p["price"] <= filters["max_price"]
            ]

        # Apply rating filter
        if "min_rating" in filters:
            filtered_products = [
                p for p in filtered_products if p["rating"] >= filters["min_rating"]
            ]

        # Apply search terms if provided
        if search_terms:
            filtered_products = [
                p
                for p in filtered_products
                if search_terms in p["name"].lower()
                or search_terms in p["category"].lower()
                or search_terms in p["brand"].lower()
                or any(search_terms in str(v).lower() for v in p["attributes"].values())
            ]

        # Limit results to top 5 for readability
        if len(filtered_products) > 5:
            filtered_products = filtered_products[:5]

        if not filtered_products:
            return "No products found matching your criteria."

        return json.dumps(filtered_products, indent=2)

    except Exception as e:
        return f"Error searching products: {str(e)}"


def recommend_products(criteria: str) -> str:
    """
    Recommend products based on user preferences and criteria.

    Args:
        criteria: Description of user preferences and needs

    Returns:
        str: JSON string of recommended products with explanations
    """
    try:
        # Parse the criteria to identify key preferences
        preferences = criteria.lower()

        # A simple recommendation algorithm based on matching keywords
        # (in a real system, this would be more sophisticated)
        matching_products = []

        for product in PRODUCT_CATALOG:
            # Skip out of stock items
            if not product["in_stock"]:
                continue

            score = 0
            product_text = (
                product["name"].lower()
                + " "
                + product["category"].lower()
                + " "
                + product["brand"].lower()
                + " "
                + " ".join(str(v).lower() for v in product["attributes"].values())
            )

            # Check for category matches
            for category in [
                "electronics",
                "clothing",
                "home",
                "kitchen",
                "books",
                "sports",
                "outdoors",
            ]:
                if category in preferences and category in product_text:
                    score += 5

            # Check for attribute matches
            for attribute in [
                "wireless",
                "bluetooth",
                "waterproof",
                "lightweight",
                "portable",
                "durable",
                "professional",
                "beginner",
                "advanced",
                "premium",
                "budget",
            ]:
                if attribute in preferences and attribute in product_text:
                    score += 3

            # Check for price indicators
            if (
                "cheap" in preferences
                or "affordable" in preferences
                or "budget" in preferences
            ):
                if product["price"] < 50:
                    score += 4
                elif product["price"] < 100:
                    score += 2
            elif (
                "premium" in preferences
                or "high-end" in preferences
                or "luxury" in preferences
            ):
                if product["price"] > 200:
                    score += 4
                elif product["price"] > 100:
                    score += 2

            # Boost score for highly-rated products
            if product["rating"] >= 4.5:
                score += 3
            elif product["rating"] >= 4.0:
                score += 1

            # Add product if it has a reasonable match score
            if score > 5:
                matching_products.append((product, score))

        # Sort by score (descending)
        matching_products.sort(key=lambda x: x[1], reverse=True)

        # Take top 3 products
        top_products = (
            matching_products[:3] if len(matching_products) >= 3 else matching_products
        )

        if not top_products:
            return "No products found matching your criteria."

        # Generate explanations for each recommendation
        recommendations = []
        for product, score in top_products:
            explanation = recommendation_explanation_chain.run(
                customer_query=criteria, product_details=json.dumps(product, indent=2)
            )

            recommendation = {
                "product": product,
                "explanation": explanation,
                "match_score": score,
            }
            recommendations.append(recommendation)

        return json.dumps(recommendations, indent=2)

    except Exception as e:
        return f"Error recommending products: {str(e)}"


def compare_products(product_ids: str) -> str:
    """
    Compare multiple products side by side.

    Args:
        product_ids: Comma-separated list of product IDs to compare

    Returns:
        str: Comparison table or error message
    """
    try:
        # Parse product IDs
        ids = [int(id.strip()) for id in product_ids.split(",")]

        # Find products
        products_to_compare = []
        for product_id in ids:
            product = next((p for p in PRODUCT_CATALOG if p["id"] == product_id), None)
            if product:
                products_to_compare.append(product)

        if not products_to_compare:
            return "No products found with the provided IDs."

        if len(products_to_compare) < 2:
            return "Please provide at least two product IDs to compare."

        # Create comparison table
        comparison = {
            "products": products_to_compare,
            "comparison_points": ["price", "rating", "reviews", "in_stock"],
        }

        # Add common attributes as comparison points
        # Find common attribute keys across all products
        common_attributes = set.intersection(
            *[set(p["attributes"].keys()) for p in products_to_compare]
        )
        for attr in common_attributes:
            comparison["comparison_points"].append(f"attributes.{attr}")

        return json.dumps(comparison, indent=2)

    except Exception as e:
        return f"Error comparing products: {str(e)}"


def explain_recommendation(explanation_request: str) -> str:
    """
    Generate a personalized explanation for why a product is recommended.

    Args:
        explanation_request: Format should be "product_id|user query"

    Returns:
        str: Personalized explanation
    """
    try:
        # Parse request
        parts = explanation_request.split("|", 1)
        if len(parts) != 2:
            return "Please provide both product ID and user query, separated by '|'"

        product_id_str, user_query = parts

        try:
            product_id = int(product_id_str.strip())
        except ValueError:
            return "Product ID must be a number."

        # Find product
        product = next((p for p in PRODUCT_CATALOG if p["id"] == product_id), None)
        if not product:
            return f"No product found with ID {product_id}."

        # Generate explanation
        explanation = recommendation_explanation_chain.run(
            customer_query=user_query, product_details=json.dumps(product, indent=2)
        )

        return explanation

    except Exception as e:
        return f"Error generating explanation: {str(e)}"


# Create tool instances
tools = [
    Tool(
        name="SearchProducts",
        func=search_products,
        description="Search for products in the catalog. You can include filters in the format [category:X], [brand:Y], [min_price:N], [max_price:M], and [min_rating:R]. Example: 'wireless headphones [category:Electronics] [min_price:50] [max_price:200]'",
    ),
    Tool(
        name="RecommendProducts",
        func=recommend_products,
        description="Recommend products based on user preferences and needs. Input should be a detailed description of what the user is looking for, including preferences for category, price range, features, etc.",
    ),
    Tool(
        name="CompareProducts",
        func=compare_products,
        description="Compare multiple products side by side. Input should be a comma-separated list of product IDs. Example: '1001,1005,1010'",
    ),
    Tool(
        name="ExplainRecommendation",
        func=explain_recommendation,
        description="Generate a personalized explanation for why a specific product is recommended for the user. Input format: 'product_id|user query'. Example: '1005|I need a durable laptop for gaming'",
    ),
]

# Initialize agent with memory
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)

# Test queries
queries = [
    "Can you recommend a good pair of wireless headphones?",
    "I need a new laptop for video editing, my budget is around $1200",
    "Show me some camping equipment that's durable and lightweight",
    "Compare these products for me: 1005, 1020, 1035",
    "Why would product 1010 be good for someone who enjoys outdoor photography?",
]


def main():
    print("Testing E-commerce Product Recommendation Agent:")
    print("-" * 50)

    for query in queries:
        print(f"\nCustomer: {query}")
        try:
            response = agent.invoke({"input": query})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
