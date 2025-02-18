# LangChain Agent: E-commerce Product Recommendation Agent (Usecase-009)

This example demonstrates how to create a LangChain agent that functions as an AI shopping assistant for e-commerce product recommendations.

## Overview

The agent uses OpenAI's language models combined with custom product recommendation algorithms to help users discover products, compare options, and understand why specific items match their needs.

## Features

- Search product catalog using natural language with filters
- Provide personalized product recommendations based on user preferences
- Compare multiple products side-by-side
- Generate natural language explanations for recommendations
- Maintain conversation context through memory
- Handle complex shopping queries in natural language

## Requirements

- Python 3.9+
- OpenAI API key (set as environment variable)
- Required packages: see `requirements.txt`

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

Run the script:
```bash
python main.py
```

The script will:
1. Create a mock product catalog if one doesn't exist
2. Initialize the recommendation agent with various tools
3. Test several example shopping queries
4. Display the agent's responses

## Available Tools

1. **SearchProducts**: Search the catalog with filters for category, brand, price range, and rating
2. **RecommendProducts**: Get personalized product recommendations based on preferences
3. **CompareProducts**: Compare multiple products side-by-side to see differences
4. **ExplainRecommendation**: Generate a natural language explanation for why a product matches user needs

## Mock Product Catalog

The agent uses a mock product catalog with randomly generated products in these categories:
- Electronics
- Clothing
- Home & Kitchen
- Books
- Sports & Outdoors

Each product includes relevant attributes for its category, pricing, ratings, and availability status.

## Recommendation Algorithm

The current implementation uses a basic keyword matching algorithm with scoring based on:
- Category matches
- Attribute/feature matches
- Price range alignment with budget terms
- Product ratings

In a production system, this would be replaced with a more sophisticated recommendation engine.

## Customization

- Replace the mock catalog with a real product database by modifying the data loading logic
- Enhance the recommendation algorithm with collaborative filtering or ML-based approaches
- Add additional tools for features like checking order status or processing returns
- Implement user profiles to improve personalization over time
- Add support for product images and rich media content

## Limitations

- Uses a simplified mock product catalog
- Basic recommendation algorithm without personalization history
- Limited to text-based interactions
- Lacks integration with actual e-commerce systems
- No support for checkout or transaction processing

## Next Steps

To make this e-commerce agent production-ready, consider:
- Integrating with a real product database and inventory system
- Implementing more sophisticated recommendation algorithms
- Adding support for user accounts and purchase history
- Creating tools for order processing and checkout
- Adding multi-modal capabilities for image recognition and visual search
- Implementing A/B testing to optimize recommendation effectiveness