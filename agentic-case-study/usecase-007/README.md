# LangChain Agent: Data Analysis Agent (Usecase-007)

This example demonstrates how to create a LangChain agent that can perform data analysis tasks on tabular data.

## Overview

The agent uses OpenAI's language model to interpret natural language requests about data and then leverages pandas, matplotlib, and seaborn to perform analysis and create visualizations.

## Features

- Load and inspect CSV datasets
- Generate comprehensive data summaries
- Analyze individual columns with appropriate statistics
- Create visualizations (histograms, scatter plots, bar charts, line charts)
- Answer natural language questions about the data
- Identify trends and patterns in the data

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
1. Create a sample sales dataset if one doesn't exist
2. Set up data analysis tools
3. Run test queries to demonstrate the agent's capabilities
4. Display responses for each query

## Available Tools

1. **GetDataSummary**: Provides a comprehensive overview of the dataset
2. **AnalyzeColumn**: Performs detailed analysis on a specific column
3. **CreateVisualization**: Generates data visualizations based on specified parameters
4. **AnswerDataQuestion**: Answers natural language questions about the data

## Visualization Types

The agent supports several visualization types:
- **Histogram**: Distribution of values in a single numerical column
- **Scatter Plot**: Relationship between two numerical columns
- **Bar Chart**: Comparison of a numerical metric across categorical groups
- **Line Chart**: Trends over time or sequence

## Customization

- Replace the sample dataset with your own CSV file by changing the `DATA_FILE` variable
- Add new visualization types by extending the `create_visualization()` function
- Enhance the question answering capabilities by expanding the `answer_data_question()` function
- Add additional tools for more advanced analysis (regression, clustering, etc.)

## Limitations

- Currently only supports CSV files
- Basic question answering without sophisticated NLP understanding
- Visualizations are saved to temporary files rather than displayed directly
- Limited to predefined analysis types
- No support for time series analysis or forecasting

## Next Steps

To enhance this data analysis agent, consider:
- Adding support for more file formats (Excel, JSON, SQL databases)
- Implementing more advanced statistical analysis
- Integrating machine learning for predictive analytics
- Creating an interactive dashboard for visualization
- Implementing dynamic file loading through a file browser interface
- Adding support for data transformation and cleaning operations