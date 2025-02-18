"""
LangChain Agent: Data Analysis Agent
===============================

This module implements a LangChain agent for data analysis tasks.
The agent can:
- Load and preprocess CSV data
- Perform exploratory data analysis (EDA)
- Generate statistical summaries
- Create data visualizations
- Answer natural language questions about data

This is useful for creating assistants that can help analyze and visualize datasets.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import OpenAI

load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0)


# Create a sample dataset if needed
def create_sample_dataset():
    """Create a sample sales dataset for demonstration."""
    data = {
        "Date": pd.date_range(start="2023-01-01", periods=100, freq="D"),
        "Product": ["Product A", "Product B", "Product C", "Product D"] * 25,
        "Region": ["North", "South", "East", "West"] * 25,
        "Sales": [100 + i + (i % 20) * 10 for i in range(100)],
        "Units": [5 + i % 10 for i in range(100)],
        "Customer_Satisfaction": [3.5 + (i % 30) / 10 for i in range(100)],
    }
    df = pd.DataFrame(data)

    # Create directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save the dataset
    df.to_csv("data/sales_data.csv", index=False)
    return "data/sales_data.csv"


# Create sample dataset if it doesn't exist
if not os.path.exists("data/sales_data.csv"):
    create_sample_dataset()


# Helper functions for data analysis
def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error loading CSV: {str(e)}"


def get_data_summary(file_path):
    """Generate a summary of the dataset."""
    try:
        df = load_csv(file_path)

        # Basic information
        summary = "Dataset Summary:\n\n"
        summary += f"Number of rows: {df.shape[0]}\n"
        summary += f"Number of columns: {df.shape[1]}\n"
        summary += f"Column names: {list(df.columns)}\n\n"

        # Data types
        summary += "Data types:\n"
        for col, dtype in df.dtypes.items():
            summary += f"- {col}: {dtype}\n"
        summary += "\n"

        # Summary statistics for numerical columns
        numerical_summary = df.describe().to_string()
        summary += f"Numerical Summary Statistics:\n{numerical_summary}\n\n"

        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            summary += "Missing Values:\n"
            for col, count in missing_values.items():
                if count > 0:
                    summary += f"- {col}: {count} missing values\n"
        else:
            summary += "No missing values found.\n"

        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def analyze_column(file_path, column_name):
    """Analyze a specific column in the dataset."""
    try:
        df = load_csv(file_path)

        if column_name not in df.columns:
            return f"Column '{column_name}' not found in the dataset."

        col_data = df[column_name]
        analysis = f"Analysis of column '{column_name}':\n\n"

        # Determine data type and provide appropriate analysis
        if pd.api.types.is_numeric_dtype(col_data):
            # Numerical column analysis
            stats = col_data.describe()
            analysis += f"Data type: Numerical\n"
            analysis += f"Summary statistics:\n{stats.to_string()}\n\n"
            analysis += f"Number of unique values: {col_data.nunique()}\n"
            analysis += f"Number of missing values: {col_data.isnull().sum()}\n"

        elif pd.api.types.is_datetime64_dtype(col_data):
            # Date/time column analysis
            analysis += f"Data type: Datetime\n"
            analysis += f"Earliest date: {col_data.min()}\n"
            analysis += f"Latest date: {col_data.max()}\n"
            analysis += f"Date range: {(col_data.max() - col_data.min()).days} days\n"

        else:
            # Categorical/text column analysis
            analysis += f"Data type: Categorical/Text\n"
            analysis += f"Number of unique values: {col_data.nunique()}\n"
            analysis += f"Number of missing values: {col_data.isnull().sum()}\n"

            # Value counts for categorical data
            value_counts = col_data.value_counts().head(10)
            analysis += f"Top values:\n{value_counts.to_string()}\n"

        return analysis
    except Exception as e:
        return f"Error analyzing column: {str(e)}"


def create_visualization(file_path, viz_type, columns):
    """
    Create a visualization based on specified columns.

    Args:
        file_path: Path to the CSV file
        viz_type: Type of visualization (histogram, scatter, bar, line, etc.)
        columns: List of column names to include

    Returns:
        str: Path to the saved visualization or error message
    """
    try:
        df = load_csv(file_path)
        columns = columns.split(",")
        columns = [col.strip() for col in columns]

        # Validate columns
        for col in columns:
            if col not in df.columns:
                return f"Column '{col}' not found in the dataset."

        # Set the style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))

        # Create visualization based on type
        if viz_type.lower() == "histogram":
            if len(columns) != 1:
                return "Histogram requires exactly one numerical column."
            if not pd.api.types.is_numeric_dtype(df[columns[0]]):
                return f"Column '{columns[0]}' must be numerical for a histogram."

            sns.histplot(data=df, x=columns[0], kde=True)
            plt.title(f"Histogram of {columns[0]}")

        elif viz_type.lower() == "scatter":
            if len(columns) != 2:
                return "Scatter plot requires exactly two numerical columns."
            if not (
                pd.api.types.is_numeric_dtype(df[columns[0]])
                and pd.api.types.is_numeric_dtype(df[columns[1]])
            ):
                return "Both columns must be numerical for a scatter plot."

            sns.scatterplot(data=df, x=columns[0], y=columns[1])
            plt.title(f"Scatter Plot: {columns[0]} vs {columns[1]}")

        elif viz_type.lower() == "bar":
            if len(columns) != 2:
                return "Bar plot requires exactly two columns (one categorical, one numerical)."
            if not pd.api.types.is_numeric_dtype(df[columns[1]]):
                return f"Column '{columns[1]}' must be numerical for a bar plot."

            # Aggregate data if needed
            agg_data = df.groupby(columns[0])[columns[1]].mean().reset_index()
            sns.barplot(data=agg_data, x=columns[0], y=columns[1])
            plt.title(f"Bar Plot: Average {columns[1]} by {columns[0]}")
            plt.xticks(rotation=45)

        elif viz_type.lower() == "line":
            if len(columns) < 2:
                return "Line plot requires at least two columns (one for x-axis, one for y-axis)."

            # Assuming first column is for x-axis
            x_col = columns[0]
            for y_col in columns[1:]:
                if not pd.api.types.is_numeric_dtype(df[y_col]):
                    return f"Column '{y_col}' must be numerical for a line plot."

                plt.plot(df[x_col], df[y_col], label=y_col)

            plt.title(f"Line Plot with {x_col} on x-axis")
            plt.legend()
            plt.xticks(rotation=45)

        else:
            return f"Visualization type '{viz_type}' not supported."

        # Save the visualization
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            plt.tight_layout()
            plt.savefig(temp_file.name)
            plt.close()

            # In a real app, you might upload this to cloud storage or serve it directly
            return f"Visualization saved to {temp_file.name}"

    except Exception as e:
        return f"Error creating visualization: {str(e)}"


def answer_data_question(file_path, question):
    """
    Answer questions about the data using basic analysis.

    Args:
        file_path: Path to the CSV file
        question: Natural language question about the data

    Returns:
        str: Answer to the question
    """
    try:
        df = load_csv(file_path)

        # Basic question categories - in a real implementation, you would use
        # a more sophisticated approach with embedding-based retrieval and chaining

        if "maximum" in question.lower() or "highest" in question.lower():
            for col in df.columns:
                if col.lower() in question.lower() and pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    max_value = df[col].max()
                    max_idx = df[col].idxmax()
                    answer = f"The maximum value for {col} is {max_value}."
                    if (
                        "which" in question.lower()
                        or "when" in question.lower()
                        or "where" in question.lower()
                    ):
                        row_data = df.iloc[max_idx].to_dict()
                        answer += f" Details for this entry: {row_data}"
                    return answer

        elif "minimum" in question.lower() or "lowest" in question.lower():
            for col in df.columns:
                if col.lower() in question.lower() and pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    min_value = df[col].min()
                    min_idx = df[col].idxmin()
                    answer = f"The minimum value for {col} is {min_value}."
                    if (
                        "which" in question.lower()
                        or "when" in question.lower()
                        or "where" in question.lower()
                    ):
                        row_data = df.iloc[min_idx].to_dict()
                        answer += f" Details for this entry: {row_data}"
                    return answer

        elif "average" in question.lower() or "mean" in question.lower():
            for col in df.columns:
                if col.lower() in question.lower() and pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    mean_value = df[col].mean()
                    return f"The average (mean) value for {col} is {mean_value:.2f}."

        elif "unique" in question.lower():
            for col in df.columns:
                if col.lower() in question.lower():
                    unique_values = df[col].nunique()
                    return (
                        f"There are {unique_values} unique values in the {col} column."
                    )

        elif "correlation" in question.lower():
            for col1 in df.columns:
                if col1.lower() in question.lower() and pd.api.types.is_numeric_dtype(
                    df[col1]
                ):
                    for col2 in df.columns:
                        if (
                            col2.lower() in question.lower()
                            and pd.api.types.is_numeric_dtype(df[col2])
                            and col1 != col2
                        ):
                            corr = df[col1].corr(df[col2])
                            return f"The correlation between {col1} and {col2} is {corr:.4f}."

        elif "trend" in question.lower() or "pattern" in question.lower():
            for col in df.columns:
                if col.lower() in question.lower() and pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    # Simple trend analysis - more sophisticated in real application
                    values = df[col].values
                    n = len(values)
                    if n > 2:
                        start_avg = values[: n // 3].mean()
                        end_avg = values[-n // 3 :].mean()
                        change = end_avg - start_avg
                        pct_change = (
                            (change / start_avg) * 100
                            if start_avg != 0
                            else float("inf")
                        )

                        if change > 0:
                            return f"{col} shows an upward trend. The average increased from {start_avg:.2f} to {end_avg:.2f}, a change of {pct_change:.2f}%."
                        elif change < 0:
                            return f"{col} shows a downward trend. The average decreased from {start_avg:.2f} to {end_avg:.2f}, a change of {pct_change:.2f}%."
                        else:
                            return f"{col} shows no significant trend. The average remained around {start_avg:.2f}."

        # More sophisticated analysis would be needed for more complex questions
        return "I don't have enough information to answer that question about the data. Try asking about maximums, minimums, averages, correlations, or trends in specific columns."

    except Exception as e:
        return f"Error answering question: {str(e)}"


# Define the file path
DATA_FILE = "data/sales_data.csv"

# Create tool instances
tools = [
    Tool(
        name="GetDataSummary",
        func=lambda _: get_data_summary(DATA_FILE),
        description="Provides a summary of the dataset, including basic statistics, data types, and missing values.",
    ),
    Tool(
        name="AnalyzeColumn",
        func=lambda column_name: analyze_column(DATA_FILE, column_name),
        description="Analyzes a specific column in the dataset. Input should be a column name.",
    ),
    Tool(
        name="CreateVisualization",
        func=lambda query: create_visualization(DATA_FILE, *parse_viz_query(query)),
        description="Creates a data visualization. Input should be in the format 'type:columns' where type is one of 'histogram', 'scatter', 'bar', or 'line', and columns is a comma-separated list of column names.",
    ),
    Tool(
        name="AnswerDataQuestion",
        func=lambda question: answer_data_question(DATA_FILE, question),
        description="Answers questions about the data. Input should be a natural language question about the dataset.",
    ),
]


def parse_viz_query(query):
    """Parse a visualization query in the format 'type:columns'."""
    try:
        viz_type, columns = query.split(":", 1)
        return viz_type.strip(), columns.strip()
    except ValueError:
        return "Invalid format. Use 'type:columns'.", ""


# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# Test queries
queries = [
    "Give me a summary of the dataset",
    "Analyze the Sales column",
    "What's the correlation between Sales and Customer_Satisfaction?",
    "Create a histogram for the Sales column",
    "Create a scatter plot showing Sales vs Units",
    "What's the average Customer_Satisfaction by Region?",
    "What trends do you see in the Sales data?",
]


def main():
    print("Testing Data Analysis Agent:")
    print("-" * 50)

    for query in queries:
        print(f"\nQuestion: {query}")
        try:
            response = agent.invoke(query)
            print(f"Answer: {response['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
