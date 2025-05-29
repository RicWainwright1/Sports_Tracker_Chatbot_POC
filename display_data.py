import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("Children_Toy_Survey_Dummy_Data.csv")

# Define color palette
custom_colors = ["#2A317F", "#004FB9", "#39A8E0"]

# Function to determine chart type and layout
def determine_chart_type_and_layout(data, column, question_code=None, question_type=None):
    answers = data[column].value_counts().reset_index()
    answers.columns = ['answer', 'count']

    count = len(answers)
    longest_length = max(answers['answer'].astype(str).apply(len))
    ordered_chart = 'answer_order' in data.columns if isinstance(data, pd.DataFrame) else False

    # Special case override
    if question_code == 'FEEL005':
        return "horizontalBar", 2

    if longest_length > 55:
        return "table", 1

    if count <= 5:
        if ordered_chart:
            return "bar", 1
        else:
            if question_type in ["single_choice", "exclusive"]:
                return "horizontalBar", 2
    elif count <= 10:
        return "horizontalBar", 2

    return "bar", 2

# Streamlit UI
st.title("Children's Toy Survey Explorer")
st.sidebar.header("Filter Options")

# User input for age and gender
age_input = st.sidebar.text_input("Enter age or age range (e.g., 4 or 2-6)")
gender = st.sidebar.selectbox("Select Gender", options=["All", "Male", "Female"])
column = st.sidebar.selectbox("Select Question (column)", options=df.columns[3:])

# Dummy metadata input (simulate source)
question_code = st.sidebar.text_input("Question Code (optional)", value="")
question_type = st.sidebar.selectbox("Question Type (optional)", options=["", "single_choice", "exclusive", "multi_choice"])

# Data filtering
if "-" in age_input:
    age_range = list(map(int, age_input.split('-')))
    df_filtered = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
elif age_input.isdigit():
    df_filtered = df[df['Age'] == int(age_input)]
else:
    df_filtered = df.copy()

if gender != "All":
    df_filtered = df_filtered[df_filtered['Gender'] == gender]

# Determine chart type
chart_type, chart_size = determine_chart_type_and_layout(df_filtered, column, question_code, question_type)

# Show result
st.subheader(f"Chart for: {column}")

if df_filtered.empty:
    st.warning("No data matches your filter criteria.")
elif chart_type == "table":
    st.dataframe(df_filtered[[column]])
else:
    value_counts = df_filtered[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']

    if chart_type == "bar":
        fig = px.bar(value_counts, x=column, y='Count', title=column, color=column, color_discrete_sequence=custom_colors)
    elif chart_type == "horizontalBar":
        fig = px.bar(value_counts, x='Count', y=column, orientation='h', title=column, color=column, color_discrete_sequence=custom_colors)
    elif chart_type == "line":
        fig = px.line(value_counts, x=column, y='Count', title=column, color_discrete_sequence=custom_colors)
    elif chart_type == "pie":
        fig = px.pie(value_counts, names=column, values='Count', title=column, color_discrete_sequence=custom_colors)

    st.plotly_chart(fig, use_container_width=True)
