import streamlit as st

# Page Title
st.title("PMDS Project - Poker Hand Analysis")
st.write("Patru Gheorghe-Eduard, FMI-BIG DATA")

# What Do We Study?
st.markdown("## 1. What Do We Study?")
st.markdown("""
We study **Poker hand probabilities**, focusing on:
- The likelihood of obtaining different Poker hands (e.g., Royal Flush, Straight Flush).
- Comparing **empirical probabilities** (from the dataset) with **theoretical probabilities** (via Monte Carlo simulations).
""")

# How Do We Study It?
st.markdown("## 2. How Do We Study It?")
st.markdown("""
### Steps:
1. **Understand Hand Rules**:
   - Define the rules for classifying Poker hands (e.g., Flush, Straight).
2. **Empirical Analysis**:
   - Analyze the frequency of each Poker hand class in the dataset.
3. **Monte Carlo Simulation**:
   - Simulate millions of Poker hands by randomly shuffling and dealing cards.
   - Classify the hands based on the defined rules.
   - Estimate the probabilities of different Poker hands.
4. **Comparison**:
   - Compare empirical probabilities from the dataset with simulated probabilities.
   - Assess how well the dataset represents theoretical probabilities.
""")

# What For?
st.markdown("## 3. What For?")
st.markdown("""
The purpose of the study is:
- **Validation**:
    - Check if the dataset aligns with theoretical probabilities derived from simulations.
- **Insight**:
    - Gain insights into Poker hand distributions and their randomness.
- **Application**:
    - Use findings to explore:
        - Game fairness analysis.
        - Strategy optimization in Poker.
""")

# Plan for Implementation
st.markdown("## 4. Plan for Implementation")
st.markdown("""
1. **Load and Explore Data**:
   - Analyze hand class frequencies in the dataset.
2. **Monte Carlo Framework**:
   - Write a function to simulate Poker hands.
   - Classify hands into categories.
3. **Statistical Analysis**:
   - Compare probabilities between the dataset and simulations.
4. **Visualization**:
   - Plot frequency distributions and comparisons.
""")
