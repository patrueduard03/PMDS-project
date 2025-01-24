import streamlit as st
import pandas as pd

# Page Title
st.title("PMDS Project")
st.write("Patru Gheorghe-Eduard, FMI-BIG DATA")

st.title("Poker Dataset Information")
st.write("### Welcome to the Poker Dataset Explorer!")
st.write("This page provides a detailed overview of the dataset structure, attributes, and statistics.")

# Dataset Description
st.markdown("## 1. Dataset Description")
st.markdown("""
- **Instance Type**: Each record represents a Poker hand consisting of 5 cards drawn from a standard 52-card deck.
- **Attributes**:
    - **Predictive Attributes**: 10 (Card suits and ranks for each of the 5 cards).
    - **Target Attribute**: 1 (Poker hand classification).
- **Order of Cards**: Important to maintain combinations and probabilities.
- **Special Note**: Rare hands (e.g., Royal Flush) are over-sampled in training and testing datasets.
""")

# Dataset Statistics
st.markdown("## 2. Dataset Statistics")
st.write("""
- **Training Set**: 25,010 instances.
- **Testing Set**: 1,000,000 instances.
- **Domain Size**: 311,875,200 possible Poker hands.
""")

# Attribute Information
st.markdown("## 3. Attribute Information")
attribute_data = {
    "Attribute": ["S1, S2, S3, S4, S5", "C1, C2, C3, C4, C5", "CLASS"],
    "Description": [
        "Suit of cards 1–5 (Ordinal 1-4: {Hearts, Spades, Diamonds, Clubs})",
        "Rank of cards 1–5 (Numerical 1-13: {Ace, 2, 3, ..., Queen, King})",
        "Poker Hand classification (Ordinal 0-9)"
    ]
}
attribute_df = pd.DataFrame(attribute_data)
st.table(attribute_df)

# Poker Hand Classes
st.markdown("## 4. Poker Hand Classes")
class_data = {
    "Class ID": list(range(10)),
    "Hand Name": [
        "Nothing in hand", "One Pair", "Two Pairs", "Three of a Kind", "Straight",
        "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"
    ],
    "Description": [
        "Not a recognized Poker hand", "One pair of equal ranks", "Two pairs of equal ranks",
        "Three equal ranks", "Five sequentially ranked cards",
        "Five cards of the same suit", "Pair + three of a kind",
        "Four equal ranks", "Straight + Flush", "{Ace, King, Queen, Jack, Ten} + Flush"
    ]
}
class_df = pd.DataFrame(class_data)
st.table(class_df)

# Class Distribution
st.markdown("## 5. Class Distribution")
st.write("### Training Set")
training_distribution = {
    "Class": ["Nothing in hand", "One Pair", "Two Pairs", "Three of a Kind", "Straight",
              "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"],
    "Instances": [12493, 10599, 1206, 513, 93, 54, 36, 6, 5, 5],
    "Percentage (%)": [49.95, 42.38, 4.82, 2.05, 0.37, 0.22, 0.14, 0.02, 0.02, 0.02],
    "Domain Probability (%)": [50.12, 42.26, 4.75, 2.11, 0.39, 0.20, 0.14, 0.02, 0.001, 0.0002]
}
training_df = pd.DataFrame(training_distribution)
st.table(training_df)

st.write("### Testing Set")
testing_distribution = {
    "Class": ["Nothing in hand", "One Pair", "Two Pairs", "Three of a Kind", "Straight",
              "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"],
    "Instances": [501209, 422498, 47622, 21121, 3885, 1996, 1424, 230, 12, 3],
    "Representation Ratio": [1.000063, 0.999832, 1.001746, 0.999647, 0.989897,
                              1.015569, 0.988491, 0.957934, 0.866426, 1.948052]
}
testing_df = pd.DataFrame(testing_distribution)
st.table(testing_df)

# Combinatorial Statistics
st.markdown("## 6. Combinatorial Statistics")
st.write("### Poker Hand Combinations")
combinatorial_data = {
    "Poker Hand": [
        "Royal Flush", "Straight Flush", "Four of a Kind", "Full House",
        "Flush", "Straight", "Three of a Kind", "Two Pairs", "One Pair", "Nothing in Hand"
    ],
    "# of Hands": [4, 36, 624, 3744, 5108, 10200, 54912, 123552, 1098240, 1302540],
    "Probability (%)": [0.0002, 0.0014, 0.024, 0.14, 0.20, 0.39, 2.11, 4.75, 42.26, 50.12],
    "# of Combinations": [480, 4320, 74880, 449280, 612960, 1224000, 6589440, 14826240, 131788800, 156304800]
}
combinatorial_df = pd.DataFrame(combinatorial_data)
st.table(combinatorial_df)

# Observations
st.markdown("## 7. Observations")
st.markdown("""
1. **Class Over-Representation**:
    - Rare hands like Royal Flush and Straight Flush are over-sampled in the training data.
2. **Training and Testing Size**:
    - Training: 25,010 instances.
    - Testing: 1,000,000 instances.
3. **Importance of Card Order**:
    - Card order determines combinatorial probabilities, leading to 480 possible Royal Flush hands instead of just 4.
""")

# Glossary Section
st.markdown("## 8. Glossary")
st.markdown("Below is a table explaining the key terms used throughout this dataset overview:")

glossary_data = {
    "Term": [
        "Instances",
        "Percentage (%)",
        "Domain Probability (%)",
        "Representation Ratio",
        "Training Set",
        "Testing Set",
        "# of Hands",
        "Probability (%)",
        "# of Combinations"
    ],
    "Explanation": [
        "The number of examples or records in the dataset.",
        "The percentage representation of a specific class or statistic within the dataset.",
        "The theoretical probability of a Poker hand occurring in the entire domain of possible hands.",
        "The ratio of a class's representation in the dataset compared to its true domain representation. Values >1 indicate over-representation, <1 indicate under-representation.",
        "A subset of the dataset used for training models, consisting of 25,010 instances.",
        "A larger subset of the dataset used for testing models, consisting of 1,000,000 instances.",
        "The total number of possible hands for a specific Poker hand type.",
        "The probability of obtaining a specific Poker hand type, expressed as a percentage.",
        "The number of unique card combinations that result in a specific Poker hand type."
    ]
}
glossary_df = pd.DataFrame(glossary_data)
st.table(glossary_df)

# Additional Notes on Poker Game Type
st.markdown("### Poker Game Type and Rules")
st.markdown("""
The dataset represents a simplified form of **5-card draw Poker**, adhering to the following rules:
- **Game Type**: **5-Card Draw**
    - Each hand consists of exactly **5 cards**, randomly dealt from a standard 52-card deck.
- **Deck Details**:
    - **Ranks**: Ace (1), 2–10, Jack, Queen, King (13 ranks).
    - **Suits**: Hearts, Spades, Diamonds, Clubs (4 suits).
- **Order of Cards**:
    - The order of cards in a hand is important and affects the classification of the hand.
- **No Player Actions**:
    - The dataset focuses solely on the classification of hands and does not include player actions like betting, folding, or drawing replacement cards.
- **Purpose**:
    - The dataset is suitable for analyzing probabilities and distributions of Poker hands within the 5-card draw framework.
""")
