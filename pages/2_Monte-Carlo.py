import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from collections import Counter

# Class names
class_names = [
    "Nothing in hand", "One Pair", "Two Pairs", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"
]

# File paths
data_folder = "PokerData"
training_file = os.path.join(data_folder, "poker-hand-training-true.data")

# Column names
column_names = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'Hand_Class']

# Load dataset
def load_poker_dataset():
    """Load the Poker dataset."""
    try:
        training_data = pd.read_csv(training_file, header=None, names=column_names)
        return training_data
    except FileNotFoundError:
        st.error("Training data file not found.")
        return None


# Analyze hand class distribution
def analyze_distribution(training_data):
    """Analyze and visualize hand class distribution in the dataset."""
    hand_class_distribution = training_data['Hand_Class'].value_counts(normalize=True).sort_index()
    hand_class_percentages = hand_class_distribution * 100

    # Create a DataFrame with hand class names
    distribution_table = pd.DataFrame({
        'Hand Class': class_names,
        'Frequency': hand_class_distribution.values,
        'Percentage (%)': hand_class_percentages.values
    })

    # Display the distribution table
    st.write("### Poker Hand Class Distribution")
    st.dataframe(distribution_table.set_index('Hand Class'))

    # Plot distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(distribution_table['Hand Class'], distribution_table['Percentage (%)'])

    # Add labels to bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}%', ha='center', va='bottom')

    ax.set_xlabel('Poker Hand Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Poker Hand Class Distribution')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    st.pyplot(fig)

    return distribution_table


# Monte Carlo simulation
def monte_carlo_simulation(num_simulations=100000):
    """Simulate Poker hands and calculate probabilities using Monte Carlo."""
    suits = ['Hearts', 'Spades', 'Diamonds', 'Clubs']
    ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    deck = [(rank, suit) for rank in ranks for suit in suits]

    def classify_hand(hand):
        """Classify a Poker hand into one of the numeric classes (0–9)."""
        ranks = [card[0] for card in hand]
        suits = [card[1] for card in hand]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        rank_values = {
            "Ace": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
            "Jack": 11, "Queen": 12, "King": 13
        }
        numeric_ranks = sorted([rank_values[rank] for rank in ranks])
        is_straight = all(numeric_ranks[i] + 1 == numeric_ranks[i + 1] for i in range(len(numeric_ranks) - 1))
        is_flush = len(set(suits)) == 1
        is_royal = is_straight and is_flush and numeric_ranks == [10, 11, 12, 13, 1]

        if is_royal:
            return 9  # Royal Flush
        elif is_straight and is_flush:
            return 8  # Straight Flush
        elif 4 in rank_counts.values():
            return 7  # Four of a Kind
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            return 6  # Full House
        elif is_flush:
            return 5  # Flush
        elif is_straight:
            return 4  # Straight
        elif 3 in rank_counts.values():
            return 3  # Three of a Kind
        elif list(rank_counts.values()).count(2) == 2:
            return 2  # Two Pairs
        elif 2 in rank_counts.values():
            return 1  # One Pair
        else:
            return 0  # High Card

    results = [classify_hand(random.sample(deck, 5)) for _ in range(num_simulations)]

    # Calculate probabilities
    results_count = Counter(results)
    total = sum(results_count.values())
    probabilities = {hand: (count / total) * 100 for hand, count in results_count.items()}

    st.write("### Monte Carlo Simulation Results")
    st.dataframe(pd.DataFrame(probabilities.items(), columns=["Hand Class", "Probability (%)"]))

    # Plot simulation results with labels
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(probabilities.keys(), probabilities.values())

    # Add labels with dynamic vertical and horizontal adjustments
    for i, bar in enumerate(bars):
        height = bar.get_height()
        offset = max(1, height * 0.05)  # Vertical offset for readability
        stagger = 3 if i % 2 == 0 else -3  # Staggered vertical offset for adjacent bars
        horizontal_adjustment = bar.get_width() / 3  # Adjust simulated bars horizontally

        # Adjust positions for simulated vs. empirical bars
        x_pos = bar.get_x() + bar.get_width() / 2
        if i % 2 == 0:  # Simulated bars (even indices)
            x_pos += horizontal_adjustment

        ax.text(x_pos, height + offset + stagger,  # Combine adjustments
                f'{height:.2f}%', ha='center', va='bottom')

    ax.set_xlabel('Hand Class')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Monte Carlo Simulation of Poker Hands')
    st.pyplot(fig)

    return probabilities


# Comparison between empirical and simulated probabilities
def compare_probabilities(empirical, simulated):
    """Compare empirical and simulated probabilities."""
    comparison_table = pd.DataFrame({
        "Empirical (%)": empirical["Percentage (%)"],
        "Simulated (%)": [simulated.get(hand, 0) for hand in empirical["Hand_Class"]]
    }, index=empirical["Hand_Class"])
    st.write("### Comparison of Empirical and Simulated Probabilities")
    st.dataframe(comparison_table)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = comparison_table.index
    bar_width = 0.4

    empirical_bars = ax.bar(x - bar_width / 2, comparison_table["Empirical (%)"], width=bar_width, label="Empirical")
    simulated_bars = ax.bar(x + bar_width / 2, comparison_table["Simulated (%)"], width=bar_width, label="Simulated")

    for bar in empirical_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', ha='center', va='bottom')

    for bar in simulated_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', ha='center', va='bottom')

    ax.set_xlabel("Poker Hand Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Empirical vs. Simulated Probabilities")
    ax.legend()
    st.pyplot(fig)

# Main execution
st.title("Monte Carlo Analysis of Poker Hands")
training_data = load_poker_dataset()
if training_data is not None:
    empirical_distribution = analyze_distribution(training_data)
    simulated_distribution = monte_carlo_simulation()
    compare_probabilities(empirical_distribution, simulated_distribution)
