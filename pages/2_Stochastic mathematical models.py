import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from collections import Counter
import plotly.graph_objects as go
import numpy as np

# ======================================
# Section 1: File Paths and Configurations
# ======================================
# Define folder and file paths
# These variables specify where the dataset is located and help load the data correctly.
data_folder = "PokerData"
training_file = os.path.join(data_folder, "poker-hand-training-true.data")

# Define column names for the dataset
# These correspond to the suits (S) and ranks (R) of the cards and the final hand class.
column_names = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'Hand_Class']

# Define human-readable class names for poker hands
class_names = [
    "Nothing in hand", "One Pair", "Two Pairs", "Three of a Kind", "Straight",
    "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush"
]

# ======================================
# Section 2: Loading and Preprocessing Dataset
# ======================================
def load_poker_dataset():
    """Load the Poker dataset."""
    try:
        # Load the dataset with predefined column names
        training_data = pd.read_csv(training_file, header=None, names=column_names)
        return training_data
    except FileNotFoundError:
        # Handle the case where the dataset file is missing
        st.error("Training data file not found.")
        return None

# ======================================
# Section 3: Empirical Distribution Analysis
# ======================================
def analyze_distribution(training_data):
    """Analyze the distribution of poker hand classes in the dataset."""
    # Calculate the frequency and percentage of each hand class
    hand_class_distribution = training_data['Hand_Class'].value_counts(normalize=True).sort_index()
    hand_class_percentages = hand_class_distribution * 100

    # Create a table summarizing the distribution
    distribution_table = pd.DataFrame({
        'Hand Class ID': hand_class_distribution.index,
        'Hand Name': [class_names[i] for i in hand_class_distribution.index],
        'Frequency': hand_class_distribution.values,
        'Percentage (%)': hand_class_percentages.values
    })

    # Display the distribution as a bar chart
    st.write("### Empirical Poker Hand Class Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(distribution_table['Hand Name'], distribution_table['Percentage (%)'])

    # Annotate bars with percentage values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}%', ha='center', va='bottom')

    ax.set_xlabel('Poker Hand Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Poker Hand Class Distribution')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    st.pyplot(fig)

    return distribution_table

# ======================================
# Section 4: Monte Carlo Simulation
# ======================================
def monte_carlo_simulation(num_simulations=100000):
    """Simulate poker hands and estimate probabilities using Monte Carlo simulation."""

    st.write("### Monte Carlo Poker Hand Class Distribution")

    # Define a full poker deck with suits and ranks
    suits = ['Hearts', 'Spades', 'Diamonds', 'Clubs']
    ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    deck = [(rank, suit) for rank in ranks for suit in suits]

    # Define a helper function to classify poker hands
    def classify_hand(hand):
        ranks = [card[0] for card in hand]
        suits = [card[1] for card in hand]
        rank_values = {"Ace": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "Jack": 11, "Queen": 12, "King": 13}
        numeric_ranks = sorted([rank_values[rank] for rank in ranks])

        # Check for different poker hand conditions
        is_straight = all(numeric_ranks[i] + 1 == numeric_ranks[i + 1] for i in range(len(numeric_ranks) - 1))
        is_flush = len(set(suits)) == 1
        is_royal = is_straight and is_flush and numeric_ranks == [10, 11, 12, 13, 1]

        if is_royal:
            return 9  # Royal Flush
        elif is_straight and is_flush:
            return 8  # Straight Flush
        elif 4 in Counter(ranks).values():
            return 7  # Four of a Kind
        elif 3 in Counter(ranks).values() and 2 in Counter(ranks).values():
            return 6  # Full House
        elif is_flush:
            return 5  # Flush
        elif is_straight:
            return 4  # Straight
        elif 3 in Counter(ranks).values():
            return 3  # Three of a Kind
        elif list(Counter(ranks).values()).count(2) == 2:
            return 2  # Two Pairs
        elif 2 in Counter(ranks).values():
            return 1  # One Pair
        return 0  # High Card

    # Simulate hands and classify them
    results = [classify_hand(random.sample(deck, 5)) for _ in range(num_simulations)]

    # Calculate the probabilities of each hand class
    results_count = Counter(results)
    probabilities = {hand: count / num_simulations * 100 for hand, count in results_count.items()}

    # Display the probabilities as a bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(probabilities)), list(probabilities.values()), tick_label=[class_names[i] for i in probabilities.keys()])

    # Annotate bars with percentages
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}%', ha='center', va='bottom')

    ax.set_xlabel('Poker Hand Class')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Monte Carlo Simulation of Poker Hands')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    return probabilities

# ======================================
# Section 5: Markov Chain Analysis
# ======================================
def create_transition_matrix(data):
    """Create a transition matrix from the dataset."""
    num_classes = 10  # Total number of hand classes
    transition_matrix = np.zeros((num_classes, num_classes))

    # Count transitions between consecutive rows
    for i in range(len(data) - 1):
        current_class = data.iloc[i]['Hand_Class']
        next_class = data.iloc[i + 1]['Hand_Class']
        transition_matrix[current_class][next_class] += 1

    # Normalize the matrix to convert counts to probabilities
    smoothing_factor = 1e-8
    transition_matrix += smoothing_factor
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

    return transition_matrix

def simulate_markov_chain(transition_matrix, num_steps=100, initial_state=None):
    """Simulate the evolution of hand probabilities using a Markov chain."""
    num_classes = len(transition_matrix)

    # Use a skewed initial state if none is provided
    if initial_state is None:
        initial_state = np.zeros(num_classes)
        initial_state[0] = 1  # Start with all probability in "Nothing in hand"

    state = initial_state
    states = [state]

    # Update the state vector iteratively
    for _ in range(num_steps):
        state = state.dot(transition_matrix)
        states.append(state)

    return np.array(states)

def plot_markov_results(states):
    """Visualize the Markov chain results."""
    adjusted_states = states + 1e-3  # Add small offset for visibility

    fig = go.Figure()
    for idx in range(adjusted_states.shape[1]):
        fig.add_trace(go.Scatter(
            y=adjusted_states[:, idx],
            mode='lines+markers',
            name=f'{class_names[idx]}',
            line=dict(width=3 if np.max(adjusted_states[:, idx]) > 0.1 else 2)
        ))

    fig.update_layout(
        title="Markov Chain Simulation Over Time",
        xaxis_title="Steps",
        yaxis_title="State Probability",
        yaxis=dict(type='log', tickformat=".2f"),
        legend_title="Poker Hand Classes",
        template="plotly_white",
        width=1000,
        height=600
    )
    st.plotly_chart(fig)


# ======================================
# Section 6: Comparing Empirical and Simulated Probabilities
# ======================================
def compare_probabilities(empirical, simulated):
    """Compare empirical and simulated probabilities and visualize the results."""
    # Prepare data for the plot
    categories = empirical['Hand Name']
    empirical_values = empirical['Percentage (%)']
    simulated_values = [simulated.get(hand_id, 0) for hand_id in empirical['Hand Class ID']]

    # Format percentages to three decimal places for display
    empirical_text = [f'{x:.3f}%' for x in empirical_values]
    simulated_text = [f'{x:.3f}%' for x in simulated_values]

    # Create a Plotly figure for side-by-side comparison
    fig = go.Figure()

    # Add empirical data as a horizontal bar chart
    fig.add_trace(go.Bar(
        name='Empirical',
        y=categories,
        x=empirical_values,
        orientation='h',
        hoverinfo='x',
        marker_color='rgba(255, 99, 132, 0.8)',  # Red color for empirical
        text=empirical_text,  # Display percentages on bars
        textposition='inside',  # Position text inside bars
        textfont=dict(size=14, color='white')  # Text font adjustments
    ))

    # Add simulated data as a horizontal bar chart
    fig.add_trace(go.Bar(
        name='Monte Carlo',
        y=categories,
        x=simulated_values,
        orientation='h',
        hoverinfo='x',
        marker_color='rgba(54, 162, 235, 0.8)',  # Blue color for Monte Carlo
        text=simulated_text,  # Display percentages on bars
        textposition='inside',  # Position text inside bars
        textfont=dict(size=14, color='white')  # Text font adjustments
    ))

    # Update layout for better readability
    fig.update_layout(
        barmode='group',
        title='Comparison of Empirical and Monte Carlo Simulated Probabilities',
        title_font_size=24,
        xaxis_title='Percentage (%)',
        xaxis_title_font_size=16,
        yaxis_title='Poker Hand Class',
        yaxis_title_font_size=16,
        legend_title='Source',
        legend_title_font_size=16,
        legend_font_size=14,
        hovermode='y unified',
        width=1200,  # Adjusted width for clarity
        height=800   # Adjusted height for larger labels
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)


# ======================================
# Section 7: Main Execution
# ======================================
st.title("Stochastic Mathematical Models")

# Load the dataset
training_data = load_poker_dataset()
if training_data is not None:
    # Analyze empirical distribution
    empirical_distribution = analyze_distribution(training_data)

    # Perform Monte Carlo simulation
    simulated_distribution = monte_carlo_simulation()

    # Compare empirical and simulated distributions
    compare_probabilities(empirical_distribution, simulated_distribution)

    # Create the transition matrix for the Markov chain
    transition_matrix = create_transition_matrix(training_data)

    # Simulate Markov chain evolution
    initial_state = np.zeros(10)
    initial_state[0] = 1  # Start with "Nothing in hand"
    markov_results = simulate_markov_chain(transition_matrix, num_steps=50, initial_state=initial_state)

    # Visualize the Markov chain results
    plot_markov_results(markov_results)
