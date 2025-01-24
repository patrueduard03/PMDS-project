import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from collections import Counter

# Define the folder containing the datasets
data_folder = "PokerData"

# File paths
training_file = os.path.join(data_folder, "poker-hand-training-true.data")
testing_file = os.path.join(data_folder, "poker-hand-testing.data")
information = os.path.join(data_folder, "poker-hand.names")

# Column names based on the dataset documentation
column_names = [
    'S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'Hand_Class'
]

# Load datasets
def load_poker_datasets():
    try:
        # Load the training data
        training_data = pd.read_csv(training_file, header=None, names=column_names)
        print("Training data loaded successfully:")
        print(training_data.head())

        # Load the testing data
        testing_data = pd.read_csv(testing_file, header=None, names=column_names)
        print("\nTesting data loaded successfully:")
        print(testing_data.head())

        return training_data, testing_data

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the 'PokerData' folder is in the project directory and contains the dataset files.")
        return None, None

# Analyze and visualize the distribution of hand classes in the training data
def distribution_of_hand(training_data):
    # Frequency distribution of the Hand_Class column
    hand_class_distribution = training_data['Hand_Class'].value_counts(normalize=True).sort_index()

    # Convert to percentages for better interpretation
    hand_class_percentages = hand_class_distribution * 100

    # Creating a distribution table for clarity
    hand_class_distribution_table = pd.DataFrame({
        'Hand_Class': hand_class_distribution.index,
        'Frequency': hand_class_distribution.values,
        'Percentage (%)': hand_class_percentages.values
    })

    # Printing the distribution table
    print("Poker Hand Class Distribution:")
    print(hand_class_distribution_table)

    # Plotting the percentage distribution of Poker hands
    plt.figure(figsize=(10, 6))
    plt.bar(hand_class_distribution_table['Hand_Class'], hand_class_distribution_table['Percentage (%)'],
            tick_label=hand_class_distribution_table['Hand_Class'])
    plt.xlabel('Poker Hand Class')
    plt.ylabel('Percentage (%)')
    plt.title('Poker Hand Class Distribution in Training Data')
    plt.xticks(hand_class_distribution_table['Hand_Class'])
    plt.show()

def info():
    print(information)
    # Open the file and process it
    with open(information, 'r', encoding='latin1') as file:
        poker_hand_names_content = file.readlines()

    # Extract relevant information starting from the line containing "4:"
    start_section = False
    relevant_info = []
    for line in poker_hand_names_content:
        if "4." in line:  # Start collecting from this line
            start_section = True
        if start_section:
            relevant_info.append(line.strip())

    # Join and display the relevant content
    relevant_info_text = "\n".join(relevant_info)
    print(relevant_info_text)

# Monte Carlo simulation
def monte_carlo_simulation(num_simulations=100000):
    # Generate a standard 52-card deck
    suits = ['Hearts', 'Spades', 'Diamonds', 'Clubs']
    ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    deck = [(rank, suit) for rank in ranks for suit in suits]

    # Classify Poker hands (placeholder logic)
    def classify_hand(hand):
        ranks = [card[0] for card in hand]
        suits = [card[1] for card in hand]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        if len(suit_counts) == 1:  # All cards of the same suit
            return "Flush"
        elif len(rank_counts) == 2:  # Full house or four of a kind
            return "Four of a Kind" if 4 in rank_counts.values() else "Full House"
        elif len(rank_counts) == 3:  # Three of a kind or two pairs
            return "Three of a Kind" if 3 in rank_counts.values() else "Two Pairs"
        elif len(rank_counts) == 4:  # One pair
            return "One Pair"
        else:
            return "High Card"

    # Simulate hands
    results = []
    for _ in range(num_simulations):
        hand = random.sample(deck, 5)  # Deal 5 random cards
        hand_type = classify_hand(hand)
        results.append(hand_type)

    # Count results
    results_count = Counter(results)
    total = sum(results_count.values())
    probabilities = {hand: (count / total) * 100 for hand, count in results_count.items()}

    # Display results
    print("Monte Carlo Simulation Results:")
    for hand, probability in probabilities.items():
        print(f"{hand}: {probability:.2f}%")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(probabilities.keys(), probabilities.values())
    plt.xlabel('Hand Type')
    plt.ylabel('Probability (%)')
    plt.title('Monte Carlo Simulation of Poker Hands')
    plt.xticks(rotation=45)
    plt.show()

# Main Function
if __name__ == "__main__":
    # Load the datasets
    training_data, testing_data = load_poker_datasets()

    if training_data is not None:
        # Display extracted information
        info()

        # Analyze the hand distribution in the training data
        distribution_of_hand(training_data)

        # Perform Monte Carlo simulation
        monte_carlo_simulation()
