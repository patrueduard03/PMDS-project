import streamlit as st
from poker_functions import (
    load_poker_datasets,
    extract_information,
    distribution_of_hand,
    monte_carlo_simulation,
)


# Define app pages
def homepage():
    st.title("Poker Dataset Information")
    st.markdown("### Welcome to the Poker Dataset Explorer!")

    information_file = "PokerData/poker-hand.names"
    relevant_info_text = extract_information(information_file)
    st.markdown("### Extracted Dataset Information")
    st.text(relevant_info_text)


def monte_carlo_page():
    st.title("Monte Carlo Analysis")
    st.markdown("### Analyze Poker Dataset with Stochastic Models")

    # Load datasets
    training_data, testing_data = load_poker_datasets()
    if training_data is not None:
        st.markdown("#### Training Data Overview")
        st.write(training_data.head())

        # Display distribution
        st.markdown("### Poker Hand Class Distribution")
        distribution_of_hand(training_data)

        # Perform Monte Carlo simulation
        st.markdown("### Monte Carlo Simulation")
        simulation_results = monte_carlo_simulation()
        st.write(simulation_results)


# Streamlit page navigation
PAGES = {
    "Home": homepage,
    "Monte Carlo Analysis": monte_carlo_page,
}

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()
