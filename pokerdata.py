import streamlit as st
import os

# Function to extract information from poker-hand.names
def extract_information(information_file):
    try:
        with open(information_file, 'r', encoding='latin1') as file:
            poker_hand_names_content = file.readlines()
        start_section = False
        relevant_info = []
        for line in poker_hand_names_content:
            if "4." in line:  # Start collecting from this line
                start_section = True
            if start_section:
                relevant_info.append(line.strip())
        return "\n".join(relevant_info)
    except FileNotFoundError:
        return "File not found. Ensure the 'PokerData' folder contains the poker-hand.names file."