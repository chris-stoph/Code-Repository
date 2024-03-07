#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[ ]:


# Import necessary libraries
import tkinter as tk
from tkinter import simpledialog, filedialog
import pandas as pd
from pyDOE2 import fullfact
import os


# ## CREATE FACTORIAL DESIGN PLAN

# In[ ]:


# Function to generate factor levels DataFrame
def generate_factor_levels_df(factors, levels):
    """
    Generate a DataFrame with factor names and number of levels.

    Args:
    - factors (list): List of factor names.
    - levels (list): List containing the number of levels for each factor.

    Returns:
    - df (DataFrame): DataFrame representing factor names and levels.
    """
    # Check if the number of factors matches the number of levels
    if len(factors) != len(levels):
        raise ValueError("Number of factors and levels must be the same.")
    # Check if all levels are positive integers
    if not all(isinstance(level, int) and level > 0 for level in levels):
        raise ValueError("Levels must be positive integers.")

    # Create a dictionary to hold factor names and levels
    data = {'Factor': factors, 'Levels': levels}
    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)
    return df

# Function to create full factorial design DataFrame
def create_full_factorial_design(factor_levels_df):
    """
    Generate a full factorial design DataFrame based on specified factors and levels.

    Args:
    - factor_levels_df (DataFrame): DataFrame containing factor names and levels.

    Returns:
    - df (DataFrame): DataFrame representing the full factorial design.
    """
    # Extract factor names and levels from the DataFrame
    factors = factor_levels_df['Factor'].tolist()
    levels = factor_levels_df['Levels'].tolist()
    # Generate the full factorial design using pyDOE2's fullfact() function
    design = fullfact(levels)
    # Convert the design into a DataFrame with appropriate column names
    df = pd.DataFrame(design, columns=factors)
    return df

# Define the custom mapping function
def custom_mapping(value, levels):
    """
    Custom mapping function to transform values according to specific rules.

    Args:
    - value (int): Value representing factor level.
    - levels (int): Number of levels for the factor.

    Returns:
    - mapped_value (float): Transformed value.
    """
    # Check if the value is within the valid range of levels
    if value < 0 or value >= levels:
        return value  # Value outside the defined levels, return as is

    # Calculate the spacing between each level
    spacing = 2 / (levels - 1)

    # Map the value to the corresponding value in the range [-1, 1]
    mapped_value = -1 + value * spacing

    return mapped_value

# Function to handle user input
def get_user_input():
    """
    Prompt the user to input factors, levels, number of replicates, Excel file name, and export folder.

    Returns:
    - factors (list): List of factor names.
    - levels (list): List of levels corresponding to each factor.
    - num_replicates (int): Number of replicates.
    - excel_file_name (str): Excel file name.
    - export_folder (str): Export folder path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt user for the number of factors
    num_factors = simpledialog.askinteger("Input", "Enter the number of factors:")

    factors = []
    levels = []

    # Prompt user to enter factor names and levels
    for i in range(num_factors):
        factor = simpledialog.askstring("Input", f"Enter the name of factor {i+1}:")
        factors.append(factor)

        level = simpledialog.askinteger("Input", f"Enter the number of levels for factor {factor}:")
        levels.append(level)

    # Prompt user to enter the number of replicates
    num_replicates = simpledialog.askinteger("Input", "Enter the number of replicates:")

    # Prompt user to enter the Excel file name
    excel_file_name = simpledialog.askstring("Input", "Enter the Excel file name:")

    # Prompt user to select the export folder
    export_folder = filedialog.askdirectory(title="Select Export Folder")

    return factors, levels, num_replicates, excel_file_name, export_folder

# Main function
def main():
    try:
        # Get user input
        factors, levels, num_replicates, excel_file_name, export_folder = get_user_input()

        # Generate factor levels DataFrame
        factor_levels_df = generate_factor_levels_df(factors, levels)

        # Create full factorial design DataFrame
        full_factorial_df = create_full_factorial_design(factor_levels_df)

        # Apply custom mapping to each value in the DataFrame
        for column in full_factorial_df.columns:
            num_levels = factor_levels_df.loc[factor_levels_df['Factor'] == column, 'Levels'].iloc[0]
            full_factorial_df[column] = full_factorial_df[column].apply(lambda x: custom_mapping(x, num_levels))

        # Duplicate the full factorial design DataFrame
        full_factorial_df_duplicated = pd.concat([full_factorial_df] * num_replicates, ignore_index=True)

        # Merge the two DataFrames
        merged_df = pd.concat([factor_levels_df, pd.DataFrame(columns=['']), full_factorial_df_duplicated], axis=1)

        # Add a blank column with heading 'Results' after the last factor
        merged_df.insert(len(merged_df.columns), 'Results', '')

        # Append '.xlsx' extension if not provided
        if not excel_file_name.endswith('.xlsx'):
            excel_file_name += '.xlsx'

        # Write the merged DataFrame to an Excel file in the export folder
        excel_file_path = os.path.join(export_folder, excel_file_name)
        merged_df.to_excel(excel_file_path, index=False)
        print("Excel file saved successfully.")

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()

