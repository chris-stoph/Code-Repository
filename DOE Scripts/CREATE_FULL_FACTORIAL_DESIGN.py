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
import json
import datetime
import sys
from IPython import get_ipython
import numpy as np
import tkinter.messagebox as messagebox


# ## CREATE FACTORIAL DESIGN PLAN

# In[ ]:


def generate_factor_levels_df(factors, levels, high_levels, low_levels):
    """
    Generate a DataFrame with factor names, number of levels, and high and low levels.

    Args:
    - factors (list): List of factor names.
    - levels (list): List containing the number of levels for each factor.
    - high_levels (list): List of high factor levels for each factor.
    - low_levels (list): List of low factor levels for each factor.

    Returns:
    - df (DataFrame): DataFrame representing factor names, levels, high, mid, and low levels.
    """
    # Check if the number of factors matches the number of levels
    if len(factors) != len(levels) != len(high_levels) != len(low_levels):
        raise ValueError("Number of factors, levels, high, and low levels must be the same.")
    # Check if all levels are positive integers
    if not all(isinstance(level, int) and level > 0 for level in levels):
        raise ValueError("Levels must be positive integers.")

    # Create lists to hold factor names, levels, high, mid, and low levels
    factor_list = []
    level_list = []
    high_list = []
    mid_list = []
    low_list = []

    # Iterate through factors and calculate mid levels if levels > 2
    for factor, num_levels, high_level, low_level in zip(factors, levels, high_levels, low_levels):
        factor_list.append(factor)
        level_list.append(num_levels)
        high_list.append(high_level)
        low_list.append(low_level)

        # Calculate mid levels
        if num_levels > 2:
            mid = [(low_level + ((high_level - low_level) * i) / (num_levels - 1)) for i in range(1, num_levels - 1)]
            # Round mid levels to two decimal places
            mid = sorted([round(m, 1) for m in mid], reverse=True)
            mid_list.append(mid)
        else:
            mid_list.append(None)

    # Create a dictionary to hold factor names, levels, high, mid, and low levels
    data = {'Factor': factor_list, 'Levels': level_list, 'High(+1)': high_list, '-1<Mid<1': mid_list, 'Low(-1)': low_list}
    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    return df

def create_full_factorial_design(factor_levels_df):
    """
    Generate a full factorial design DataFrame based on specified factors and levels.

    Args:
    - factor_levels_df (DataFrame): DataFrame containing factor names, levels, high, mid, and low levels.

    Returns:
    - df (DataFrame): DataFrame representing the full factorial design.
    """
    # Extract factor names and levels from the DataFrame
    factors = factor_levels_df['Factor'].tolist()
    levels = factor_levels_df['Levels'].tolist()
    
    # Create lists to hold factor names and their corresponding mapped columns
    factor_columns = []
    mapped_columns = []

    # Iterate through factors and create corresponding mapped columns
    for factor, num_levels in zip(factors, levels):
        # Create factor column
        factor_columns.append(factor)
        # Create mapped column
        mapped_column_name = factor + '_mapped'
        mapped_columns.append(mapped_column_name)
        
    # Generate the full factorial design using pyDOE2's fullfact() function
    design = fullfact(levels)
    # Convert the design into a DataFrame with appropriate column names
    df = pd.DataFrame(design, columns=factor_columns)

    # Apply custom mapping to each value in the DataFrame
    for factor, mapped_column in zip(factors, mapped_columns):
        num_levels = factor_levels_df.loc[factor_levels_df['Factor'] == factor, 'Levels'].iloc[0]
        df[factor] = df[factor].apply(lambda x: custom_mapping(x, num_levels))
        # Add blank column for mapped values
        df[mapped_column] = ''

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


def get_user_input():
    """
    Prompt the user to input factors, levels, number of replicates, Excel file name, and export folder.
    Additionally, prompt for high, mid, and low factor levels for each factor based on the number of levels.

    Returns:
    - factors (list): List of factor names.
    - levels (list): List of levels corresponding to each factor.
    - high_levels (list): List of high factor levels for each factor.
    - low_levels (list): List of low factor levels for each factor.
    - num_replicates (int): Number of replicates.
    - excel_file_name (str): Excel file name.
    - export_folder (str): Export folder path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt user to input the number of factors
    num_factors_str = simpledialog.askstring("Input", "Enter the number of factors:")
    num_factors = int(num_factors_str)

    factors = []
    levels = []
    high_levels = []
    low_levels = []

    for i in range(num_factors):
        # Prompt user to input factor name
        factor_name = simpledialog.askstring("Input", f"Enter the name of factor {i + 1}:")
        factors.append(factor_name)

        # Prompt user to input levels for the factor
        levels_str = simpledialog.askstring("Input", f"Enter the number of levels for factor {factor_name}:")
        levels.append(int(levels_str))

        # Prompt user to input high level for the factor
        high_level_str = simpledialog.askstring("Input", f"Enter the high level for factor {factor_name}:")
        high_level = int(high_level_str)

        # Prompt user to input low level for the factor
        while True:
            low_level_str = simpledialog.askstring("Input", f"Enter the low level for factor {factor_name}:")
            low_level = int(low_level_str)
            if low_level <= high_level:
                break
            else:
                messagebox.showerror("Error", "Low level cannot be larger than high level. Please enter a valid low level.")

        high_levels.append(high_level)
        low_levels.append(low_level)

    # Prompt user to input number of replicates
    num_replicates_str = simpledialog.askstring("Input", "Enter number of replicates:")
    num_replicates = int(num_replicates_str)

    # Prompt user to input Excel file name
    excel_file_name = simpledialog.askstring("Input", "Enter Excel file name (without extension):")

    # Prompt user to select the export folder
    export_folder = filedialog.askdirectory(title="Select Export Folder")

    return factors, levels, high_levels, low_levels, num_replicates, excel_file_name, export_folder


def main():
    try:
        # Get user input
        factors, levels, high_levels, low_levels, num_replicates, excel_file_name, export_folder = get_user_input()

        # Generate factor levels DataFrame
        factor_levels_df = generate_factor_levels_df(factors, levels, high_levels, low_levels)

        # Create full factorial design DataFrame
        full_factorial_df = create_full_factorial_design(factor_levels_df)

        # Duplicate the full factorial design DataFrame
        full_factorial_df_duplicated = pd.concat([full_factorial_df] * num_replicates, ignore_index=True)

        # Merge the two DataFrames
        merged_df = pd.concat([factor_levels_df, pd.DataFrame(columns=[''] * 2), full_factorial_df_duplicated], axis=1)

        # Add a blank column with heading 'Results' after the last factor
        merged_df.insert(len(merged_df.columns), 'Results', '')

        # Append '.xlsx' extension if not provided
        if not excel_file_name.endswith('.xlsx'):
            excel_file_name += '.xlsx'

        # Write the merged DataFrame to an Excel file in the export folder
        excel_file_path = os.path.join(export_folder, excel_file_name)

        # Write DataFrame to Excel file with header included
        merged_df.to_excel(excel_file_path, sheet_name='Worksheet', index=False, header=True)

        print("Excel file saved successfully.")

    except Exception as e:
        # Write the error message to a file
        with open("error_log.txt", "w") as f:
            f.write(str(e))
        # Print a message indicating where the error log file is located
        print("An error occurred. Check error_log.txt for details.")

if __name__ == "__main__":
    main()

