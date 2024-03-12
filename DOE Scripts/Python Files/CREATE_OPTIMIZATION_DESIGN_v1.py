#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[7]:


# Import necessary libraries
import tkinter as tk
from tkinter import Tk, simpledialog, filedialog, messagebox  # Import necessary GUI modules
import os
import pandas as pd
from pyDOE2 import bbdesign


# ## CREATE BOX-BEHNKEN DESIGN PLAN

# In[8]:


# Function to generate a DataFrame with factor names, high, low, and mid levels
def generate_factor_levels_df(factors, high_levels, low_levels):
    """
    Generate a DataFrame containing factor names, high levels, low levels, and mid levels.

    Args:
    - factors (list): List of factor names.
    - high_levels (list): List of high levels for each factor.
    - low_levels (list): List of low levels for each factor.

    Returns:
    - df (DataFrame): DataFrame representing factor names, high levels, low levels, and mid levels.
    """
    # Calculate mid levels as the average of high and low levels
    mid_levels = [(high + low) / 2 for high, low in zip(high_levels, low_levels)]
    # Create DataFrame
    data = {'Factor': factors, 'High': high_levels, 'Mid': mid_levels, 'Low': low_levels}
    df = pd.DataFrame(data)
    return df

# Function to generate a Box-Behnken design DataFrame
def generate_bb_design(factor_df, center_points=0):
    """
    Generate a Box-Behnken design DataFrame based on specified factors and center points.

    Args:
    - factor_df (DataFrame): DataFrame containing factor names, high levels, low levels, and mid levels.
    - center_points (int): Number of center points to include in the design.

    Returns:
    - df (DataFrame): DataFrame representing the Box-Behnken design.
    """
    factors = factor_df['Factor'].tolist()

    # Create factor columns and mapped columns
    factor_columns = []
    mapped_columns = []
    for factor in factors:
        factor_columns.append(factor)
        mapped_column_name = factor + '_mapped'
        mapped_columns.append(mapped_column_name)
    
    # Generate Box-Behnken design
    design = bbdesign(len(factors), center=center_points)

    column_names = factors.copy()
    df = pd.DataFrame(design, columns=column_names)

    # Add mapped columns
    for factor, mapped_column in zip(factors, mapped_columns):
        df[mapped_column] = ''

    return df

# Define the custom mapping function
def custom_mapping(value):
    """
    Custom mapping function to transform values according to specific rules.

    Args:
    - value (int): Value representing factor level.

    Returns:
    - mapped_value (float): Transformed value.
    """
    return value

# Function to handle user input
def get_user_input():
    """
    Prompt the user to input factors, high levels, low levels, number of replicates, Excel file name, and export folder.

    Returns:
    - factors (list): List of factor names.
    - high_levels (list): List of high levels corresponding to each factor.
    - low_levels (list): List of low levels corresponding to each factor.
    - num_replicates (int): Number of replicates.
    - excel_file_name (str): Excel file name.
    - export_folder (str): Export folder path.
    """
    try:
        root = tk.Tk()
        root.withdraw()  

        num_factors = simpledialog.askinteger("Input", "Enter the number of factors:")
        num_factors = int(num_factors)

        factors = []
        high_levels = []
        low_levels = []

        for i in range(num_factors):
            factor = simpledialog.askstring("Input", f"Enter the name of factor {i+1}:")
            if factor is None:
                raise ValueError("Factor name cannot be empty.")
            factors.append(factor)

            high_level_str = simpledialog.askstring("Input", f"Enter the high level for factor {factor}:")
            if high_level_str is None:
                raise ValueError("High level cannot be empty.")
            high_level = int(high_level_str)

            low_level_str = simpledialog.askstring("Input", f"Enter the low level for factor {factor}:")
            if low_level_str is None:
                raise ValueError("Low level cannot be empty.")
            low_level = int(low_level_str)

            high_levels.append(high_level)
            low_levels.append(low_level)

        num_replicates_str = simpledialog.askstring("Input", "Enter number of replicates:")
        num_replicates = int(num_replicates_str)

        excel_file_name = simpledialog.askstring("Input", "Enter Excel file name (without extension):")

        export_folder = filedialog.askdirectory(title="Select Export Folder")

        return factors, high_levels, low_levels, num_replicates, excel_file_name, export_folder
    
    except ValueError as ve:
        messagebox.showerror("Value Error", str(ve))
        return None, None, None, None, None, None
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None, None, None, None, None

# Main function
def main():
    try:
        # Get user input
        factors, high_levels, low_levels, num_replicates, excel_file_name, export_folder = get_user_input()
        if factors is None or high_levels is None or low_levels is None or num_replicates is None or excel_file_name is None or export_folder is None:
            return

        # Generate factor levels DataFrame
        factor_levels_df = generate_factor_levels_df(factors, high_levels, low_levels)

        # Check if there are at least 3 factors
        if len(factors) < 3:
            raise ValueError("Box-Behnken design requires at least 3 factors.")

        # Create Box-Behnken design DataFrame
        optimization_df = generate_bb_design(factor_levels_df)

        # Duplicate the design DataFrame
        optimization_df_duplicated = pd.concat([optimization_df] * num_replicates, ignore_index=True)

        # Merge the DataFrames
        merged_df = pd.concat([factor_levels_df, pd.DataFrame(columns=['']), optimization_df_duplicated], axis=1)
        merged_df.insert(len(merged_df.columns), 'Results', '')

        # Append '.xlsx' extension if not provided
        if not excel_file_name.endswith('.xlsx'):
            excel_file_name += '.xlsx'

        # Write the merged DataFrame to an Excel file in the export folder
        excel_file_path = os.path.join(export_folder, excel_file_name)
        merged_df.to_excel(excel_file_path, index=False)
        print("Excel file saved successfully.")

    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()

