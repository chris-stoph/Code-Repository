#!/usr/bin/env python
# coding: utf-8

# ## IMPORT LIBRARIES

# In[1]:


# Import necessary libraries
# Import necessary libraries
import pandas as pd
from tkinter import Tk, simpledialog, filedialog, messagebox  # Import necessary GUI modules
import os  # Import operating system module
from pyDOE2 import pbdesign  # Assuming you have a library named pbdesign for Plackett-Burman design


# ## CREATE SCREENING DESIGN PLAN

# In[2]:


# Function to generate a DataFrame with factor names, high levels, and low levels
def generate_factors_df(factors, high_levels, low_levels):
    """
    Generate a DataFrame with factor names, high levels, and low levels.

    Args:
    - factors (list): List of factor names.
    - high_levels (list): List of high levels for each factor.
    - low_levels (list): List of low levels for each factor.

    Returns:
    - df (DataFrame): DataFrame representing factor names, high levels, and low levels.
    """
    # Create a dictionary from the lists of factors, high levels, and low levels
    data = {'Factor': factors, 'High(+1)': high_levels, 'Low(-1)': low_levels}
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)
    return df  # Return the DataFrame

def create_plackett_burman_design(factors_df):
    try:
        # Extract the list of factors from the DataFrame
        factors = factors_df['Factor'].tolist()
        
        # Generate Plackett-Burman design matrix
        design = pbdesign(len(factors))
        
        # Convert the design matrix to a DataFrame with column names as factors
        df = pd.DataFrame(design, columns=factors)
        
        # Create mapped columns for each factor and leave them blank
        for factor in factors:
            df[f'{factor}_mapped'] = ''
        
        return df
    except Exception as e:
        raise Exception("Error occurred while generating Plackett-Burman design: " + str(e))

# Function to prompt user for input
def get_user_input():
    try:
        root = Tk()  # Create a Tkinter root window
        root.withdraw()  # Hide the root window

        # Prompt user for the number of factors
        num_factors = simpledialog.askinteger("Input", "Enter the number of factors:")
        if num_factors is None:
            raise ValueError("Number of factors cannot be empty.")

        factors = []
        high_levels = []
        low_levels = []

        # Prompt user to enter factor names, high levels, and low levels
        for i in range(num_factors):
            factor = simpledialog.askstring("Input", f"Enter the name of factor {i+1}:")
            if factor is None:
                raise ValueError("Factor name cannot be empty.")
            factors.append(factor)

            high_level_str = simpledialog.askstring("Input", f"Enter the high level for factor {factor}:")
            if high_level_str is None:
                raise ValueError("High level cannot be empty.")
            high_level = int(high_level_str)

            while True:
                low_level_str = simpledialog.askstring("Input", f"Enter the low level for factor {factor}:")
                if low_level_str is None:
                    raise ValueError("Low level cannot be empty.")
                low_level = int(low_level_str)
                if low_level <= high_level:
                    break
                else:
                    messagebox.showerror("Error", "Low level cannot be larger than high level. Please enter a valid low level.")

            high_levels.append(high_level)
            low_levels.append(low_level)

        # Prompt user to enter the number of replicates, Excel file name, and export folder
        num_replicates = simpledialog.askinteger("Input", "Enter the number of replicates:")
        if num_replicates is None:
            raise ValueError("Number of replicates cannot be empty.")
        excel_file_name = simpledialog.askstring("Input", "Enter the Excel file name:")
        if excel_file_name is None:
            raise ValueError("Excel file name cannot be empty.")
        export_folder = filedialog.askdirectory(title="Select Export Folder")
        if not export_folder:
            raise ValueError("Export folder cannot be empty.")

        return factors, high_levels, low_levels, num_replicates, excel_file_name, export_folder
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
        return None, None, None, None, None, None

# Main function
def main():
    try:
        factors, high_levels, low_levels, num_replicates, excel_file_name, export_folder = get_user_input()

        if factors is None:
            return

        factors_df = generate_factors_df(factors, high_levels, low_levels)

        mapped_df = create_plackett_burman_design(factors_df)

        merged_df = pd.concat([factors_df, pd.DataFrame(columns=['']), mapped_df], axis=1)
        merged_df.insert(len(merged_df.columns), 'Results', '')

        if not excel_file_name.endswith('.xlsx'):
            excel_file_name += '.xlsx'

        excel_file_path = os.path.join(export_folder, excel_file_name)
        merged_df.to_excel(excel_file_path, index=False)
        print("Excel file saved successfully.")

    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()

