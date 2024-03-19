#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from tabulate import tabulate
from tkinter import Tk, filedialog, simpledialog, messagebox
from sklearn.preprocessing import PolynomialFeatures


# # SET PLOT APPEARANCE

# In[ ]:


# Set font sizes
title_fontsize = 16
label_fontsize = 14
legend_fontsize = 14
tick_label_fontsize = 12


# # IMPORT DATA AND PERFORM TEST

# In[ ]:


def read_excel_file(file_path):
    """Reads an Excel file and returns a DataFrame."""
    try:
        df = pd.read_excel(file_path)
        df = df.dropna(axis=1, how='all')
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def get_file_path():
    """Open file dialog and return selected file path."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
    root.destroy()  # Close the Tkinter window
    return file_path

def get_user_column_selection(df, title):
    """Gets user input for selecting a column using dialog boxes."""
    columns = df.columns.tolist()
    root = Tk()
    root.withdraw()
    col_name = simpledialog.askstring(title, f"Enter the column name for {title}:\nAvailable columns: {', '.join(columns)}\n")
    root.destroy()  # Close the Tkinter window
    return col_name

def fit_linear_regression(df, X_cols, y_col):
    """Fit a multiple linear regression model."""
    X = df[X_cols]
    X = sm.add_constant(X)  # Add constant for intercept
    y = df[y_col]

    model = sm.OLS(y, X).fit()
    return model

def plot_actual_vs_predicted(model, X, y, title):
    """Plot actual vs predicted values."""
    y_pred = model.predict(sm.add_constant(X))  # Calculate predicted values

    fig, ax = plt.subplots()
    ax.scatter(y_pred, y, color='blue', zorder=2)  # Set z-order for markers
    ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, zorder=1)  # Solid red line for reference
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    # Add major gridlines
    ax.grid(True, which='major', zorder=0)  # Only major gridlines
    ax.minorticks_on()

    plt.show()

def plot_residuals_vs_predicted(model, X, y, title):
    """Plot residuals vs predicted values."""
    y_pred = model.predict(sm.add_constant(X))  # Calculate predicted values
    residuals = y - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, color='green', marker='o', zorder=2)  # Set z-order for markers
    ax.axhline(y=0, color='red', linestyle='--', zorder=1)  # Add horizontal line at y=0
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residuals')
    ax.set_title(title)

    plt.show()

def main():
    file_path = get_file_path()
    if not file_path:
        print("No file selected.")
        return

    df = read_excel_file(file_path)
    if df is None:
        print("Error reading Excel file.")
        return

    # Get the number of independent variables from user
    num_vars = simpledialog.askinteger("Number of Independent Variables", "Enter the number of independent variables:")
    if num_vars is None or num_vars <= 0:
        print("Invalid number of independent variables.")
        return

    x_cols = []  # List to store selected independent variables
    for _ in range(num_vars):
        x_col = get_user_column_selection(df, "Independent Variable (X)")
        if x_col is None:
            print("Error: Invalid input for independent variable.")
            return
        x_cols.append(x_col)

    y_col = get_user_column_selection(df, "Dependent Variable (Y)")
    if y_col is None:
        print("Error: Invalid input for dependent variable.")
        return

    linear_model = fit_linear_regression(df, x_cols, y_col)

    # Print model summary
    print(linear_model.summary())

    X = df[x_cols]
    y = df[y_col]

    # Plot actual vs predicted
    plot_actual_vs_predicted(linear_model, X, y, "Actual vs Predicted")

    # Plot residuals vs predicted
    plot_residuals_vs_predicted(linear_model, X, y, "Residuals vs Predicted")

if __name__ == "__main__":
    main()

