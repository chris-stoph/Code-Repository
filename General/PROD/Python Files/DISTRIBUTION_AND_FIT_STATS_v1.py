#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import math
import traceback

# Global variables
df = None
columns = None
window = None

# Read Excel file
def read_excel_file(file_path):
    """Reads an Excel file and returns a DataFrame."""
    try:
        global df, columns
        df = pd.read_excel(file_path)
        columns = df.columns.tolist()
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

# Function to plot histogram for a selected column
def plot_histogram(column, num_bins=8):
    """Plots histogram for the selected column."""
    try:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True, color='blue', stat='density', bins=num_bins)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.show()
    except Exception as e:
        print(f"Error plotting histogram: {e}")

# Function to plot Q-Q plot for a specified distribution
def plot_qq_plot(data, dist, ax):
    """Plot Q-Q plot for the specified distribution."""
    try:
        if dist == 'weibull_min':
            dist_object = getattr(stats, dist)
            params = dist_object.fit(data, floc=0)
            stats.probplot(data, dist=dist_object, sparams=params[1:], plot=ax)
        else:
            dist_object = getattr(stats, dist)
            stats.probplot(data, dist=dist_object, plot=ax)
        ax.set_title(f'Q-Q Plot for {dist.capitalize()} Distribution')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Ordered Values')
        ax.grid(True)
    except Exception as e:
        traceback.print_exc()



# Function to fit common distributions to the selected column and evaluate goodness-of-fit
def fit_distributions(column):
    """Fits common distributions to the selected column and evaluates goodness-of-fit."""
    try:
        data = df[column].dropna()
        if len(data) == 0:
            print("Error: Selected column has no data.")
            return None, None
        
        distributions = ['norm', 'expon', 'logistic', 'gumbel_l', 'gumbel_r', 'weibull_min']
        results = {}
        for dist_name in distributions:
            if dist_name == 'weibull_min':
                dist = stats.weibull_min
                params = dist.fit(data, floc=0)
            else:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
            result = stats.anderson(data, dist_name)
            crit_values = result.critical_values
            sig_levels = result.significance_level
            results[dist_name] = {'params': params, 'AD_statistic': result.statistic, 'critical_values': crit_values, 'significance_levels': sig_levels}
        return distributions, results
    except Exception as e:
        print(f"Error fitting distributions: {e}")
        traceback.print_exc()
        return None, None


# Main function to create GUI window and handle user interactions
def main():
    try:
    
        global df, window

        window = tk.Tk()
        window.title("Distribution Fitting")

        def open_file_dialog():
            file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
            if file_path:
                df = read_excel_file(file_path)
                if df is not None:
                    show_column_selection()

        file_button = tk.Button(window, text="Select Excel File", command=open_file_dialog)
        file_button.pack(pady=20)

        def show_column_selection():
            for widget in window.winfo_children():
                widget.destroy()

            label = tk.Label(window, text="Select Column:")
            label.pack(pady=10)

            global combo
            combo = ttk.Combobox(window, values=columns)
            combo.pack(pady=10)

            analyze_button = tk.Button(window, text="Analyze", command=analyze_data)
            analyze_button.pack(pady=10)

        def analyze_data():
            selected_column = combo.get()
            descriptive_stats = df[selected_column].describe()
            print("=" * 80)
            print(" " * 30 + f"Descriptive Statistics for {selected_column}")
            print("=" * 80)
            print(descriptive_stats)
            print("\n\n")
            print("The descriptive statistics provide a summary of the central tendency, dispersion, and shape of the distribution of data.")
            print("Here are the key values and their interpretations:")
            print(" ~ count: Number of non-missing observations in the dataset.")
            print(" ~ mean: Average value of the data.")
            print(" ~ std: Standard deviation, which measures the dispersion of data points around the mean.")
            print(" ~ min: Minimum value in the dataset.")
            print(" ~ 25%: Lower quartile or first quartile, representing the value below which 25% of the data fall.")
            print(" ~ 50%: Median or second quartile, representing the middle value of the dataset.")
            print(" ~ 75%: Upper quartile or third quartile, representing the value below which 75% of the data fall.")
            print(" ~ max: Maximum value in the dataset.")
            print("Interpretation: These statistics help understand the distribution of the data and identify any outliers or skewness.")
            print("=" * 80)
            print("\n\n\n")
            plot_histogram(selected_column)

            print("\n\n\n")
            print("=" * 80)
            print("Q-Q Plot".center(80))
            print("=" * 80)
            print("A Q-Q plot (quantile-quantile plot) is a graphical method for comparing two probability distributions by plotting")
            print("their quantiles against each other. The points should approximately lie on the straight line. If they do, it suggests")
            print("that the two distributions are similar.")
            print("=" * 80)
            
            if selected_column:
                selected_data = df[selected_column].dropna()
                distributions, results = fit_distributions(selected_column)
                        
                num_plots = len(distributions)
                num_cols = 3
                num_rows = math.ceil(num_plots / num_cols)

                fig_qq, axes_qq = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
                        
                for i, dist_name in enumerate(distributions):
                    row = i // num_cols
                    col = i % num_cols
                    ax_qq = axes_qq[row, col] if num_plots > 1 else axes_qq

                    if selected_column and dist_name in results:
                        plot_qq_plot(selected_data, dist_name, ax_qq)
                        ax_qq.set_title(f'Q-Q Plot for {dist_name.capitalize()} Distribution')
                        ax_qq.set_xlabel('Theoretical Quantiles')
                        ax_qq.set_ylabel('Ordered Values')
                        ax_qq.grid(True)
                
                plt.tight_layout()
                plt.show()

            print("\n\n\n")
            print("=" * 80)
            print("Goodness-of-Fit (Anderson-Darling)".center(80))
            print("=" * 80)
            

            if results:

                print("{:<15} {:<15} {:<40} {:<40}".format('Distribution', 'AD Statistic', 'Critical Values', 'Significance Levels'))

                for dist_name, result in results.items():
                    crit_values_str = ', '.join([f"{crit_val:.2f}" for crit_val in result['critical_values']])
                    sig_levels_str = ', '.join([f"{sig_level:.2f}" for sig_level in result['significance_levels']])
                    print("{:<15} {:<15f} {:<40} {:<40}".format(dist_name, result['AD_statistic'], crit_values_str, sig_levels_str))
                print("\n")
                print("The Anderson-Darling test measures the goodness-of-fit of a sample to a specified distribution. It evaluates the null hypothesis")
                print("that the data follows the specified distribution. The AD Statistic is compared with critical values at different significance levels.")
                print("If the AD Statistic is greater than the critical value, the null hypothesis is rejected, indicating a poor fit.")
                print("The lower the AD Statistic, the better the fit of the distribution to the data.")
                print("=" * 80)

            else:
                print("No results.")

        window.mainloop()

    except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()

