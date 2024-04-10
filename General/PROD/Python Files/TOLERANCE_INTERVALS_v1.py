#!/usr/bin/env python
# coding: utf-8

# In[203]:


import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


# # TWO-SIDED LOOKUP TABLE

# In[204]:


# Define values for confidence level and reliability
confidence_levels = [80, 90, 95]
reliabilities = [80, 90, 95, 99]


two_sided_table = {}

# Define columns for two-sided table
num_samples_column = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 30, 35, 40
]

two_sided_columns = {
    (80, 80): [
        3.133, 2.475, 2.187, 2.022, 1.915, 1.839, 1.783, 1.738,
        1.703, 1.673, 1.649, 1.628, 1.610, 1.594, 1.580, 1.567,
        1.556, 1.546, 1.536, 1.528, 1.520, 1.513, 1.507, 1.480,
        1.460, 1.445
    ],

    (90, 90): [
        5.851, 4.167, 2.586, 3.131, 2.901, 2.742, 2.625, 2.535,
        2.463, 2.404, 2.355, 2.313, 2.277, 2.246, 2.219, 2.194,
        2.172, 2.152, 2.134, 2.118, 2.103, 2.090, 2.077, 2.025,
        1.988, 1.959
    ],
    
    (90, 95): [
        6.972, 4.965, 4.164, 3.730, 3.457, 3.368, 3.128, 3.021, 
        2.935, 2.865, 2.806, 2.757, 2.714, 2.676, 2.644, 2.614, 
        2.588, 2.567, 2.543, 2.524, 2.506, 2.490, 2.475, 2.413, 
        2.368, 2.334
    ],

    (90,99): [
        9.163, 6.525, 5.472, 4.903, 4.543, 4.294, 4.111, 3.970, 
        3.857, 3.765, 3.688, 3.623, 3.566, 3.517, 3.474, 3.436, 
        3.402, 3.371, 3.343, 3.317, 3.294, 3.272, 3.252, 3.171, 
        3.112, 3.067
    ],

    (95, 90): [
        8.380, 5.369, 4.275, 3.712, 3.369, 3.136, 2.967, 2.839, 
        2.737, 2.655, 2.587, 2.529, 2.480, 2.437, 2.400, 2.366, 
        2.337, 2.310, 2.286, 2.264, 2.244, 2.225, 2.208, 2.140,
        2.090, 2.052
    ],

    (95, 95): [
        9.916, 6.370, 5.079, 4.414, 4.007, 3.732, 3.532, 3.379, 
        3.259, 3.162, 3.081, 3.012, 2.954, 2.903, 2.858, 2.819, 
        2.784, 2.752, 2.723, 2.697, 2.673, 2.651, 2.631, 2.549,
        2.490, 2.445
    ]
}

for confidence_level in confidence_levels:
    for reliability in reliabilities:
        if (confidence_level, reliability) not in [(80, 90), (80, 95), (80, 99), (90,80), (95, 80), (95,99)]:  # Exclude specified combinations
            for i, num_samples in enumerate(num_samples_column):
                two_sided_table[(num_samples, confidence_level, reliability)] = two_sided_columns[(confidence_level, reliability)][i]


# # ONE-SIDED LOOKUP TABLE

# In[205]:


one_sided_table = {}

num_samples_column = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 30, 35, 40
]

one_sided_columns = {

    (80, 80): [
        1.710, 1.510, 1.404, 1.336, 1.288, 1.251, 1.222, 1.198,
        1.179, 1.162, 1.147, 1.134, 1.123, 1.113, 1.104, 1.095, 
        1.088, 1.081, 1.074, 1.068, 1.063, 1.058, 1.053, 1.033,
        1.018, 1.006
    ],

    (90, 90): [
        3.868, 2.955, 2.586, 2.378, 2.242, 2.145, 2.071, 2.012, 1.964, 1.924,
        1.891, 1.861, 1.836, 1.813, 1.793, 1.775, 1.759, 1.744, 1.730, 1.717,
        1.706, 1.659, 1.685, 1.644, 1.613, 1.588
    ],

    (90, 95): [
        4.822, 3.668, 3.207, 2.950, 2.783, 2.663, 2.573, 2.503, 2.445, 2.397,
        2.356, 2.321, 2.291, 2.264, 2.240, 2.218, 2.199, 2.181, 2.165, 2.150,
        2.137, 2.124, 2.112, 2.064, 2.027, 1.999
    ],

    (90, 99): [
        6.653, 5.040, 4.401, 4.048, 3.820, 3.659, 3.538, 3.442, 3.365, 3.301,
        3.247, 3.201, 3.160, 3.123, 3.093, 3.065, 3.039, 3.016, 2.994, 2.975,
        2.957, 2.940, 2.925, 2.862, 2.814, 2.777
    ],

    (95, 90): [
        6.158, 4.163, 3.407, 3.006, 2.755, 2.582, 2.454, 2.355, 2.275, 2.21, 
        2.155, 2.108, 2.068, 2.032, 2.001, 1.974, 1.949, 1.926, 1.905, 1.887, 
        1.869, 1.853, 1.838, 1.778, 1.732, 1.697
    ],

    (95, 95): [
        7.655, 5.145, 4.202, 3.707, 3.399, 3.188, 3.031, 2.911, 2.815, 2.736, 
        2.670, 2.614, 2.566, 2.523, 2.486, 2.453, 2.423, 2.396, 2.371, 2.350, 
        2.329, 2.309, 2.292, 2.22, 2.166, 2.126
    ]
}
# Populate the one-sided table dictionary
for confidence_level in confidence_levels:
    for reliability in reliabilities:
        if (confidence_level, reliability) not in [(80, 90), (80, 95), (80, 99), (90,80), (95, 80), (95,99)]:  # Exclude specified combinations
            for i, num_samples in enumerate(num_samples_column):
                one_sided_table[(num_samples, confidence_level, reliability)] = one_sided_columns[(confidence_level, reliability)][i]


# # SAMPLE SIZE CALCULATIONS

# In[206]:


# Function to find the next smallest value in the table
def find_k_smallest_value(table, confidence_level, reliability, k):
    k_next_smallest = None
    min_difference = float('inf')

    for key, value in table.items():
        samples, conf_level, rel = key
        if conf_level == confidence_level and rel == reliability:
            if value < k and k - value < min_difference:
                min_difference = k - value
                k_next_smallest = value

    return k_next_smallest

# Function to find the sample size associated with a k-value
def find_n_for_k(table, value, confidence_level, reliability):
    num_samples = None

    for key, val in table.items():
        samples, conf_level, rel = key
        if val == value and conf_level == confidence_level and rel == reliability:
            num_samples = samples
            break

    return num_samples

# GUI for sample size calculation
def sample_size_calculation_window():
    window = tk.Tk()
    window.title("Sample Size Calculation")

    # Labels and entry fields for input parameters
    tk.Label(window, text="Confidence Level:").grid(row=0, column=0)
    tk.Label(window, text="Reliability:").grid(row=1, column=0)
    tk.Label(window, text="LSL:").grid(row=2, column=0)
    tk.Label(window, text="USL:").grid(row=3, column=0)
    tk.Label(window, text="Sample Mean:").grid(row=4, column=0)
    tk.Label(window, text="Sample Standard Deviation:").grid(row=5, column=0)

    confidence_level_combo = ttk.Combobox(window, values=confidence_levels)
    confidence_level_combo.grid(row=0, column=1)
    confidence_level_combo.current(0)

    reliability_combo = ttk.Combobox(window, values=reliabilities)
    reliability_combo.grid(row=1, column=1)
    reliability_combo.current(0)

    LSL_entry = tk.Entry(window)
    USL_entry = tk.Entry(window)
    sample_mean_entry = tk.Entry(window)
    sample_std_dev_entry = tk.Entry(window)

    LSL_entry.grid(row=2, column=1)
    USL_entry.grid(row=3, column=1)
    sample_mean_entry.grid(row=4, column=1)
    sample_std_dev_entry.grid(row=5, column=1)

    # Function to calculate sample size
    def calculate_sample_size():
        try:
            confidence_level = int(confidence_level_combo.get())
            reliability = int(reliability_combo.get())
            LSL = float(LSL_entry.get())
            USL = float(USL_entry.get())
            sample_mean = float(sample_mean_entry.get())
            sample_std_dev = float(sample_std_dev_entry.get())

            k_LSL = (sample_mean - LSL) / sample_std_dev
            k_USL = (USL - sample_mean) / sample_std_dev

            # Find the num_samples associated with the next smallest value for both LTL and UTL
            k_smallest_two_sided_LSL = find_k_smallest_value(two_sided_table, confidence_level, reliability, k_LSL)
            k_smallest_two_sided_USL = find_k_smallest_value(two_sided_table, confidence_level, reliability, k_USL)
            k_smallest_one_sided_LSL = find_k_smallest_value(one_sided_table, confidence_level, reliability, k_LSL)
            k_smallest_one_sided_USL = find_k_smallest_value(one_sided_table, confidence_level, reliability, k_USL)

            # Find the num_samples associated with the next smallest value
            num_samples_two_sided_LSL = find_n_for_k(two_sided_table, k_smallest_two_sided_LSL, confidence_level, reliability)
            num_samples_two_sided_USL = find_n_for_k(two_sided_table, k_smallest_two_sided_USL, confidence_level, reliability)
            num_samples_one_sided_LSL = find_n_for_k(one_sided_table, k_smallest_one_sided_LSL, confidence_level, reliability)
            num_samples_one_sided_USL = find_n_for_k(one_sided_table, k_smallest_one_sided_USL, confidence_level, reliability)

            # Generate the result text including the k smallest values
            report_text = (
                f"Sample Size Calculation Report\n\n"                
                f"Confidence Level: {confidence_level}\n"
                f"Reliability: {reliability}\n"
                f"LSL: {LSL}\n"
                f"USL: {USL}\n"
                f"Sample Mean: {sample_mean}\n"
                f"Sample Standard Deviation: {sample_std_dev}\n"
                f"k_LSL: {k_LSL}, k_USL: {k_USL}\n"
                f"Two-sided LSL: n = {num_samples_two_sided_LSL} with k = {k_smallest_two_sided_LSL}\n"
                f"Two-sided USL: n = {num_samples_two_sided_USL} with k = {k_smallest_two_sided_USL}\n"
                f"One-sided LSL: n = {num_samples_one_sided_LSL} with k = {k_smallest_one_sided_LSL}\n"
                f"One-sided USL: n = {num_samples_one_sided_USL} with k = {k_smallest_one_sided_USL}"
            )

            # Print the result text in the terminal
            print(report_text)

            # Update the GUI label to indicate the report is printed in the terminal
            result_text = "Results printed in terminal."

        except ValueError:
            result_text = "Error: Please enter valid numeric values for sample size, mean, and standard deviation."

        result_label.config(text=result_text)
        
    tk.Button(window, text="Calculate Sample Size", command=calculate_sample_size).grid(row=6, columnspan=2)

    result_label = tk.Label(window, text="")
    result_label.grid(row=6, columnspan=2)

    window.mainloop()


# # TOLERANCE INTERVAL CALCULATIONS 

# In[207]:


# Function to find the k-value for a given sample size, confidence level, and reliability
def find_k_tolerance_calc(table, num_samples, confidence_level, reliability):
    k_value = None

    for key, value in table.items():
        samples, conf_level, rel = key
        if samples == num_samples and conf_level == confidence_level and rel == reliability:
            k_value = value
            break

    return k_value

# Function to plot tolerance interval
def plot_tolerance_interval(sample_mean, sample_std_dev, k, confidence_level, reliability):
    # Create x values for the plot
    x = np.linspace(sample_mean - 4 * sample_std_dev, sample_mean + 4 * sample_std_dev, 1000)

    # Calculate the probability density function for a normal distribution
    y = (1 / (sample_std_dev * np.sqrt(2 * np.pi))) * np.exp(-(x - sample_mean)**2 / (2 * sample_std_dev**2))

    # Plot the probability density curve
    plt.plot(x, y, label='Probability Density', color='g')

    # Calculate tolerance interval
    lower_limit = sample_mean - k * sample_std_dev
    upper_limit = sample_mean + k * sample_std_dev

    # Plot the tolerance interval
    plt.axvline(x=lower_limit, color='r', linestyle='--', linewidth=2, label='Tolerance Interval')
    plt.axvline(x=upper_limit, color='r', linestyle='--', linewidth=2)

    # Annotate the lower and upper limits with their values
    plt.text(lower_limit-0.3, 0.2, f'Lower Limit: {lower_limit:.2f}', rotation=90, verticalalignment='bottom')
    plt.text(upper_limit+0.3, 0.2, f'Upper Limit: {upper_limit:.2f}', rotation=90, verticalalignment='bottom')

    # Add labels and title
    plt.xlabel('Data')
    plt.ylabel('Probability Density')
    plt.title(f'Tolerance Interval and Probability Density Plot\n(Confidence Level: {confidence_level}, Reliability: {reliability})')
    plt.legend()

    # Show the plot
    plt.show()

# GUI for tolerance interval calculation
def tolerance_interval_calculation_window():
    window = tk.Tk()
    window.title("Tolerance Interval Calculation")

    tk.Label(window, text="Sample Size:").grid(row=0, column=0)
    tk.Label(window, text="Sample Mean:").grid(row=1, column=0)
    tk.Label(window, text="Sample Standard Deviation:").grid(row=2, column=0)
    tk.Label(window, text="Confidence Level:").grid(row=3, column=0)
    tk.Label(window, text="Reliability:").grid(row=4, column=0)

    sample_size_entry = tk.Entry(window)
    sample_mean_entry = tk.Entry(window)
    sample_std_dev_entry = tk.Entry(window)

    sample_size_entry.grid(row=0, column=1)
    sample_mean_entry.grid(row=1, column=1)
    sample_std_dev_entry.grid(row=2, column=1)

    confidence_level_combo = ttk.Combobox(window, values=[80, 90, 95])
    confidence_level_combo.grid(row=3, column=1)
    confidence_level_combo.current(0)

    reliability_combo = ttk.Combobox(window, values=[80, 90, 95, 99])
    reliability_combo.grid(row=4, column=1)
    reliability_combo.current(0)

    def calculate_tolerance_interval():
        try:
            num_samples = int(sample_size_entry.get())
            sample_mean = float(sample_mean_entry.get())
            sample_std_dev = float(sample_std_dev_entry.get())
            confidence_level = int(confidence_level_combo.get())
            reliability = int(reliability_combo.get())

            k = find_k_tolerance_calc(two_sided_table, num_samples, confidence_level, reliability)

            if k is None:
                result_text = "Error: Could not find k-value in the table."
            else:
                LTL = sample_mean - k * sample_std_dev
                UTL = sample_mean + k * sample_std_dev

                # Generate the plot
                plot_tolerance_interval(sample_mean, sample_std_dev, k, confidence_level, reliability)

                # Generate the report text
                report_text = (
                    f"Tolerance Interval Calculation Report\n\n"
                    f"Sample Size: {num_samples}\n"
                    f"Sample Mean: {sample_mean}\n"
                    f"Sample Standard Deviation: {sample_std_dev}\n"
                    f"Confidence Level: {confidence_level}\n"
                    f"Reliability: {reliability}\n"
                    f"k value: {k}\n"
                    f"Lower Tolerance Limit (LTL): {LTL}\n"
                    f"Upper Tolerance Limit (UTL): {UTL}"
                )
                # Print the report text in the terminal
                print(report_text)

                # Update the GUI label to indicate the report is printed in the terminal
                result_text = "Results printed in terminal."

        except ValueError:
            result_text = "Error: Please enter valid numeric values for sample size, mean, and standard deviation."

        result_label.config(text=result_text)

    tk.Button(window, text="Calculate Tolerance Interval", command=calculate_tolerance_interval).grid(row=5, columnspan=2)

    result_label = tk.Label(window, text="")
    result_label.grid(row=6, columnspan=2)

    window.mainloop()


# In[208]:


# Main function
def main():
    window = tk.Tk()
    window.title("Tolerance Interval and Sample Size Calculator")

    tk.Button(window, text="Open Tolerance Interval Calculator", command=tolerance_interval_calculation_window).pack(pady=10)
    tk.Button(window, text="Open Sample Size Calculator", command=sample_size_calculation_window).pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    main()

