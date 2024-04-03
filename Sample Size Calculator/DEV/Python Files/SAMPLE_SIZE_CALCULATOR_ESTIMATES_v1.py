#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
from tkinter import ttk
import math
import statsmodels.stats.power as smp
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

def calculate_sample_size(confidence_level, margin_of_error, data_type, value, test_type):
    z_score = {
        "80%": 1.2816,
        "85%": 1.4408,
        "90%": 1.6449,
        "95%": 1.9600,
        "98%": 2.3263,
        "99%": 2.5758
    }

    z = z_score.get(confidence_level)

    if z is None:
        print(f"Confidence level '{confidence_level}' not supported.")
        return None, None  # Return None for both values
    
    if data_type == 'Continuous':
        # Calculate sample size for continuous data
        sample_size = ((z ** 2 * value ** 2) / margin_of_error ** 2)
        method_description = "Sample size calculation for Continuous data using normal distribution and known standard deviation."

    elif data_type == 'Discrete':
        if test_type == 'One-sided':
            # Adjust z-score for one-sided test
            if confidence_level == "80%":
                z = 0.8416  # One-sided Z-score for 80% confidence
            elif confidence_level == "85%":
                z = 1.0364  # One-sided Z-score for 85% confidence
            elif confidence_level == "90%":
                z = 1.2816  # One-sided Z-score for 90% confidence
            elif confidence_level == "95%":
                z = 1.6449  # One-sided Z-score for 95% confidence
            elif confidence_level == "98%":
                z = 2.0537  # One-sided Z-score for 98% confidence
            elif confidence_level == "99%":
                z = 2.3263  # One-sided Z-score for 99% confidence
            else:
                print(f"Confidence level '{confidence_level}' not supported for a one-sided test.")
                return None, None
        elif test_type == 'Two-sided':
            z = abs(z)  # Absolute value for two-sided test
        
        # Calculate sample size for proportion estimation (discrete data)
        sample_size = (z**2 * value * (1 - value)) / (margin_of_error**2)
        method_description = f"Sample size calculation for {'One-sided' if test_type == 'One-sided' else 'Two-sided'} Discrete data using normal approximation to binomial distribution and known proportion."

    else:
        print("Invalid data type. Supported values are 'Continuous' and 'Discrete'.")
        return None, None

    return math.ceil(sample_size), method_description  # Return the calculated values

    
def generate_report(confidence_level, margin_of_error, data_type, value, test_type, sample_size, method_description):
    if sample_size is None or method_description is None:
        print("Error occurred during sample size calculation.")
        return

    report = f"Sample Size Calculation Report\n\n"
    report += f"Confidence Level: {confidence_level}\n"
    report += f"Margin of Error: {margin_of_error}\n"
    report += f"Data Type: {data_type}\n"
    if data_type == 'Continuous':
        report += f"Population Standard Deviation: {value}\n"
    elif data_type == 'Discrete':
        report += f"Expected Proportion: {value}\n"
    report += f"Test Type: {test_type}\n"
    report += f"Sample Size (Estimates): {sample_size}\n"
    report += f"Method Description:\n{method_description}\n"

    print(report)  # Print the report in the terminal

def update_fields_state(event=None):  # Added event=None to handle combobox event
    data_type = data_combobox.get()
    if data_type == 'Continuous':
        std_dev_entry.config(state='normal')
        proportion_entry.config(state='disabled')
    elif data_type == 'Discrete':
        std_dev_entry.config(state='disabled')
        proportion_entry.config(state='normal')
    else:
        std_dev_entry.config(state='disabled')
        proportion_entry.config(state='disabled')
    
def calculate_button_clicked():
    try:
        conf_level = confidence_combobox.get()
        margin_err = float(margin_entry.get())
        data_type = data_combobox.get()
        if data_type == 'Continuous':
            value = float(std_dev_entry.get())
        elif data_type == 'Discrete':
            value = float(proportion_entry.get())
        else:
            raise ValueError("Invalid data type. Supported values are 'Continuous' and 'Discrete'.")

        test_type = test_combobox.get()
        
        sample_size, method_description = calculate_sample_size(conf_level, margin_err, data_type, value, test_type)

        # Generate the report
        generate_report(conf_level, margin_err, data_type, value, test_type, sample_size, method_description)

        result_label.config(text="Report printed in terminal!")
    except ValueError as e:
        result_label.config(text=f"Error: {e}")

# Create main window
root = tk.Tk()
root.title("Sample Size Calculator")

# Create and place GUI components
ttk.Label(root, text="Confidence Level:").pack()
confidence_combobox = ttk.Combobox(root, values=["80%", "85%", "90%", "95%", "98%", "99%"])
confidence_combobox.pack()
confidence_combobox.set("95%")  # Default value

ttk.Label(root, text="Margin of Error:").pack()
margin_entry = ttk.Entry(root)
margin_entry.pack()
margin_entry.insert(0, "0.05")  # Default value

ttk.Label(root, text="Data Type:").pack()
data_combobox = ttk.Combobox(root, values=["", "Continuous", "Discrete"])
data_combobox.pack()
data_combobox.set("")  # Default value
data_combobox.bind("<<ComboboxSelected>>", update_fields_state)  # Bind event to update field state

ttk.Label(root, text="Population Standard Deviation (for Continuous):").pack()
std_dev_entry = ttk.Entry(root, state='disabled')
std_dev_entry.pack()

ttk.Label(root, text="Expected Proportion (for Discrete):").pack()
proportion_entry = ttk.Entry(root, state='disabled')
proportion_entry.pack()

ttk.Label(root, text="Test Type:").pack()
test_combobox = ttk.Combobox(root, values=["One-sided", "Two-sided"])
test_combobox.pack()
test_combobox.set("One-sided")  # Default value

calculate_button = ttk.Button(root, text="Calculate Sample Size", command=calculate_button_clicked)
calculate_button.pack()

result_label = ttk.Label(root, text="")
result_label.pack()

root.mainloop()

