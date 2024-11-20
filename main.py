import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# Standard method functions

def convert_base(number, from_base, to_base):
    """
    Converts a number from a given base to another base.
    """
    # Convert the number to decimal (base 10)
    decimal_number = int(number, from_base)
    if to_base == 10:
        return str(decimal_number)

    # Convert from decimal to the target base
    converted_number = ""
    while decimal_number > 0:
        remainder = decimal_number % to_base
        if remainder < 10:
            converted_number = chr(48 + remainder) + converted_number
        else:
            converted_number = chr(87 + remainder) + converted_number
        decimal_number = decimal_number // to_base
    return converted_number or "0"

# Reasearched method functions

# Initialize the factorial cache for efficient computation
factorial_cache = {0: 1}

def get_factorial(n):
    """
    Returns the factorial of n, caching it if not already cached.
    """
    if n in factorial_cache:
        return factorial_cache[n]
    else:
        factorial_cache[n] = n * get_factorial(n - 1)
        return factorial_cache[n]

def permutation(n, r):
    """
    Calculate nPr (permutations of n items taken r at a time) without using factorials.
    """
    if r > n:
        return 0
    perm = 1
    for i in range(n, n - r, -1):
        perm *= i
    return perm

def convert_to_digits(original_base, number):
    """
    Converts a number in the original base to a list of its digits in base 10.
    """
    digits = []
    for char in number:
        if '0' <= char <= '9':
            digit = int(char)
        else:
            digit = ord(char.upper()) - ord('A') + 10  # Convert letters A-Z to 10-35
        if digit < 0 or digit >= original_base:
            raise ValueError(f"Digit '{char}' is not valid for base {original_base}")
        digits.append(digit)
    return digits

def calculate_glist(digit_list, original_base, target_base):
    """
    Calculates the GList based on the digit list, original base, and target base.
    """
    x = len(digit_list)
    q = original_base // target_base
    r = original_base % target_base
    GList = []
    for k in range(x):
        Beginning = (q ** k) / get_factorial(k)
        Temp = 0
        for i in range(x - k):
            Permutation = permutation(x - i - 1, k)
            Term = digit_list[i] * Permutation * (r ** ((x - k) - (i + 1)))
            Temp += Term
        GList.append(int(Temp * Beginning))
    return GList

def convert_from_base10(num, target_base):
    """
    Converts a base 10 number to the target base.
    """
    if num == 0:
        return '0'
    digits = []
    while num > 0:
        remainder = num % target_base
        if remainder < 10:
            digits.append(str(remainder))
        else:
            digits.append(chr(remainder - 10 + ord('A')))  # Convert to A-Z for bases > 10
        num //= target_base
    digits.reverse()
    return ''.join(digits)

def add_trailing_zeros(clist):
    """
    Adds trailing zeros to each element of the list based on its position.
    """
    c_list = []
    for index, item in enumerate(clist):
        zeros_to_add = index  # Add zeros based on the current index
        new_item = item + '0' * zeros_to_add
        c_list.append(new_item)
    return c_list

def add_elements_in_target_base(clist, target_base):
    """
    Add all elements in the list directly in the target base and return the sum in the target base.
    """
    result = []
    carry = 0
    max_length = max(len(num) for num in clist)
    padded_clist = [num.zfill(max_length) for num in clist]
    for i in range(max_length - 1, -1, -1):
        digit_sum = carry
        for num in padded_clist:
            digit = num[i]
            if '0' <= digit <= '9':
                digit_value = int(digit)
            else:
                digit_value = ord(digit.upper()) - ord('A') + 10
            digit_sum += digit_value
        new_digit_value = digit_sum % target_base
        carry = digit_sum // target_base
        if new_digit_value < 10:
            result.append(str(new_digit_value))
        else:
            result.append(chr(new_digit_value - 10 + ord('A')))
    while carry > 0:
        new_digit_value = carry % target_base
        carry //= target_base
        if new_digit_value < 10:
            result.append(str(new_digit_value))
        else:
            result.append(chr(new_digit_value - 10 + ord('A')))
    result.reverse()
    return ''.join(result)

# Statistical Functions
def generate_test_values(threshold):
    """
    Prepares a list of test values as powers of 10 up to a specified threshold in base 10
    """
    test_values = []
    power = 0
    while 10**power <= threshold:
        base_10_value = 10**power
        test_values.append(base_10_value)
        power += 1
    return test_values

def remove_outliers(data):
    """
    Removes outliers that are more than 3 standard deviations away from the mean.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = 3 * std_dev
    return [x for x in data if abs(x - mean) <= threshold]

def cohen_d(x, y):
   x = np.array(x)
   y = np.array(y)
   mean_x = np.mean(x)
   mean_y = np.mean(y)
   var_x = np.var(x, ddof=1)
   var_y = np.var(y, ddof=1)
   n_x = len(x)
   n_y = len(y)
   pooled_std = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
   return (mean_x - mean_y) / pooled_std

# Main function
def main():
    """
    Main function to compare base conversion methods and plot their performance.
    """
    try:
        # Get user input for base conversion
        original_base = int(input("Enter the original base (2-36): "))
        target_base = int(input("Enter the target base (2-36): "))

        # Validate the bases
        if original_base < 2 or target_base < 2 or original_base > 36 or target_base > 36:
            print("Base must be between 2 and 36.")
            return

        # Define a high threshold for test values
        threshold = 10 ** 150
        test_values_base_10 = generate_test_values(threshold)

        # Standard method execution
        test_values_in_original_base = [convert_base(str(value), 10, original_base) for value in test_values_base_10]
        execution_times_standard = []

        for number_to_convert in test_values_in_original_base:
            print(f"\nTesting conversion for number: {number_to_convert} (Base {original_base} to Base {target_base})")
            try:
                num_iterations = 1000
                warmup_iterations = 900
                all_times = []

                # Measure execution time for base conversion
                for i in range(num_iterations):
                    iteration_start_time = time.perf_counter()
                    result = convert_base(number_to_convert, original_base, target_base)
                    iteration_end_time = time.perf_counter()
                    execution_time = (iteration_end_time - iteration_start_time) * 1_000_000  # Convert to microseconds

                    # Collect times after warm-up phase
                    if i >= warmup_iterations:
                        all_times.append(execution_time)

                # Remove outliers and compute average execution time
                filtered_times = remove_outliers(all_times)
                average_execution_time = np.mean(filtered_times)
                print(f"Converted Result: {result}")
                print(f"Average Execution Time (after removing outliers): {average_execution_time:.4f} microseconds")
                execution_times_standard.append(average_execution_time)

            except ValueError as e:
                print(e)

        # Reasearched method execution
        test_values = [convert_from_base10(value, original_base) for value in test_values_base_10]
        execution_times_researched = []

        for number_to_convert in test_values:
            print(f"\nTesting conversion for number: {number_to_convert} (Base {original_base} to Base {target_base})")
            try:
                num_iterations = 10
                warmup_iterations = 5
                all_times = []

                # Measure execution time for researched method
                for i in range(num_iterations):
                    start_time = time.perf_counter()

                    digit_list = convert_to_digits(original_base, number_to_convert)
                    GList = calculate_glist(digit_list, original_base, target_base)
                    Clist = [convert_from_base10(num, target_base) for num in GList]
                    clist = add_trailing_zeros(Clist)
                    final_result = add_elements_in_target_base(clist, target_base)

                    end_time = time.perf_counter()
                    execution_time = (end_time - start_time) * 1_000_000  # Convert to microseconds

                    # Record the time if it's after the warm-up phase
                    if i >= warmup_iterations:
                        all_times.append(execution_time)

                # Remove outliers and compute average execution time
                filtered_times = remove_outliers(all_times)
                average_execution_time = np.mean(filtered_times)
                print(f"Final Result: {final_result}")
                print(f"Average Execution Time (after removing outliers): {average_execution_time:.2f} μs")
                execution_times_researched.append(average_execution_time)

            except ValueError as e:
                print(e)

        # Convert lists to NumPy arrays for fitting
        test_values_base_10 = np.array(test_values_base_10, dtype=float)
        execution_times_standard = np.array(execution_times_standard, dtype=float)
        execution_times_researched = np.array(execution_times_researched, dtype=float)

        # Prepare data for curve fitting
        log_test_values = np.log10(test_values_base_10)
        log_execution_times_standard = np.log10(execution_times_standard)
        log_execution_times_researched = np.log10(execution_times_researched)

        # Fit a polynomial curve
        poly_degree = 4
        coeffs_standard = np.polyfit(log_test_values, log_execution_times_standard, poly_degree)
        coeffs_researched = np.polyfit(log_test_values, log_execution_times_researched, poly_degree)

        # Generate polynomial curves
        poly_standard = np.poly1d(coeffs_standard)
        poly_researched = np.poly1d(coeffs_researched)

        # Calculate the difference polynomial
        coeffs_difference = coeffs_researched - coeffs_standard
        poly_difference = np.poly1d(coeffs_difference)

        # Generate values for the curve
        x_curve = np.linspace(log_test_values.min(), log_test_values.max(), 100)
        y_curve_standard = poly_standard(x_curve)
        y_curve_researched = poly_researched(x_curve)
        y_curve_difference = poly_difference(x_curve)

        # Calculate R^2 values
        r2_standard = r2_score(log_execution_times_standard, poly_standard(log_test_values))
        r2_researched = r2_score(log_execution_times_researched, poly_researched(log_test_values))

        # Plot the results for both methods
        plt.figure(figsize=(12, 8))

        # Plot standard method
        plt.scatter(test_values_base_10, execution_times_standard, color='blue', label='Standard Method', marker='o')

        # Plot researched method
        plt.scatter(test_values_base_10, execution_times_researched, color='red', label='Researched Method', marker='x')

        # Plot best fit curves
        plt.plot(10 ** x_curve, 10 ** y_curve_standard, color='blue', linestyle='--', label='Standard Method Fit')
        plt.plot(10 ** x_curve, 10 ** y_curve_researched, color='red', linestyle='--', label='Researched Method Fit')

        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Input Value')
        plt.ylabel('Execution Time (microseconds)')
        plt.title('Base Conversion Execution Time Comparison for Base ' + str(original_base) + ' to ' + str(target_base))

        # Custom x-axis and y-axis limits
        plt.xlim(1e0, float(threshold * 100))
        plt.ylim(0.1, max(max(execution_times_standard), max(execution_times_researched)) * 1.5)

        # Formatter for scientific notation on x-axis and y-axis
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()

        # Add equation text annotations
        def format_polynomial(coeffs):
            terms = []
            degree = len(coeffs) - 1
            for i, coeff in enumerate(coeffs):
                power = degree - i
                if coeff == 0:
                    continue
                coeff_str = f"{coeff:.2e}"
                if power == 0:
                    terms.append(f"{coeff_str}")
                elif power == 1:
                    terms.append(f"{coeff_str}x")
                else:
                    terms.append(f"{coeff_str}x^{power}")
            return " + ".join(terms)

        equation_standard = f"Standard Fit: y = {format_polynomial(coeffs_standard)}\n$R^2 = {r2_standard:.2f}$"
        equation_researched = f"Researched Fit: y = {format_polynomial(coeffs_researched)}\n$R^2 = {r2_researched:.2f}$"
        plt.text(0.05, 0.95, equation_standard, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 color='blue')
        plt.text(0.05, 0.90, equation_researched, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 color='red')

        plt.tight_layout()
        plt.show()

        # Plot the difference polynomial
        plt.figure(figsize=(12, 8))

        # Plot difference polynomial curve
        plt.plot(10 ** x_curve, 10 ** y_curve_difference, color='green', linestyle='-',
                 label='Difference (Researched - Standard)')

        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Input Value')
        plt.ylabel('Difference in Execution Time (microseconds)')
        plt.title('Difference in Execution Time Between Researched and Standard Methods for Base ' + str(
            original_base) + ' to ' + str(target_base))

        # Custom x-axis and y-axis limits
        plt.xlim(1e0, float(threshold*100))
        plt.ylim(0.1, 10 ** (np.ceil(np.log10(max(10 ** y_curve_difference)))))

        # Formatter for scientific notation on x-axis and y-axis
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()

        # Add equation text annotation for the difference polynomial
        equation_difference = f"Difference Fit: y = {format_polynomial(coeffs_difference)}"
        plt.text(0.05, 0.95, equation_difference, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 color='green')

        plt.tight_layout()
        plt.show()

        # Bar plot for power of 10 ranges
        bins = [10 ** i for i in
                range(-1, int(np.log10(max(max(execution_times_standard), max(execution_times_researched))) + 2))]
        labels = [f"10^{i} ~ 10^{i + 1}" for i in
                  range(-1, int(np.log10(max(max(execution_times_standard), max(execution_times_researched))) + 1))]

        counts_standard, _ = np.histogram(execution_times_standard, bins=bins)
        counts_researched, _ = np.histogram(execution_times_researched, bins=bins)

        # Create bar positions and width
        bar_width = 0.35
        bar_positions = np.arange(len(labels))

        plt.figure(figsize=(14, 8))

        # Plot bars for standard method
        plt.bar(bar_positions - bar_width / 2, counts_standard, bar_width, label='Standard Method', color='blue',
                edgecolor='black')

        # Plot bars for researched method
        plt.bar(bar_positions + bar_width / 2, counts_researched, bar_width, label='Researched Method', color='red',
                edgecolor='black')

        # Bar plot for power of 10 ranges
        bins = [10 ** i for i in range(-1, int(np.log10(max(max(execution_times_standard), max(execution_times_researched))) + 2))]
        labels = [f"10^{i} ~ 10^{i + 1}" for i in range(-1, int(np.log10(max(max(execution_times_standard), max(execution_times_researched))) + 1))]

        plt.yscale('log')
        plt.xlabel('Execution Time Range (microseconds)')
        plt.ylabel('Number of Test Values')
        plt.title('Number of Test Values in Execution Time Ranges for Base ' + str(original_base) + ' to ' + str(target_base))
        plt.xticks(bar_positions, labels)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

        # Perform statistical tests to compare execution times
        t_statistic, p_value_ttest = stats.ttest_ind(execution_times_standard, execution_times_researched,
                                                     equal_var=False)
        u_statistic, p_value_mannwhitney = stats.mannwhitneyu(execution_times_standard, execution_times_researched,
                                                              alternative='two-sided')

        # Print statistical analysis results
        print("\nStatistical Analysis:")
        print(f"T-Test P-Value: {p_value_ttest:.50f}")
        print(f"Mann-Whitney U Test P-Value: {p_value_mannwhitney:.50f}")

        # Calculate and Print Cohen's d effect size
        effect_size = cohen_d(execution_times_standard, execution_times_researched)
        print(f"\nCohen's d effect size: {effect_size:.4f}")



    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()