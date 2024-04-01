import matplotlib.pyplot as plt

# Defining the data with updated colors
labels_group1 = ['a', 'b', 'c']
x_group1 = [50, 60, 30]
y_group1 = [5.2, 7, 2.8]
sizes_group1 = [240, 120, 40]  # Example sizes for visibility
colors_group1 = 'red'  # Changed to red

labels_group2 = ['d', 'e', 'f']
x_group2 = [10, 5, 90]
y_group2 = [3, 0.8, 12]
sizes_group2 = [40, 80, 120]  # Example sizes for visibility
colors_group2 = 'purple'  # Changed to purple

# Creating the plot
plt.figure(figsize=(10, 6))

# Plotting Group 1 and its labels
for i in range(len(x_group1)):
    plt.scatter(x_group1[i], y_group1[i], s=sizes_group1[i], c=colors_group1, alpha=0.6, label='group_1' if i==0 else "")
    plt.text(x_group1[i], y_group1[i], labels_group1[i], ha='right', va='bottom')

# Plotting Group 2 and its labels
for i in range(len(x_group2)):
    plt.scatter(x_group2[i], y_group2[i], s=sizes_group2[i], c=colors_group2, alpha=0.6, label='group_2' if i==0 else "")
    plt.text(x_group2[i], y_group2[i], labels_group2[i], ha='right', va='bottom')

# Setting the plot limits to start from 0,0 and adding grid
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Adding labels, title, and customizing legend
plt.xlabel('%')
plt.ylabel('%/day')
plt.title('Scatter Plot with Labels and Grid')
plt.legend(title='Group Legend')

# Display the plot
plt.show()
