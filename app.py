import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Create and train the model
model = LinearRegression()
model.fit(x, y)

# Make predictions
x_new = np.array([[0], [2]])
y_predict = model.predict(x_new)

# Plot the results
plt.figure(figsize=(4, 2), facecolor='lightgray')  # Set figure size and background color
plt.scatter(x, y)
plt.plot(x_new, y_predict, color='red', linewidth=2)
plt.title('Linear Regression Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()

# Save the plot as an image
plt.savefig('linear_regression_plot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot to free memory
