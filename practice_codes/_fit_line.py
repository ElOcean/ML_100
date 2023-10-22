from matplotlib import pyplot as plt
import numpy as np

def getData():

    # Set the limits for the plot's x and y axes

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    # Collect mouse clicks from the user indefinitely until manually terminated
    # on macOS mouse_add = one click, mouse_pop = double click
    data = []
    while (len(data)< 2):
        plt.text(0.1, 0.1, "Set min 2 points and click enter to plot linear fit")
        data = plt.ginput(n=-1, timeout=-1, show_clicks=True, mouse_add=1, mouse_pop=3)
        
    # Separate x and y coordinates from the collected data
    x, y = zip(*data)
    x_values, y_values = list(x), list(y)
    
    plt.close()

    # Create a new plot for the data points
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    return x_values, y_values

def linfit(x, y): 
    x = np.array(x)
    y = np.array(y)

    # Calculate the coefficients of a linear fit using the formulas
    a = (len(x) * sum(x * y) - (sum(x) * sum(y))) / (len(x) * sum(x * x) - sum(x) * sum(x))
    b = (sum(y) - a * sum(x)) / len(x)

    print(f"a={a}, b={b}")

    return a, b

def main():
    # Collect data points from the user
    x, y = getData()
    
    # linear regression
    a, b = linfit(x, y)

    # Plot the data points and the linear fit
    plt.plot(x, y, 'kx')  # Data points
    xp = np.arange(0.2, 9.7, 0.1)  # Range for the fitted line
    plt.plot(xp, a * xp + b, 'r-')  

    points = len(x)
    plt.text(0.1, 0.1, f'Number of points: {points}')
    plt.text(4, 0.1, f'My fit: a={a:.3f} and b={b:.3f}')
    
    # Display the calculated coefficients in terminal 
    print(f"My fit: a={a:2f} and b={b:2f}")

    # Show the plot to the user
    plt.show()
main()
