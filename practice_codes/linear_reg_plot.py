from matplotlib import pyplot as plt
import numpy as np

def getData():
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
        

    data = plt.ginput(n=-1,timeout=-1,show_clicks=True, mouse_add=1, mouse_pop=3)

    x,y = zip(*data)
    x_values, y_values = list(x), list(y)
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    #ax.plot(x, y)
    
    return x_values, y_values
       
def linfit(x,y): 
    x = np.array(x)
    y = np.array(y)

    a = (len(x) * sum(x*y) - (sum(x) *sum(y))) / (len(x) * sum(x*x) - sum(x)*sum(x))
    b = (sum(y) - a * sum(x)) / len(x)

    #Testing readymade function for the proof
    #a, b = np.polyfit(x, y, 1)

    print(f"a={a}, b={b}")
    return a,b


def main():
      
    x,y = getData()
    a , b = linfit(x,y)
    plt.plot(x,y,'kx')
    xp = np.arange(0.2 ,9.7,0.1)
    plt.plot(xp,a*xp+b ,'r-')
    print (f"My fit : a={a} and b={b}")

    plt.show()   
main()
