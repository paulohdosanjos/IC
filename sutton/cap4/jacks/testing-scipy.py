import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt


def main():
    mu = 3

    fig, ax = plt.subplots(1,1)
    print(poisson.ppf(0.0000001,mu))
    print(poisson.ppf(0.999999,mu))
    x = np.arange(poisson.ppf(0.0000001,mu), poisson.ppf(0.999999,mu),1)
    ax.plot(x,poisson.pmf(x,mu))
    plt.show()


if __name__== "__main__":
    main()

    
