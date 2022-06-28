import numpy as np
import matplotlib.pyplot as plt
import sys, os

from numpy.core.fromnumeric import std

font = {'family': 'sans',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }

def walk(N, I, Prob):
    """
    Esta función recibe un parámetro N y 
    da las coordenadas de un randomwalk 2D tras I iteraciones

    INPUT:
    N: Número de saltos
    I: Número de iteraciones
    Prob: Array de probabilidades (Orden "N", "S", "O", "E")

    OUTPUT:
    x: ndArray (N,) de las corrdenadas x para el random walk
    y: ndArray (N,) de las corrdenadas y para el random walk
    """
    x_total, y_total = [], []
    for iteracion in range(I):
        x, y = np.zeros(N), np.zeros(N) #Coordenadas para cada salto tras r iteraciones
        for i in range(1, N):
            #Se realiza un salto hacia cualquier dirección con probabilidad dada
            salto = np.random.choice(a=["N", "S", "O", "E"], p=Prob)
            if salto == "N":
                x[i] = x[i-1]
                y[i] = y[i-1]+1
            elif salto == "O":
                x[i] = x[i-1]-1
                y[i] = y[i-1]
            elif salto == "E":
                x[i] = x[i-1]+1
                y[i] = y[i-1]
            elif salto == "S":
                x[i] = x[i-1]
                y[i] = y[i-1]-1
        x_total.append(x)
        y_total.append(y)
    return x_total, y_total

def graficar(x_total, y_total, nm_graph, N, I):
    fig, ax = plt.subplots()
    for x, y in zip(x_total, y_total):
        plt.plot(x, y)
    plt.title("Random Walk: {} saltos; {} iteraciones".format(N, I))
    plt.xlabel("X")
    fig.set_size_inches(10, 10)
    plt.ylabel("Y")
    plt.xticks(np.arange(np.min(x_total), np.max(x_total)+1, 2))
    plt.yticks(np.arange(np.min(y_total), np.max(y_total)+1, 2))
    plt.plot(0, 0, "ko")
    plt.savefig(nm_graph)
    plt.show()

    
def gauss(x, mu, sigma):
    return (1 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-(x-mu)**2 / (2*sigma**2))

def histrogramas(x_total, y_total, nm_graph, N, I):
    mx_total = np.array([x[-1] for x in x_total])
    my_total = np.array([y[-1] for y in y_total])

    x_space = np.linspace(np.min(mx_total), np.max(mx_total), 1000)
    y_space = np.linspace(np.min(my_total), np.max(my_total), 1000)

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
    fig.set_size_inches(7, 6)
    fig.suptitle("Distribuciones de \"x\" e \"y\" comparada a una gaussiana: {} Saltos, {} Iteraciones".format(N, I))
    plt.tight_layout()
    ax1.hist(mx_total, bins=30, density=True, stacked=True, color="blue", edgecolor="black", linewidth=1) 
    ax1.set_xlabel("Distancia (m)")
    ax1.set_ylabel("$P_X(m)$")
    ax1.text(0.1, 0.9,'$\mu$ = {0:.3f} \n$\sigma = ${1:.3f} \n$N = ${2} \n$I = ${3}'.format(np.mean(mx_total), np.std(mx_total), N, I), \
     horizontalalignment='left', \
     verticalalignment='top', \
     transform = ax1.transAxes, 
     fontdict=font) 
    ax1.plot(x_space, gauss(x=x_space, mu=np.mean(mx_total), sigma=np.std(mx_total)),\
         c="r", label="Distribución gaussiana")
    ax1.grid()

    ax2.set_ylabel("$P_Y(m)$")
    ax2.set_xlabel("Distancia (m)")
    ax2.hist(my_total, bins=30, density=True, stacked=True, edgecolor="black", linewidth=1)
    ax2.plot(y_space, gauss(x=y_space, mu=np.mean(my_total), sigma=np.std(my_total)),\
         c="r", label="Distribución gaussiana")
    ax2.text(0.1, 0.9,'$\mu$ = {0} \n$\sigma = ${1:.3f} \n$N = ${2} \n$I = ${3}'.format(np.mean(my_total), np.std(my_total), N, I), \
     horizontalalignment='left', \
     verticalalignment='top', \
     transform = ax2.transAxes,
     fontdict=font)
    fig.set_size_inches(10, 10)
    ax2.grid()

    fig.savefig(nm_graph)
    fig.show()

def main():
    graficar()
    histrogramas()

if __name__ == "__main__":
    N = int(input("Número de saltos: "))
    I = int(input("Número de iteraciones: "))
    main()

