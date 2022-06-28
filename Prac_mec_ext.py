import numpy as np
import matplotlib.pyplot as plt
import funcionesmec as mecest


N=100
I=5000
datos_x, datos_y = mecest.walk(N=100, I=5000, Prob=[0.25, 0.25, 0.25, 0.25])
mecest.graficar(datos_x, datos_y, nm_graph="N100I5000", N=N, I=I)


for N in [100, 1000, 2000]:
    datos_x, datos_y = mecest.walk(N=N, I=5000, Prob=[0.25, 0.25, 0.25, 0.25])
    nombre = "grafica_I5000_N{}".format(N) 
    mecest.histrogramas(datos_x, datos_y, nm_graph=nombre, N=N, I=I)


sigmas_x = np.zeros(500)
sigmas_y = np.zeros(500)
for n in range(1, 501):
    datos_x, datos_y = mecest.walk(N=n, I=1000, Prob=[0.25, 0.25, 0.25, 0.25])
    print(n)
    mx_total = np.array([x[-1] for x in datos_x])
    my_total = np.array([y[-1] for y in datos_y])
    sigmas_x[n-1] = np.std(mx_total)
    sigmas_y[n-1] = np.std(my_total)

def curva1(N):
    return 2*np.sqrt( (N/2) * (1/2) * (1/2) ) # P = 1/2, Q = 1/2, N' = N/2

print("OK")
lisp = np.linspace(1, 501, 2500)

fig, ax = plt.subplots()
ax.scatter(np.arange(1, 501), sigmas_x, s=1.2, c="r", marker="^", label="Desviación en x")
ax.scatter(np.arange(1, 501), sigmas_y, s=1.2, c="b", marker="o", label="Desviación en y")
ax.plot(lisp, curva1(lisp), color="green", linestyle='dashed', label=r"$\sqrt{N/4}$")
ax.legend(loc="best")
ax.set_xlabel("N")
ax.set_ylabel("$\sigma(N)$")
ax.set_xticks(np.arange(1, 501, 10))
ax.set_title("Desviación estándar respecto de N")
fig.savefig("Sigma")
fig.show()


datosx, datosy = mecest.walk(N=1000, I=1000, Prob=[0.5, 0.0, 0.0, 0.5])
print(datosx, datosy)
mecest.graficar(datosx, datosy, nm_graph="Prob_5_0_0_5", N=1000, I=1000)
mecest.histrogramas(datosx, datosy, nm_graph="Hist_5_0_0_5", N=1000, I=1000)

datosx, datosy = mecest.walk(N=1000, I=1000, Prob=[0.1, 0.4, 0.4, 0.1])
mecest.graficar(datosx, datosy, nm_graph="Prob_1_4_4_1", N=1000, I=1000)
mecest.histrogramas(datosx, datosy, nm_graph="Hist_1_4_4_1", N=1000, I=1000)


mu_x = np.zeros(200)
mu_y = np.zeros(200)
for n in range(1, 201):
    datos_x, datos_y = mecest.walk(N=n, I=1000, Prob=[0.5, 0.0, 0.0, 0.5])
    print(n)
    mx_total = np.array([x[-1] for x in datos_x])
    my_total = np.array([y[-1] for y in datos_y])
    mu_x[n-1] = np.mean(mx_total)
    mu_y[n-1] = np.mean(my_total)

def curva2_x(N):
    return N*(0.5-0.0) 

def curva2_y(N):
    return N*(0.5-0.0)

print("OK")
lisp = np.linspace(1, 201, 2500)

fig, ax = plt.subplots()
ax.scatter(np.arange(1, 201), mu_x, s=1.2, c="r", marker="^", label="Desviación en x")
ax.plot(lisp, curva2_x(lisp), color="green", linestyle='dashed', label=r"$N(0.5-0.0)$")
ax.legend(loc="best")
ax.set_xlabel("N")
ax.set_ylabel("$\mu(N)$")
ax.set_xticks(np.arange(1, 201, 10))
ax.set_title("Valor medio respecto de N")
fig.savefig("Mu_x")

fig, ax = plt.subplots()
ax.scatter(np.arange(1, 201), mu_y, s=1.2, c="b", marker="o", label="Desviación en y")
ax.plot(lisp, curva2_y(lisp), color="green", linestyle='dashed', label=r"$N(0.5-0.0)$")
ax.legend(loc="best")
ax.set_xlabel("N")
ax.set_ylabel("$\mu(N)$")
ax.set_xticks(np.arange(1, 201, 10))
ax.set_title("Valor medio respecto de N")
fig.savefig("Mu_y")


for I in [100, 1000, 2000]:
    datos_x, datos_y = mecest.walk(N=5000, I=I, Prob=[0.25, 0.25, 0.25, 0.25])
    nombre = "grafica_N5000_I{}".format(I) 
    mecest.histrogramas(datos_x, datos_y, nm_graph=nombre, N=5000, I=I)
