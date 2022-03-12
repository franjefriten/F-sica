# Metropolis Monte Carlo para el modelo de Ising en 2D
# Datos de entrada:
# (1) Longitud de la caja en un direccion. Calculo para una caja cubica.
# (2) Temperatura de la simulacion
# (3) Numero total de pasos de Monte Carlo

import numpy as np
from matplotlib import pyplot as plt

# Leemos los datos de entrada por pantalla.
# Longitud de la caja en una direccion.


def ising2d(length=None, T=None, istepmax=None, seed=30):
    length = length or int(input('Longitud de la caja = '))
    cmap = plt.cm.brg
    # Temperatura

    T = T or float(input('Temperatura = '))

    # Numero total de pasos de Monte Carlo

    istepmax = istepmax or int(input('Numero de pasos de Monte Carlo = '))

    # Inicializamos el generador de numeros aleatorios
    np.random.seed(seed)

    # Generamos un vector que guarda las coordenadas de cada uno de los espines, en 2 dimensiones
    # Inicializamos este vector.

    spin = np.zeros((length, length))

    # Generamos un vector que guarda la energia por espin, localizando el espin por sus coordenadas en 2D.
    # Inicializamos este vector

    eatom = np.zeros((length, length))

    # Inicializamos variables para valores promedio de todos los pasos de Monte Carlo
    # vmag0 = magnetizacion total
    # ntotal = numero total de espines

    vmag0 = 0
    ntotal = 0

    # Initialize listas para la energia total (energy), magnetizacion total (magnetization) y
    # numero de eventos aceptados (success) que se guardan en cada paso de Monte Carlo

    energy = []
    magnetization = []
    success = []

    # Generamos la configuracion inicial de espines, asignando aleatoriamente un valor de -1 o de 1
    # en una red cuadrada de longitud en cada lado la longitud inicial tomada como parametro de entrada.
    # En cada punto de esta red cuadrada tendremos un valor de espin que puede ser 1 o -1.

    for i in range(length):
        for j in range(length):
            if np.random.rand() > 0.5:
                spin[i, j] = 1
            else:
                spin[i, j] = -1

    # Calculamos la magnetizacion total, como la suma de todos los espines.

            vmag0 += spin[i, j]

    # Contamos el numero total de espines generados.

            ntotal += 1

    print('Numero total de espines = ', ntotal)
    print('Magnetizacion = ', vmag0)

    # Representamos los espines en un grafico X,Y con dos colores distintos
    plt.figure(dpi=150)
    plt.imshow(spin, aspect='auto', interpolation='none',
               extent=[0, length, 0, length], cmap=cmap)
    plt.axis('equal')
    plt.title("Modelo de Ising 2D - Configuración inicial")
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.ylim((0, length))
    plt.xlim((0, length))
    plt.show()

    # Calculamos la energia total del sistema.

    totalenergy = 0.0

    # Buscamos los vecinos de cada espin, considerando solo primeros vecinos y
    # con condiciones periodicas.
    # Este calculo se puede hacer de muchas maneras. La siguiente es solo una de ellas.
    # Vecinos son los espines que estan arriba, abajo, hacia la derecha o hacia la izquierda.
    # Utilizamos condiciones periodicas: un espin en la posicion length-1 tiene como vecino el del otro extremo (posicion 0)

    for i in range(length):
        for j in range(length):
            if i == 0:
                down = length - 1
            else:
                down = i - 1
            if i == length-1:
                up = 0
            else:
                up = i + 1
            if j == 0:
                left = length - 1
            else:
                left = i - 1
            if i == length-1:
                right = 0
            else:
                right = i + 1

            eatom[i, j] = -spin[i, j] * \
                (spin[up, j]+spin[down, j]+spin[i, left]+spin[i, right])

            totalenergy += 0.5*eatom[i, j]

    print('Energía inicial total = ', totalenergy)

    # Comenzamos el bucle de Monte Carlo.
    # Es lo mismo que hemos hecho hasta ahora pero moviendo uno de los espines cada vez de -1 a 1 o al reves.
    # Inicializamos variables
    # istep = numero total de pasos
    # nsuccess = numero de pasos de Monte Carlo aceptados.
    # nfail = numero de pasos de Monte Carlo no aceptados.
    # isteplastsucess =
    # Eave = energia total promediada
    # vmag = magnetizacion promediada

    istep = 0
    nsuccess = 0
    nfail = 0
    isteplastsuccess = 0
    Eave = 0
    vmag = vmag0

    # Comenzamos el bucle de Monte Carlo para los pasos de simulacion que se han leido como dato de entrada.

    for istep in range(istepmax):

        # Seleccionamos uno de los espines de forma aleatoria, seleccionamos una posicion

        ipick = np.random.randint(0, length-1)
        jpick = np.random.randint(0, length-1)

    # Calculamos el cambio en energia del sistema debido a este cambio en la energia del espin.
    # De esta forma no tenemos que calcular la energia de todo el sistema cada vez que cambiamos un espin.
    # Para hacer este cambio tenemos que utilizar de nuevo condiciones periodicas, en el caso de elegir un espin en el borde.

        if ipick == 0:
            down = length-1
        else:
            down = ipick-1
        if ipick == length-1:
            up = 0
        else:
            up = ipick+1
        if jpick == 0:
            left = length-1
        else:
            left = jpick-1
        if jpick == length-1:
            right = 0
        else:
            right = jpick+1

        deltaenergy = 2.*spin[ipick, jpick]*(
            spin[up, jpick]+spin[down, jpick] + spin[ipick, left] + spin[ipick, right])

    # Aceptamos o rechazamos el cambio en el espin considerando el metodo de Metropolis Monte Carlo
    # Si el cambio en energia es negativo (la nueva configuracion tiene una energia mas baja) lo aceptamos.
    # Si el cambio en energia es positivo, elegimos un numero aleatorio entre 0 y 1 y comparamos con el factor de Boltzmann.
    # Si el numero es menor que ese factor aceptamos la configuracion.

        accept = 0
        if deltaenergy <= 0:
            accept = 1
        elif np.random.rand() < np.exp(-(deltaenergy/T)):
            accept = 1

        Etotold = totalenergy

        if accept == 1:

            # Se se acepta la configuracion, se le da la vuelta al espin
            # Se calcula la nueva magnetizacion y se actualiza el numero de sucesos aceptados.

            vmag = vmag - spin[ipick, jpick]
            spin[ipick, jpick] = -spin[ipick, jpick]
            vmag = vmag + spin[ipick, jpick]
            isteplastsuccess = istep
            nsuccess += 1

    # Se calcula la energia total del sistema

            totalenergy = 0.0
            for i in range(length):
                for j in range(length):
                    if i == 1:
                        down = length-1
                    else:
                        down = i-1
                    if i == length-1:
                        up = 1
                    else:
                        up = i+1
                    if j == 1:
                        left = length-1
                    else:
                        left = j-1
                    if j == length-1:
                        right = 1
                    else:
                        right = j+1

                    eatom[i, j] = -spin[i, j] * \
                        (spin[up, j]+spin[down, j] +
                         spin[i, left]+spin[i, right])
                    totalenergy = totalenergy + 0.5*eatom[i, j]

            Eave += totalenergy
        else:
            # Si no se acepta la configuracion nos quedamos con la anterior, actualizando los contadores.
            Eave += Etotold
            nfail += 1
            vmag = vmag

        magnetization.append(vmag)
        energy.append(totalenergy)
        success.append(nsuccess)
        # print('Numero total de pasos = ', istep, 'Pasos aceptados = ',
        # nsuccess, 'Energia total =', totalenergy)

    # Representamos la energia total en funcion del número de pasos, la magnetización y la configuracion final.
    # Todas las representaciones asi como los bucles son muy mejorables en este programa.

    plt.figure(dpi=150)
    plt.plot(energy, 'r--')
    plt.xlabel('Numero de pasos', fontsize=10)
    plt.ylabel('Energia total', fontsize=10)
    plt.title("Modelo de Ising 2D")
    plt.show()

    plt.figure(dpi=150)
    plt.plot(magnetization, 'r--')
    plt.xlabel('Numero de pasos', fontsize=10)
    plt.ylabel('Magnetizacion', fontsize=10)
    plt.title("Modelo de Ising 2D")
    plt.show()

    plt.figure(dpi=150)
    plt.plot(success, 'r--')
    plt.xlabel('Numero de pasos', fontsize=10)
    plt.ylabel('Numero de pasos aceptados', fontsize=10)
    plt.title("Modelo de Ising 2D")
    plt.show()
    plt.axes(xlim=(0, length), ylim=(0, length))

    plt.figure(dpi=150)
    plt.imshow(spin, aspect='auto', interpolation='none',
               extent=[0, length, 0, length], cmap=cmap)
    plt.axis('equal')
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    plt.title("Modelo de Ising 2D - Configuración final")
    plt.ylim((0, length))
    plt.xlim((0, length))
    plt.show()


def ising2d_noplot(length=None, T=None, istepmax=None, H=0, seed=30):
    length = length or int(input('Longitud de la caja = '))
    # Temperatura

    T = T or float(input('Temperatura = '))

    # Numero total de pasos de Monte Carlo

    istepmax = istepmax or int(input('Numero de pasos de Monte Carlo = '))

    # Inicializamos el generador de numeros aleatorios
    np.random.seed(seed)

    # Generamos un vector que guarda las coordenadas de cada uno de los espines, en 2 dimensiones
    # Inicializamos este vector.

    spin = np.zeros((length, length))

    # Generamos un vector que guarda la energia por espin, localizando el espin por sus coordenadas en 2D.
    # Inicializamos este vector

    eatom = np.zeros((length, length))

    # Inicializamos variables para valores promedio de todos los pasos de Monte Carlo
    # vmag0 = magnetizacion total
    # ntotal = numero total de espines

    vmag0 = 0
    ntotal = 0

    # Initialize listas para la energia total (energy), magnetizacion total (magnetization) y
    # numero de eventos aceptados (success) que se guardan en cada paso de Monte Carlo

    energy = []
    magnetization = []
    success = []

    # Generamos la configuracion inicial de espines, asignando aleatoriamente un valor de -1 o de 1
    # en una red cuadrada de longitud en cada lado la longitud inicial tomada como parametro de entrada.
    # En cada punto de esta red cuadrada tendremos un valor de espin que puede ser 1 o -1.

    for i in range(length):
        for j in range(length):
            if np.random.rand() > 0.5:
                spin[i, j] = 1
            else:
                spin[i, j] = -1

    # Calculamos la magnetizacion total, como la suma de todos los espines.

            vmag0 += spin[i, j]

    # Contamos el numero total de espines generados.

            ntotal += 1

    # Calculamos la energia total del sistema.

    totalenergy = 0.0

    # Buscamos los vecinos de cada espin, considerando solo primeros vecinos y
    # con condiciones periodicas.
    # Este calculo se puede hacer de muchas maneras. La siguiente es solo una de ellas.
    # Vecinos son los espines que estan arriba, abajo, hacia la derecha o hacia la izquierda.
    # Utilizamos condiciones periodicas: un espin en la posicion length-1 tiene como vecino el del otro extremo (posicion 0)

    for i in range(length):
        for j in range(length):
            if i == 0:
                down = length - 1
            else:
                down = i - 1
            if i == length-1:
                up = 0
            else:
                up = i + 1
            if j == 0:
                left = length - 1
            else:
                left = i - 1
            if i == length-1:
                right = 0
            else:
                right = i + 1

            eatom[i, j] = -spin[i, j] * \
                (spin[up, j]+spin[down, j]+spin[i, left]+spin[i, right])

            totalenergy += 0.5*eatom[i, j]

    # Comenzamos el bucle de Monte Carlo.
    # Es lo mismo que hemos hecho hasta ahora pero moviendo uno de los espines cada vez de -1 a 1 o al reves.
    # Inicializamos variables
    # istep = numero total de pasos
    # nsuccess = numero de pasos de Monte Carlo aceptados.
    # nfail = numero de pasos de Monte Carlo no aceptados.
    # isteplastsucess =
    # Eave = energia total promediada
    # vmag = magnetizacion promediada

    istep = 0
    nsuccess = 0
    nfail = 0
    isteplastsuccess = 0
    Eave = 0
    vmag = vmag0

    # Comenzamos el bucle de Monte Carlo para los pasos de simulacion que se han leido como dato de entrada.

    for istep in range(istepmax):

        # Seleccionamos uno de los espines de forma aleatoria, seleccionamos una posicion

        ipick = np.random.randint(0, length-1)
        jpick = np.random.randint(0, length-1)

    # Calculamos el cambio en energia del sistema debido a este cambio en la energia del espin.
    # De esta forma no tenemos que calcular la energia de todo el sistema cada vez que cambiamos un espin.
    # Para hacer este cambio tenemos que utilizar de nuevo condiciones periodicas, en el caso de elegir un espin en el borde.

        if ipick == 0:
            down = length-1
        else:
            down = ipick-1
        if ipick == length-1:
            up = 0
        else:
            up = ipick+1
        if jpick == 0:
            left = length-1
        else:
            left = jpick-1
        if jpick == length-1:
            right = 0
        else:
            right = jpick+1

        deltaenergy = 2.*spin[ipick, jpick]*(
            spin[up, jpick]+spin[down, jpick] + spin[ipick, left] + spin[ipick, right])

    # Aceptamos o rechazamos el cambio en el espin considerando el metodo de Metropolis Monte Carlo
    # Si el cambio en energia es negativo (la nueva configuracion tiene una energia mas baja) lo aceptamos.
    # Si el cambio en energia es positivo, elegimos un numero aleatorio entre 0 y 1 y comparamos con el factor de Boltzmann.
    # Si el numero es menor que ese factor aceptamos la configuracion.

        accept = 0
        if deltaenergy <= 0:
            accept = 1
        elif np.random.rand() < np.exp(-(deltaenergy/T)):
            accept = 1

        Etotold = totalenergy

        if accept == 1:

            # Se se acepta la configuracion, se le da la vuelta al espin
            # Se calcula la nueva magnetizacion y se actualiza el numero de sucesos aceptados.

            vmag = vmag - spin[ipick, jpick]
            spin[ipick, jpick] = -spin[ipick, jpick]
            vmag = vmag + spin[ipick, jpick]
            isteplastsuccess = istep
            nsuccess += 1

    # Se calcula la energia total del sistema

            totalenergy = 0.0
            for i in range(length):
                for j in range(length):
                    if i == 1:
                        down = length-1
                    else:
                        down = i-1
                    if i == length-1:
                        up = 1
                    else:
                        up = i+1
                    if j == 1:
                        left = length-1
                    else:
                        left = j-1
                    if j == length-1:
                        right = 1
                    else:
                        right = j+1

                    eatom[i, j] = -spin[i, j] * \
                        (spin[up, j]+spin[down, j] +
                         spin[i, left]+spin[i, right])
                    totalenergy = totalenergy + 0.5*eatom[i, j]

            Eave += totalenergy
        else:
            # Si no se acepta la configuracion nos quedamos con la anterior, actualizando los contadores.
            Eave += Etotold
            nfail += 1
            vmag = vmag

        magnetization.append(vmag)
        energy.append(totalenergy - H*vmag)
        success.append(nsuccess)
    return np.array(magnetization), np.array(energy)


def ising2d_T_function(T, lenght=None, istepmax=None, lasts=None, H=0):
    length = lenght or int(input('Longitud de la caja = '))
    istepmax = istepmax or int(input('Numero de pasos de Monte Carlo = '))
    lasts = lasts or istepmax
    magnetization = np.zeros(T.shape)
    energy = np.zeros(T.shape)
    energy2 = np.zeros(T.shape)

    for i, t in enumerate(T):
        M, E = ising2d_noplot(lenght, t, istepmax, seed=None, H=H)
        magnetization[i] = np.mean(M[-lasts:])
        energy[i] = np.sum(E[-lasts:])/lasts
        energy2[i] = np.sum(E[-lasts:]**2)/lasts
    Cv = (energy2-energy**2)/T**2
    plt.figure(dpi=150)
    plt.plot(T, magnetization, 'm.', label="Valores numéricos")
    plt.xlabel('T', fontsize=10)
    plt.ylabel('Magnetización', fontsize=10)
    plt.title("Modelo de Ising 2D")
    plt.legend()
    plt.grid()
    plt.savefig("Fig9a_magnetizacion")
    plt.show()

    plt.figure(dpi=150)
    plt.plot(T, energy, 'k.', label="Valores numéricos")
    plt.xlabel('T', fontsize=10)
    plt.ylabel('Energía', fontsize=10)
    plt.title("Modelo de Ising 2D")
    plt.legend()
    plt.grid()
    plt.savefig("Fig9a_energia")
    plt.show()

    plt.figure(dpi=150)
    plt.plot(T, Cv, 'r.', label="Valores numéricos")
    plt.xlabel('T', fontsize=10)
    plt.ylabel('$C_v$', fontsize=10)
    plt.title("Modelo de Ising 2D")
    plt.legend()
    plt.grid()
    plt.savefig("Fig9_Cv")
    plt.show()


T = np.linspace(0.001, 5, 200)
ising2d_T_function(T, 8, 10000, 1000, H=1)
