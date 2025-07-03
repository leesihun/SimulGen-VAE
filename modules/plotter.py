import matplotlib.pyplot as plt

def temporal_plotter(data, axis, a, c, print_graph, n):
    if axis==0:
        for i in range(n):
            plt.plot(data[a+i, :, c], label = f'{a+i}')
    elif axis==2:
        for i in range(n):
            plt.plot(data[a, :, c+i], label = f'{c+i}')

    plt.legend()
    plt.title('Temporal data')
    plt.show()