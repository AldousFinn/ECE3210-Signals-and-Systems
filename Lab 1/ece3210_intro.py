#Huxley Rust
#ECE3210 - Lab 1
#August 27, 2025

import numpy as np

def py_cumtrapz(f, f_time):
    if f.ndim != f_time.ndim or f.shape != f_time.shape:
        raise ValueError("f and f_time must have the same shape!")
    delta_t = f_time[1:] - f_time[:-1]
    y = np.cumsum((f[:-1] + f[1:])/2 * delta_t)
    y_time = (f_time[:-1] + f_time[1:]) / 2
    return y, y_time


def analytical_integral(t):
    return (5 - 5*np.exp(-t)) * (t >= 0)



if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    time_array = np.linspace(-1, 5, 10000)
    y_analytical = []
    for tt in time_array:
        y_analytical.append(analytical_integral(tt))
    y_analytical = np.array(y_analytical)
    f = 5 * np.exp(-time_array) * (time_array >= 0)
    y, y_time = py_cumtrapz(f, time_array)

    plt.figure()
    plt.plot(time_array, y_analytical, label='Analytical Solution')
    plt.plot(y_time, y, label='Numerical Solution')
    plt.xlabel('time {s}')
    plt.ylabel('y(t) {V}')
    plt.title('ECE3210 - Lab 1 - Numerical vs Analytical Solution')
    plt.grid()
    plt.legend()
    plt.savefig('numerical_vs_analytical.png', dpi=300)

    df = pd.read_csv('scope_4.csv', header=None, skiprows=7)
    time_data = df[0]
    voltage_data_in = df[1]
    voltage_data_out = df[2]

    plt.figure()
    phi = np.deg2rad(85.87)
    eq = 72.05e-3 * np.sin(2 * np.pi * 10000 * time_data + phi)
    input = np.sin(2 * np.pi * 10000 * time_data)
    plt.plot(time_data, input, label='Theoretical $V_{i}$')
    plt.plot(time_data, eq, label='Theoretical $V_{o}$')
    plt.plot(time_data, voltage_data_in, label='Measured $V_{i}$')
    plt.plot(time_data, voltage_data_out, label='Measured $V_{o}$')
    plt.xlabel('time {s}')
    plt.ylabel('Voltage {V}')
    plt.title('Voltage vs Time')
    plt.title('ECE3210 - Lab 1 - Voltage vs Time - 10kHz Sine Wave Input')
    plt.legend()
    plt.grid()
    plt.savefig('voltage_time_plot.png', dpi=300)
    

    plt.figure()
    dfunction = pd.read_csv('scope_2.csv', header=None, skiprows=7)
    frequency_data = dfunction[1]
    gain_data = dfunction[3]
    phase_data = dfunction[4]

    plt.subplot(2, 1, 1)
    plt.plot(frequency_data, gain_data, label='Gain Data', color='blue')
    plt.xscale('log')
    plt.title('ECE3210 - Lab 1 - Phase and Gain vs Frequency')
    plt.ylabel('Gain {dB}')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(frequency_data, phase_data, label='Phase Data', color='orange')
    plt.xscale('log')
    plt.xlabel('Frequency {Hz}')
    plt.ylabel('Phase {degrees}')
    plt.legend()
    plt.grid()
    plt.savefig('bode_plot.png', dpi=300)

    plt.show()