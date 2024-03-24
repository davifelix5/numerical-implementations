import numpy as np
import plotly.graph_objs as go

class Spline:
    def __init__(self, x: np.array, y: np.array, h: np.float64, fpo: np.float64, fpn: np.float64):
        self.n = x.shape[0] - 1
        self.fpo = fpo
        self.fpn = fpn
        self.x = x
        self.y = y
        self.a = np.zeros((self.n+1,))
        self.b = np.zeros((self.n+1,))
        self.c = np.zeros((self.n+1,))
        self.d = np.zeros((self.n+1,))
        self.h = np.zeros((self.n,))
        self.l = np.zeros((self.n+1,))
        self.u = np.zeros((self.n+1,))
        self.z = np.zeros((self.n+1,))
        self.alphas = np.zeros((self.n+1,))
        self.fpo = np.float64()
        self.fpn = np.float64()
        self.h = h

    def calculate_alphas(self):
        self.alphas[0] = (3*(self.a[1] - self.a[0])/self.h) - 3*self.fpo
        self.alphas[self.n] = 3*self.fpn - 3*(self.a[self.n] - self.a[self.n - 1])/self.h
        for i in range(1, self.n):
            self.alphas[i] = (3*(self.a[i + 1] - self.a[i]))/self.h  -  (3*(self.a[i] - self.a[i - 1]))/self.h

    def solve_linear_system(self):
        self.l[0] = 2*self.h
        self.u[0] = 0.5
        self.z[0] = self.alphas[0]/self.l[0]

        for i in range(1, self.n):
            self.l[i] = 2*(self.x[i + 1] -  self.x[i - 1]) - self.h*self.u[i - 1]
            
            self.u[i] = self.h/self.l[i]

            self.z[i] = (self.alphas[i] - (self.h*self.z[i-1]))/self.l[i]
        
        self.l[self.n] = (self.h*(2 - self.u[self.n - 1])) # l_n
        self.z[self.n] = (self.alphas[self.n] - self.h*self.z[self.n - 1])/self.l[self.n] # z_n

        c = np.zeros((self.n+1,))
        c[self.n] = self.z[self.n]
    
    def finish_parameters(self):
        for j in range(self.n-1, -1, -1):
            self.c[j] = self.z[j] - (self.u[j]*self.c[j+1])

            self.b[j] = (self.a[j + 1] - self.a[j])/self.h - (self.h*(self.c[j + 1] + 2*self.c[j]))/3

            self.d[j] = (self.c[j + 1] - self.c[j])/3*self.h

    def interpolate(self):
        self.a = self.y
        self.calculate_alphas()
        self.solve_linear_system()
        self.finish_parameters()

    def sj(self, xi, j):
        return self.a[j] + self.b[j]*(xi-self.x[j]) + self.c[j]*(xi-self.x[j])**2 + self.d[j]*(xi-self.x[j])**3

    def s(self, xi):
        indice = ((xi - self.x[0]) // self.h).astype(int)
        return self.sj(xi, indice)

    def plot(self, exact_f=None, N=10000):
        x_values = np.linspace(self.x[0], self.x[self.n], N)
        y_values = self.s(x_values)
        plots = [
            go.Scatter(name=f'interpolação', x=x_values, y=y_values, mode='lines',
                        showlegend=True)
        ]
        
        if exact_f is not None:
            plots.append(
                go.Scatter(name=f'função exata', x=x_values, y=exact_f(x_values), mode='lines',
                        showlegend=True, line=dict(dash='dash'))
            )

        fig = go.Figure(plots)
        fig.update_layout(
            width=800, 
            height=500, 
            title=f'Interpolação por splines cúbicas', yaxis_title='s(t)', xaxis_title='t'
        )
        fig.show()