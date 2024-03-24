import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Callable, List, Union

float_array = Union[npt.NDArray, np.float64, float]

"""
Classe para implementar e avaliar um método numérico qualquer
"""
class NumericalMethod:
    def __init__(self, T: float, y_0: npt.NDArray, f: Callable[[float_array, float_array], float_array], 
                 var_names: List[str], n: int = 1000, t_0: float = 0):
        """
        n: número de pontos na partição uniforme de discretização
        t_0: tempo inicial
        T: tempo final
        y_0: condição incial da variável de estado
        f: função f do problema de Cauchy
        """
        self.T = T
        self.t_0 = t_0
        self.f = f
        self.var_names = var_names
        self.n = self.n0 = n
        self.y_0 = y_0
        self.y = [self.y_0]
        self.t = [self.t_0]

    def run(self):
        """
        Função que executa o método numérico
        """
        raise NotImplementedError

    # Definindo a propriedade n e seu setter para que o h seja redefinido quando n é atualizado
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value
        self.h = (self.T-self.t_0)/value

    def convergence_table_with_exact_solution(self, r: int, n_0: int, n_loops: int, exact_value_on_T: npt.NDArray):
        """
        Constrói uma tabela de convergência com a ordem de convergência estimada a partir do valor exato da solução.
        r: razão de refinamento
        n_0: valor de n na primeira aplicação do método
        n_loops: quantidade de aplicações do método para a tabela
        exact_value_on_T: valor da solução exata no instante de tempo final
        """
        i = 0
        n_values = []
        h_values = []
        global_errors = []
        logs = [np.array([0])]
        for i in range(n_loops):
            self.n = n_0*r**(i)
            n_values.append(self.n)
            h_values.append(self.h)
            self.run()
            y_n = self.y[-1]
            global_errors.append(np.abs(y_n - exact_value_on_T))
            if i >= 1:
                q = abs(global_errors[i-1]/global_errors[i])
                logs.append(np.log(q)/np.log(r))
        return list(zip(n_values, h_values, global_errors, logs))
    
    def convergence_table_without_exact_solution(self, r: int, n_0: int, n_loops: int):
        """
        Constrói uma tabela de convergência com a ordem de convergência estimada numericamente sem a solução exata.
        r: razão de refinamento
        n_0: valor de n na primeira aplicação do método
        n_loops: quantidade de linhas na tabela
        exact_value_on_T: valor da solução exata no instante de tempo final
        """
        i = 0
        n_values = []
        h_values = []
        numerical_solutions = []
        logs = [np.array([0]), np.array([0])]
        for i in range(n_loops):
            self.n = n_0*r**(i)
            n_values.append(self.n)
            h_values.append(self.h)
            self.run()
            numerical_solutions.append(self.y[-1])
            if i >= 2:
                q = abs((numerical_solutions[i-2]-numerical_solutions[i-1])/(numerical_solutions[i-1]-numerical_solutions[i]))
                logs.append(np.log(q)/np.log(r))
        return list(zip(n_values, h_values, numerical_solutions, logs))
    
    def plot_solutions(self, exact_function: Union[List[Callable], None]=None):
        """
        Função que plota a solução numérica encontrada pelo método. Caso seja fornecida uma solução exata,
        ela também é plotada com sua respetiva leganda.
        """
        self.n = self.n0  # Volta ao intervalo de discretização original
        self.run()

        state_results = np.array(self.y).transpose()
        amount_of_state_vars = state_results.shape[0]
        plots = [[] for _ in range(amount_of_state_vars)]

        for k, result in enumerate(state_results):
            plots[k].append(go.Scatter(name='Solução numérica', x=self.t, y=result,
                               mode='lines', showlegend=True, line=dict(color='black')))

        for k in range(amount_of_state_vars):
            if exact_function:
                x = np.linspace(self.t_0, self.T, 1000)
                plots[k].append(go.Scatter(name='Exact Solution', x=x, y=exact_function[k](x),
                                          mode='lines', showlegend=True, line=dict(dash='dash', color='black'),
                                          opacity=0.8))
            fig = go.Figure(plots[k])
            fig.update_layout(yaxis_title='y(t)',
                              title=f'Aproximação numérica para {self.var_names[k]}(t)',
                              width=850, height=500, plot_bgcolor='rgba(0,0,0,0)')
            fig.update_yaxes(ticks='inside', showline=True, linewidth=1, linecolor='black', mirror=True, title=f'{self.var_names[k]}(t)')
            fig.update_xaxes(ticks='inside', showline=True, linewidth=1, linecolor='black', mirror=True, title='t')
            fig.show()

    def plot_solutions_plt(self, exact_function: Union[List[Callable], None]=None):
        """
        Função que plota a solução numérica encontrada pelo método. Caso seja fornecida uma solução exata,
        ela também é plotada com sua respetiva leganda.
        """
        self.n = self.n0  # Volta ao intervalo de discretização original
        self.run()

        state_results = np.array(self.y).transpose()
        amount_of_state_vars = state_results.shape[0]
        fig, ax = plt.subplots(amount_of_state_vars, 1, figsize=(10,8))

        for k, result in enumerate(state_results):
            ax[k].plot(self.t, result, label=f'Solução numérica', color='black')
            ax[k].set_title(f"Solução numérica para {self.var_names[k]}(t)")

        for k in range(amount_of_state_vars):
            if exact_function:
                ax[k].append(self.t, exact_function[k](self.t), label='Solução exata')
                ax[k].legend()

        return ax

    def plot_steps(self, n_0: int, r: int, n_loops: int, y_e: Union[None, List[Callable]] = None):
        """
        Plota a aproximação numérica para diferentes valores de n 
        n_0: valor inicial de n
        r: razão com a qual n aumenta
        n_loops: quantidade de plots e aproximações a serem feitas
        """
        amount_of_state_vars = self.y_0.shape[0]
        plots = [go.Figure() for _ in range(amount_of_state_vars)]
        dashs = ['1, 10',
            '1, 1',
            '10, 3',
            '5, 10',
            '5, 5',
            '5, 1',
            '3, 10, 1, 10',
            '3, 5, 1, 5',
            '3, 1, 1, 1',
            '3, 5, 1, 5, 1, 5',
            '3, 10, 1, 10, 1, 10',
            '3, 1, 1, 1, 1, 1'
        ]

        for i in range(n_loops):
            n = n_0*(r**i)
            self.n = n
            self.run()

            state_results = np.array(self.y).transpose()

            for k, result in enumerate(state_results):
                plots[k].add_trace(go.Scatter(name=f'{self.var_names[k]}(t), n={n}', x=self.t, y=result,
                                              showlegend=True, 
                                              mode='lines',
                                              line=dict(dash=dashs[i%len(dashs)], color='black', shape='linear')))
            
        for k in range(amount_of_state_vars):
            if y_e:
                x = np.linspace(self.t_0, self.T, self.n0)
                plots[k].add_trace(go.Scatter(name=f'{self.var_names[k]}(t) exata', x=x, y=y_e[k](x),
                                          mode='lines', showlegend=True, line=dict(color='black')))
            plots[k].update_layout(title=f'Aproximações numéricas de {self.var_names[k]}(t)',
                          yaxis_title='y(t)', width=850, height=500, plot_bgcolor='rgba(0,0,0,0)')
            plots[k].update_xaxes(ticks='inside', showline=True, linewidth=1, linecolor='black', mirror=True, title='t')
            plots[k].update_yaxes(ticks='inside', showline=True, linewidth=1, linecolor='black', mirror=True, title=f'{self.var_names[k]}(t)')
            plots[k].show()

    def plot_steps_plt(self, n_0: int, r: int, n_loops: int, y_e: Union[None, List[Callable]] = None):
        linestyles = [(0, (1, 10)),
        (0, (1, 1)),
        (0, (5, 10)),
        (0, (5, 5)),
        (0, (5, 1)),
        (0, (3, 10, 1, 10)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 10, 1, 10, 1, 10)),
        (0, (3, 1, 1, 1, 1, 1))]

        fig, ax = plt.subplots(len(self.y_0), 1, figsize=(10,8))

        for i in range(n_loops):
            n = n_0*(r**i)
            self.n = n
            self.run()
            
            for k, result in enumerate(np.array(self.y).transpose()):
                ax[k].plot(self.t, result, label=f"${self.var_names[k]}(t), n={n}$", linestyle=linestyles[i%len(linestyles)], color='black')
                

        for k in range(len(self.y_0)):
            ax[k].set_title(f"Variável de estado {self.var_names[k]}(t) para diferentes valores de n") 
            if y_e:
                ax[k].plot(self.t, y_e[k](self.t), label=f'${self.var_names[k]}(t)$ exata', color='black')
            ax[k].legend()
        
        return ax
            
    def plot_f_approx(self):
        self.n = self.n0
        self.run()
        for k, result in enumerate(np.array(self.y).transpose()):
            fig = go.Figure()
            fig.update_layout(width=800, height=500, title=f'Aproximação para f_{k}(t,y(t))', yaxis_title='~f(t,y(t))', xaxis_title='t',)
            fig.add_scatter(x=self.t, y=self.f(np.array(self.t), np.array(result)), mode='lines')
            fig.show()


def convert_dict_to_latex(convergence_data: list):
    """
    Função para converter a tabela de convergência em notação do Latex
    convergence_data: dados retornados pelo método da tabela de convergência
    """
    from itertools import zip_longest
    for value in convergence_data:
        final_str = '{:5d} & {:.3e}'.format(*value[:2])
        for data in zip_longest(value[2], value[3], fillvalue=0):
            final_str += ('& {:.3e}'*len(data)).format(*data)
        print(final_str)
