import plotly.graph_objs as go
import numpy as np
from typing import Callable, Union

"""
Classe para implementar e avaliar um método numérico qualquer
"""
class NumericalMethod:
    def __init__(self, T: float, y_0: float, f: Callable, n: int = 1000, t_0: float = 0):
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

    def convergence_table_with_exact_solution(self, r: int, n_0: int, n_loops: int, exact_value_on_T: float):
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
        logs = [0]
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
        logs = [0, 0]
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
    
    def plot_solutions(self, exact_function: Union[Callable, None]=None):
        """
        Função que plota a solução numérica encontrada pelo método. Caso seja fornecida uma solução exata,
        ela também é plotada com sua respetiva leganda.
        """
        self.n = self.n0  # Volta ao intervalo de discretização original
        self.run()
        figures = []
        figures.append(go.Scatter(name='Solução numérica', x=self.t, y=self.y,
                           mode='lines', showlegend=True))
        if exact_function:
            x = np.linspace(self.t_0, self.T, 1000)
            figures.append(go.Scatter(name='Exact Solution', x=x, y=exact_function(x),
                                      mode='lines', showlegend=True, line=dict(dash='dash'),
                                      opacity=0.8))
        fig = go.Figure(figures)
        fig.update_layout(yaxis_title='y(t)',
                          title='Aproximação da Solução numérica para a equação',
                          width=850, height=500)
        fig.show()

    def plot_steps(self, n_0: int, r: int, n_loops: int, y_e: Union[Callable, None]=None):
        """
        Plota a aproximação numérica para diferentes valores de n 
        n_0: valor inicial de n
        r: razão com a qual n aumenta
        n_loops: quantidade de plots e aproximações a serem feitas
        """
        figures = []
        for i in range(n_loops):
            n = n_0*(r**i)
            self.n = n
            self.run()
            figures.append(go.Scatter(name=f'n={n}', x=self.t, y=self.y, mode='lines',
                                      showlegend=True, line=dict(dash='dash')))
        if y_e:
            x = np.linspace(self.t_0, self.T, 1000)
            figures.append(go.Scatter(name='y_e', x=x, y=y_e(x),
                                      mode='lines', showlegend=True))
        fig = go.Figure(figures)
        fig.update_layout(title='Sucessivas aproximações numéricas para diferentes passos de integração',
                          yaxis_title='y(t)', width=850, height=500)
        fig.show()
        return fig

    def plot_f_approx(self):
        fig = go.Figure()
        fig.update_layout(width=800, height=500, title='Aproximação para f(t,y(t))', yaxis_title='~f(t,y(t))', xaxis_title='t')
        fig.add_scatter(x=self.t, y=self.f(np.array(self.t), np.array(self.y)), mode='lines')
        fig.show()


def convert_dict_to_latex(convergence_data: list):
    """
    Função para converter a tabela de convergência em notação do Latex
    convergence_data: dados retornados pelo método da tabela de convergência
    """
    for value in convergence_data:
        print('{:5d} & {:.3e} & {:.3e} & {:.3e} \\\\'.format(*value))
