"""           
Implementação de um método de passo único explícito genérico para 
uma variável de estado (com exemplos)
------------------------------------------------------------------
 Revisoes  :
     Data        Versao  Autor             Descricao
     21/01/2023  1.0     Davi Félix        versao inicial
------------------------------------------------------------------
"""

import numpy as np
from NumericalMethod import NumericalMethod, convert_dict_to_latex

"""
Classe genérica para um método explícito de passo único capaz de resolver uma variável de estado
"""
class OneStepMethod(NumericalMethod):

    def phi(self, t: float, y: float):
        """
        Função Phi do método de passdo único explícito
        """
        raise NotImplementedError

    def run(self):
        """
        Função que executa o laço tempora do Método
        """
        self.y = [self.y_0]
        self.t = [self.t_0]
        for k in range(self.n):
            self.y.append(self.y[k] + self.h*self.phi(self.t[k], self.y[k]))
            self.t.append(self.t[k] + self.h)


class EulerMethod(OneStepMethod):
    """
    Método de Euler simples
    """
    def phi(self, t, y):
        return self.f(t, y)
    

class EulerMethod2(OneStepMethod):
    """
    Método de Euler modificado
    """
    def phi(self, t, y):
        return 0.5*(self.f(t, y) + self.f(t + self.h, y + self.h*self.f(t, y)))
    

class RKC(OneStepMethod):
    """
    Método de Runge Kutta Clássico
    """
    def phi(self, t, y):
        k1 = self.f(t, y)
        k2 = self.f(t+self.h/2, y + self.h/2*k1)
        k3 = self.f(t+self.h/2, y + self.h/2*k2)
        k4 = self.f(t+self.h, y + self.h*k3)
    
        return 1/6*(k1 + 2*k2 + 2*k3 + k4)


if __name__ == '__main__':
    ######## EXEMPLO 1 - EX 2.2 DA SEÇÃO 2.4 DAS NOTAS DE AULA ##########
    y_e = lambda t: np.cos(t) - np.sin(t)
    T = 1
    y_0 = 1
    euler = EulerMethod(T, y_0, lambda t, y: -y*np.tan(t) - 1/np.cos(t))
    print('Tabela de convergência com solução exata de 2.2 em T=1')
    convert_dict_to_latex(
        euler.convergence_table_with_exact_solution(r=2, n_0=64, n_loops=9, exact_value_on_T=y_e(T))
    )
    print('\nTabela de convergência com solução numérica de 2.2 em T=1')
    convert_dict_to_latex(
        euler.convergence_table_without_exact_solution(r=3, n_0=64, n_loops=9)
    )
    euler.plot_solutions(exact_function=y_e)

    ######### EXEMPLO 2 - EXEXMPLO 2.4 DA SEÇÃO 2.3;1 DAS NOTAS DE AULA ##############

    y_e = lambda t: np.sin(2*np.pi*t)*np.exp(-0.2*t)
    T = 1/2
    y_0 = 0
    euler = EulerMethod(T, y_0, lambda t, y: 2*np.pi*np.cos(2*np.pi*t)*np.exp(-0.2*t) - 0.2*y)
    print('\nTabela de convergência com solução exata de 2.4 em T=1/2')
    convert_dict_to_latex(euler.convergence_table_with_exact_solution(r=2, n_0=64, n_loops=4, exact_value_on_T=y_e(T)))
    euler.T = 1  # Mudando o instante de tempo final para 1
    print('\nTabela de convergência com solução exata de 2.4 em T=1')
    convert_dict_to_latex(euler.convergence_table_with_exact_solution(r=2, n_0=64, n_loops=4, exact_value_on_T=y_e(T)))
    
    #### Exemplo 3.1 - Equação manufaturada com Euler ###
    y_e = lambda t: 3*(t**2)*np.cos(t)*np.exp(-2*t)
    f = lambda t, y: -3*np.exp(-2*t)*t*(t*np.sin(t)+2*(t-1)*np.cos(t))
    T = 10
    y_0 = 0
    euler2 = EulerMethod2(T, y_0, f)
    print('\nTabela de convergência para o exemplo 3 em T=5')
    convert_dict_to_latex(euler2.convergence_table_with_exact_solution(
        r=2, n_0=16, n_loops=8, exact_value_on_T=y_e(T)
    ))
    euler2.plot_solutions(exact_function=y_e)
   
    # Plot para soluções usando diferentes valores de n
    euler2.plot_steps(n_0=2, r=2, n_loops=9, y_e =y_e)

    ######### EXEMPLO 3 - Equação manufaturada ###################
    T = 20
    rkc = RKC(T, y_0, f)
    print('\nTabela de convergência para o exemplo 3 em T=5')
    convert_dict_to_latex(rkc.convergence_table_with_exact_solution(
        r=2, n_0=16, n_loops=8, exact_value_on_T=y_e(T)
    ))
    rkc.plot_solutions(exact_function=y_e)
   
    # Plot para soluções usando diferentes valores de n
    rkc.plot_steps(n_0=2, r=2, n_loops=9, y_e =y_e)