"""           
Implementação de um método de passo único explícito genérico para 
uma variável de estado (com exemplos)
------------------------------------------------------------------
 Revisoes  :
     Data        Versao  Autor             Descricao
     21/01/2023  1.0     Davi Félix        versao inicial
------------------------------------------------------------------
"""

from NumericalMethod import NumericalMethod
import numpy.typing as npt

"""
Classe genérica para um método explícito de passo único capaz de resolver uma variável de estado
"""
class OneStepMethod(NumericalMethod):

    def phi(self, t: float, y: npt.NDArray):
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
