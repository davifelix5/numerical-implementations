import numpy as np
class Tuberculosis:
    def __init__(self, Lambda, mu, c, d, beta, k, p, q, r1, r2, delta1, delta2, a,  b , f_, N,
                 numerical_method):
        
        self.Lambda = Lambda
        self.mu = mu
        self.c = c
        self.d = d
        self.beta = beta
        self.k = k
        self.p = p
        self.q = q
        self.r1 = r1
        self.r2 = r2
        self.delta1 = delta1
        self.delta2 = delta2
        self.a = a
        self.b = b
        self.f_ = f_
        self.N = N
        self.numerical_method = numerical_method

    def f(self, t, x):
        s, e, i, r = x
        lamb = (self.beta*self.c*i)

        f0 = (1-self.a-self.b)*self.Lambda - (lamb + self.mu)*s
        f1 = self.a*self.Lambda + self.f_*lamb*s - self.delta1*lamb*e - (self.k+self.mu)*e + self.delta2*lamb*r - self.r1*e
        f2 = self.b*self.Lambda + (1-self.f_)*lamb*s + self.delta1*lamb*e + self.k*e - (self.mu+self.d+self.p)*i + self.q*r - self.r2*i
        f3 = self.p*i - self.mu*r - (self.q+self.delta2*lamb)*r + self.r1*e  + self.r2*i 

        return np.array([f0, f1, f2, f3]).astype('float64')

    def run_numerical_method(self):
        self.numerical_method.f = lambda t, x: self.f(t, x)
        self.numerical_method.run()