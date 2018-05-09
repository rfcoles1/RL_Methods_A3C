import numpy as np

h = 0.0033
def rk4(r,t,h):
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1, t+0.5*h)
    k3 = h*f(r+0.5*k2, t+0.5*h)
    k4 = h*f(r+k3, t+h)
    return (k1+2*k2+2*k3+k4)/6.0

def f(r,t):
    A = r[0]
    B = r[1]
    C = r[2]

    k1 = r[3]
    k2 = r[4]
    k3 = r[5]

    fA = -k1*A + k2*B
    fB = k1*A - (k2*k3)*B
    fC = k3*B
   
    return np.array([fA,fB,fC, k1,k2,k3], float)



class react():
    def __init__(self):
        self.num_rates = 3
        self.maxK = 2
        self.minK = 0
        self.increment = 0.01
        self.timestep = 0.01
        self.timelimit = 1
        self.newEpisode()
        h = 0.0033

    def newEpisode(self):   
        self.A = 100
        self.B = 0 
        self.C = 0
        self.t = 0

        self.k1 = 1
        self.k2 = 1
        self.k3 = 1      

    def reaction(self,A,B,C,k1,k2,k3,currtime,timestep):
        t = currtime
        r = np.array([A,B,C,k1,k2,k3],float)
    
        while(t < currtime+timestep): 
            t+=h
            r+= rk4(r,t,h)
    
        return r[2]

    def get_State(self):
        return [self.A,self.B,self.C,self.k1,self.k2,self.k3]

    def perform_action(self,action):
        currProd = self.reaction(self.A,self.B,self.C, self.k1, self.k2, self.k3, self.t, self.timestep)
        self.change_rate(action)
        nextProd = self.reaction(self.A,self.B,self.C, self.k1, self.k2, self.k3, self.t, self.timestep)
        if (nextProd - currProd) <= 0:
            rew = -1
        else: 
            rew = 1
        self.t = self.t + self.timestep
        return rew, nextProd

    def change_rate(self, action):
        for i in range(0, self.num_rates):
                rates = self.get_Rates()
                if (action == i) & (rates[i] < self.maxK):
                    rates[i] += self.increment
                    self.set_Rates(rates)
                if (action == (i+self.num_rates)) & (rates[i] > self.minK):
                    rates[i] -= self.increment 
                    self.set_Rates(rates)
    

    def is_finished(self):
        if self.t > self.timelimit:
            return True
        return False

    def get_Rates(self):
        return [self.k1,self.k2,self.k3]
   
    def set_Rates(self, rates):
        self.k1 = rates[0]
        self.k2 = rates[1]
        self.k3 = rates[2]

    def get_NumRates(self):
        rates = 3
        return rates

    def get_NumSpecies(self):
        species = 3
        return species
