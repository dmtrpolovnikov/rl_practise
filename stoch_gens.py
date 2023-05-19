#=============================================================================#
#               Library with stochastic processes generator                   #
#=============================================================================#
# import required libraries
import numpy as np
import numpy.random as rnd
from dataclasses import dataclass
import itertools

#=============================================================================#
#                         Ornstein-Uhlenbeck generator                        #
#=============================================================================#
# simulation func for iterator-function
def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)
    
    
# State class declaration
class State:
    pass

# process class
@dataclass
class OrnsteinOuhlenbeck:
    # implement internal class - State, storing process parameters and price
    @dataclass
    class State:
        price:   float = 1
    # process parameters :        
    kappa:   float = 1
    sigma:   float = 0.1
    theta:   float = 1
    dt:      float = 0.01
    sqrt_dt: float = 0.1

        
    def next_state(self, state: State) -> State:
        return OrnsteinOuhlenbeck.State(
            # price St+1 = St + theta * (kappa - St) * dt + sigma * dWt
            price = state.price + self.theta * (self.kappa - state.price)\
                * self.dt + self.sigma * np.random.normal(0, self.sqrt_dt)
        )
     
# trace generator function
def ou_price_traces(
        start_price: float = 1,
        kappa:       float = 1,
        sigma:       float = 0.1,
        theta:       float = 1,
        dt:          float = 0.01,
        time_steps:  int   = 100,
        num_traces:  int   = 1,
        random_state: int  = 0) -> np.ndarray:
    
    if random_state :
        np.random.seed(random_state)
        
    process = OrnsteinOuhlenbeck(
            kappa=kappa,
            sigma=sigma,
            theta=theta,
            dt=dt,
            sqrt_dt=np.sqrt(dt)
    )
    start_state = OrnsteinOuhlenbeck.State(price=start_price)
    return np.vstack([
            np.fromiter((s.price
                    for s in itertools.islice(simulation(process, start_state),
                                             time_steps)), float)
            for _ in range(num_traces)])
            
#=============================================================================#
