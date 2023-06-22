import numpy as np
from scipy.stats import poisson

MAX_CARS = 20 # maximum number of cars in any location at a given time
MAX_MOVES = 5 #maximum number of cars moves per night

ld = [-100, 2, 3]
lp = [-100, 4, 3]

LOC_1 = -1
LOC_2 = 1

def dimension():
    return MAX_CARS + 1

def number_actions():
    """
    Number of possible actions in the jacks problem
    """
    return 2 * MAX_MOVES + 1

def number_states():
    return dimension()*dimension()

def S_map():
    """
    Mappings from index to states
    The array S has only one dimension. 
    Example:
        if the states are "alive", "dead", "almost dead", S has 3 elements and S could be: S = [ "alive", "dead", "almost dead" ]
    """
    n = dimension()
    n_s = number_states()
    S = np.empty(n_s, dtype = object)
    for x in range(n):
        for y in range(n):
            s = (x,y)
            i = n * x + y
            s[i] = s
    return S
    
def A_map():
    """
    Mappings from index to actions for the jacks problem
    """
    n_a = number_actions()
    A = np.empty(n_a, dtype = int)

    for i in range(n_a):
        A[i] = -MAX_MOVES + i 
        
    return A
    
def testing_pi():
    """
     policy example for testing. Policy at iteration 5 of https://alexkozlov.com/post/jack-car-rental/
    """
    n_a = number_actions() # size of pi array
    n_s = number_states()

    
    pi = np.array( [ [0,0,0,0,0,0,0,0,-1,-1,-2,-2,-2,-3,-3,-3,-3,-3,-4,-4,-4,
           0,0,0,0,0,0,0,0,0,-1,-1,-1,-2,-2,-2,-2,-2,-3,-3,-3,-3,
           0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-2,
           0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,
           1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           3,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           3,3,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           4,3,3,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           4,4,3,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           5,4,4,3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           5,5,4,3,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           5,5,4,3,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           5,5,4,4,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
           5,5,5,4,3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
           5,5,5,4,3,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
           5,5,5,4,3,3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
           5,5,5,4,4,3,2,2,1,1,1,1,1,0,0,0,0,0,0,0,0,
           5,5,5,5,4,3,3,2,2,2,2,2,1,1,1,1,0,0,0,0,0,
           5,5,5,5,4,4,3,3,3,3,3,2,2,2,2,1,1,1,0,0,0]])
    if(n_a != len(pi)): print("erro 1")
    if(n_s != len(pi[0])): print("erro 2")

    return pi


def r(_s,_a):
    """
    expected reward given index action _a in index state _s
    """
    S_ = S_map()
    s = S_[_s]
    A_ = A_map()
    a = A_[_a]


def compute_R():
    """
    Computes r(s,a) matrix of expected rewards
    """
    n_s = number_states()
    n_a = number_actions()

    R = np.zeros((n_s,n_a))
    for _s in range(n_s):
        for _a in range(n_a):
            #compute r(s,a)
            R[_s][_a] = r(_s,_a)
    return R

def p_max(z,a,LOC):
    """
    returns maximum number of requests in a certains LOC
    """
    return z + LOC * a

def P(p,z,a,LOC):
    """
    returns Pr{P_t^{1} = p1} and
            Pr{P_t^{2} = p2}
    """
    _p_max = p_max(z,a,LOC)

    if(p < _p_max): return poisson.pmf(p,lp[LOC])
    elif(p == _p_max): return 1 - poisson.cdf(p,lp[LOC]) + poisson.pmf(p,lp[LOC])
    else: return 0

def D(d,p,z,a,LOC):
    """
    returns Pr{D_t^{1} = x'-x+a-p1} and
            Pr{D_t^{2} = y'-y-a-p2}
    """
    _p_max = p_max(z,a,LOC)

    _d_max = MAX_CARS + p -_p_max

    if(d < _d_max): return poisson.pmf(d,ld[LOC])
    elif(d == _d_max): return 1 - poisson.cdf(d,ld[LOC]) + poisson.pmf(d,ld[LOC])
    else: return 0


def p_fat(zf,z,a,LOC):
    """
    returns p(x'|x,a) if LOC = LOC1 
    returns p(y'|y,a) if LOC = LOC2 
    não verifica a validade de "a"
    """
    sum = 0

    _p_max = p_max(z,a,LOC)

    for p in range(_p_max):
        sum += D(zf,z,a,p,LOC) * P(z,a,p,LOC)

    return sum


def p(sf,s,a):
    """
    Returns p(s'|s,a)
    """
    xf,yf = sf
    x,y = s

    return p_fat(xf,x,a,LOC_1) * p_fat(yf,y,a,LOC_2)


def compute_P():
    """
    Compute matrix p[s'][s][a] which entries are p(s'|s,a)
    """
    n_s = number_states()
    n_a = number_actions()

    S_ = S_map()
    A_ = A_map()

    P = np.zeros((n_s,n_s,n_a))
    for _sf in range(n_s):
        for _s in range(n_s):
            for _a in range(n_a):
                s = S_[_s] 
                sf = S_[_sf]
                a = A_[_a]
                #compute p(s'|s,a)
                P[_sf][_s][_a] = p(S_,A_,sf,s,a)
    return P

def gamma():
    """
    Discouting factor
    """
    return 0.9

def threshold():
    """
    Threshold for policy evaluation
    """
    return 0.1

def print_V(V):
    n = dimension() 
    hbar = "-" * 45
    print(hbar)
    for x in range(n):
        print("|", end = "")
        for y in range(n):
            i = n * x + y
            print("{:7.2f}".format(V[i]), end = " |")
        print()
        print(hbar)

if __name__ == "__main__":
    print("Test suit")

    print(f"dimension = {dimension()}\n")
    print(f"total number of states = {number_states()}\n")
    print(f"total number of possible actions = {number_actions()}\n")

    #print("States mapping")
    #print(S_map())

    print("Actions map:")
    print(A_map())



