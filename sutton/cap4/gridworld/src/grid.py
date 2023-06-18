import numpy as np

def get_dimension():
    return 5

def get_number_states():
    """
    Dimension of the grid
    """
    return get_dimension()*get_dimension()

def get_number_actions():
    """
    Number of possible actions in the gridworld problem
    """
    return 4

def state_A():
    return (0,1)

def state_Af():
    return (4,1)

def state_B():
    return (0,3)

def state_Bf():
    return (2,3)

def S():
    """
    Mappings from index to states
    The array S has only one dimension. 
    Example:
        if the states are "alive", "dead", "almost dead", S has 3 elements and S could be: S = [ "alive", "dead", "almost dead" ]
    """
    n = get_dimension()
    n_s = get_number_states()
    S = np.empty(n_s, dtype = object)
    for x in range(n):
        for y in range(n):
            s = (x,y)
            i = n * x + y
            S[i] = s
    return S
    
def A():
    """
    Mappings from index to actions
    """
    n_a = get_number_actions()
    A = np.empty(n_a, dtype = object)
    A[0] = "left"
    A[1] = "right"
    A[2] = "up"
    A[3] = "down"

    return A
    
def compute_pi():
    """
    Compute policy with equal probs for all states and actions
    """
    n_a = get_number_actions() #number of possible actions
    n_s = get_number_states()
    pi = np.zeros((n_a,n_s))

    for i in range(len(pi)):
        for j in range(len(pi[0])):
            pi[i][j] = 0.25
    
    return pi

def take_to_border(s,a):
    n = get_dimension()
    x,y = s
    
    LEFT = a == "left" and y == 0
    RIGHT = a == "right" and y == n-1
    UP = a == "up" and x == 0
    DOWN = a == "down" and x == n-1

    return LEFT or RIGHT or UP or DOWN


def r(_s,_a):
    S_ = S()
    s = S_[_s]
    A_ = A()
    a = A_[_a]

    if(s == state_A()): return 10
    elif(s == state_B()): return 5
    elif(take_to_border(s,a)): return -1
    else: return 0



def compute_R():
    """
    Computes r(s,a) matrix of expected rewards
    """
    n_s = get_number_states()
    n_a = get_number_actions()

    R = np.zeros((n_s,n_a))
    for _s in range(n_s):
        for _a in range(n_a):
            #compute r(s,a)
            R[_s][_a] = r(_s,_a)
    return R


def p(_sf,_s,_a):
    """
    Returns p(s'|s,a)
    """
    S_ = S()
    s = S_[_s] 
    sf = S_[_sf]
    xf,yf = sf
    x,y = s

    A_ = A()
    a = A_[_a]

    n = get_dimension()

    if(s == state_A()):
        if(sf == state_Af()): return 1
        else: return 0

    if(s == state_B()):
        if(sf == state_Bf()): return 1
        else: return 0

    # NOT (A or B)

    LEFT = xf == x and yf == max(y-1,0) and a == "left"
    UP = xf == max(x-1,0) and yf == y and a == "up" 
    RIGHT = xf == x and yf == min(n-1,y+1) and a == "right"
    DOWN = xf == min(n-1,x+1) and yf == y and a == "down"
    
    if (LEFT or RIGHT or UP or DOWN): return 1
    else: return 0

def compute_P():
    """
    Compute matrix p[s'][s][a] which entries are p(s'|s,a)
    """
    n_s = get_number_states()
    n_a = get_number_actions()

    P = np.zeros((n_s,n_s,n_a))
    for _sf in range(n_s):
        for _s in range(n_s):
            for _a in range(n_a):
                #compute p(s'|s,a)
                P[_sf][_s][_a] = p(_sf,_s,_a)
    return P

def get_gamma():
    """
    Discouting factor
    """
    return 0.9

def get_threshold():
    """
    Threshold for policy evaluation
    """
    return 0.000000000000001

def print_V(V):
    n = get_dimension() 
    hbar = "-" * 45
    print(hbar)
    for x in range(n):
        print("|", end = "")
        for y in range(n):
            i = n * x + y
            print("{:7.2f}".format(V[i]), end = " |")
        print()
        print(hbar)



