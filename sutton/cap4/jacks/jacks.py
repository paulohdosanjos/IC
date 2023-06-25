import numpy as np
from scipy.stats import poisson

MAX_CARS = 20 # maximum number of cars in any location at a given time
MAX_MOVES = 5 #maximum number of cars moves per night

ld = [-100, 2, 3]
lp = [-100, 4, 3]

LOC_1 = -1
LOC_2 = 1

INVALID_ACTION = -100000

def dimension():
    return MAX_CARS + 1

def number_actions():
    """
    Number of possible actions in the jacks problem
    """
    return 2 * MAX_MOVES + 1

def number_states():
    return dimension()*dimension()

def state_map():
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
            S[i] = s
    return S
    
def action_map():
    """
    Mappings from index to actions for the jacks problem
    """
    A = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])

    return A

def action_index(a):
    return int(MAX_MOVES + a)

def compute_loc_R(LOC):
    """
    compute R1[x][a] matrix for LOC = LOC1
    compute R2[y][a] matrix for LOC = LOC2
    """
    if(LOC == LOC_1): LABEL = 1
    else: LABEL = 2


    n = dimension()
    n_a = number_actions()
    R_loc = np.zeros((n,n_a))
    A_ = action_map()

    for _z in range(dimension()):
        for _a in range(n_a):
            z = _z 
            a = A_[_a]
            _max_requests = max_requests(z,a,LOC)
            sum = 0
            for p in range(_max_requests + 1):
                sum += p * request_prob(p,z,a,LOC)
            R_loc[_z][_a] = sum
            print(f"R{LABEL}[{z}][{a}]= {R_loc[_z][_a]}")
    return R_loc
            


def compute_R():
    """
    Computes r(s,a) matrix of expected rewards
    """
    n_s = number_states()
    n_a = number_actions()
    A_ = action_map()
    S_ = state_map()

    R = np.zeros((n_s,n_a))
    R1 = compute_loc_R(LOC_1)
    R2 = compute_loc_R(LOC_2)

    for _s in range(n_s):
        for _a in range(n_a):
            #compute r(s,a)
            x,y = S_[_s]
            a = A_[_a]
            s = S_[_s]
            _x,_y = x,y
            if(not(valid_action(a,x,y))):
                R[_s][_a] = INVALID_ACTION
                print(f"R[{s}][{a}]= {R[_s][_a]}")
            else:
                R[_s][_a] = 10*(R1[_x][_a]+R2[_y][_a]) - 2 * abs(a)
                print(f"R[{s}][{a}]= {R[_s][_a]}")
    return R

def max_requests(z,a,LOC):
    """
    returns maximum number of requests in a certains LOC
    """
    return z + LOC * a

def request_prob(p,z,a,LOC):
    """
    returns Pr{P_t^{1} = p1} and
            Pr{P_t^{2} = p2}
    suppouse valid action
    """
    _max_requests = max_requests(z,a,LOC)

    if(p < _max_requests): return poisson.pmf(p,lp[LOC])
    elif(p == _max_requests): return 1 - poisson.cdf(p,lp[LOC]) + poisson.pmf(p,lp[LOC])
    else: return 0

def return_prob(d,p,z,a,LOC):
    """
    returns Pr{D_t^{1} = x'-x+a-p1} and
            Pr{D_t^{2} = y'-y-a-p2}
    suppouse valid action
    """
    _max_requests= max_requests(z,a,LOC)

    _max_returns = MAX_CARS + p - _max_requests

    if(d < _max_returns): return poisson.pmf(d,ld[LOC])
    elif(d == _max_returns): return 1 - poisson.cdf(d,ld[LOC]) + poisson.pmf(d,ld[LOC])
    else: return 0


def loc_prob_transition(zf,z,a,LOC):
    """
    returns p(x'|x,a) if LOC = LOC1 
    returns p(y'|y,a) if LOC = LOC2 
    não verifica a validade de "a"
    """
    sum = 0

    _max_requests = max_requests(z,a,LOC)

    for p in range(_max_requests+1):
        sum += request_prob(p,z,a,LOC) * return_prob(zf-z+p-LOC*a,p,z,a,LOC)
        #print(f"p = {p}") 
        #print(f"request_prob = {request_prob(p,z,a,LOC)}")
        #print(f"return_prob = {return_prob(zf-z+p-LOC*a,p,z,a,LOC)}")

    return sum

def compute_loc_P(LOC):
    """
    Compute p[z'][z][a] matrix which entries are p(z'|z,a)
    z = x for LOC = LOC1 and z = y for LOC = LOC2
    """
    n = dimension()
    n_a = number_actions()
    A_ = action_map()

    if(LOC == LOC_1): LABEL = 1
    else: LABEL = 2

    P = np.zeros((n,n,n_a))
    for _zf in range(n):
        for _z in range(n):
            for _a in range(n_a):
                zf = _zf
                z = _z
                a = A_[_a]
                P[_zf][_z][_a] = loc_prob_transition(zf,z,a,LOC)
                print(f"p{LABEL}({zf}|{z},{a}) = {P[_zf][_z][_a]}")

    return P

def valid_action(a,x,y):
    COND1 = a <= x
    COND2 = a >= -y
    COND3 = x-a <= MAX_CARS
    COND4 = y+a <= MAX_CARS
    return COND1 and COND2 and COND3 and COND4

def compute_P():
    """
    Compute matrix p[s'][s][a] which entries are p(s'|s,a)
    """
    n_s = number_states()
    n_a = number_actions()

    S_ = state_map()
    A_ = action_map()

    P1 = compute_loc_P(LOC_1) 
    P2 = compute_loc_P(LOC_2) 
    P = np.zeros((n_s,n_s,n_a))

    for _sf in range(n_s):
        for _s in range(n_s):
            for _a in range(n_a):
                s = S_[_s] 
                sf = S_[_sf]
                a = A_[_a]

                _xf,_yf = sf
                _x,_y = s
                x,y = _x, _y
                #compute p(s'|s,a)
                if(valid_action(a,x,y)): 
                    P[_sf][_s][_a] = P1[_xf][_x][_a] * P2[_yf][_y][_a]
                else:
                    P[_sf][_s][_a] = 0 
                
                print(f"p({sf}|{s},{a}) = {P[_sf,_s,_a]}")
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
    for x in range(n):
        for y in range(n):
            i = n * x + y
            print("{:7.1f}".format(V[i]), end = " ")
        print()

if __name__ == "__main__":
    print("Test suit")

    print(f"dimension = {dimension()}\n")
    print(f"total number of states = {number_states()}\n")
    print(f"total number of possible actions = {number_actions()}\n")

    #print("States mapping")
    #print(S_map())

    print("Actions map:")
    print(action_map())

    #print("Probability matrix:")
    P = compute_P()
    #print(P)

    #print("testing sum_prob = 1 for the big matrix")
    #n_s = number_states()
    #n_a = number_actions()
    #for _s in range(n_s):
    #    for _a in range(n_a):
    #        A_ = action_map()
    #        S_ = state_map()
    #        a = A_[_a]
    #        s = S_[_s] 

    #        x, y = S_[_s]
    #        if(not(valid_action(a,x,y))): continue

    #        sum = 0
    #        for _sf in range(n_s):
    #            sum += P[_sf][_s][_a]
    #        
    #        print(f"sum(p(s'|{s},{a})) = {sum}")

    #testing sum_prob = 1 for individual locations
    #P1 = compute_loc_P(LOC_2)
    #for _x in range(dimension()):
    #    for _a in range(number_actions()):
    #        A_ = action_map()
    #        a = A_[_a]
    #        x = _x
    #        sum = 0
    #        for _xf in range(dimension()):
    #            xf = _xf
    #            sum += P1[_xf][_x][_a]
    #        print(f"sum(p(x'|{x},{a})) = {sum}")


    #print(loc_prob_transition(0,0,-5,LOC_1))
    
    #testing sum_prob = 1 for request probs 
    #n_a = number_actions()
    #A_ = action_map()
    #for _x in range(dimension()):
    #    for _a in range(n_a):
    #        sum = 0
    #        x = _x
    #        a = A_[_a]
    #        _max_requests = max_requests(x,a,LOC_1)
    #        for p1 in range(_max_requests+1):
    #            sum += request_prob(p1,x,a,LOC_1)
    #        print(f"sum(request_prob(p1|{x},{a})) = {sum}")
    #print("test finalized")

    R = compute_R()

    pi = testing_pi()







