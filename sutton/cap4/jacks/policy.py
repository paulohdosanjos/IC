from scipy.stats import f
from jacks import action_index
from jacks import valid_action 

INFTY = 10000000000

def evaluation(S_,V,pi,R,P,GAMMA,THRESHOLD):
    """
    Policy evaluation for deterministic policies
    """
    print("evaluating policy...")
    while(True):
        delta = 0
        for _s in range(len(S_)):
            v = V[_s]
            #updates V(s)
            a = pi[_s]
            _a = action_index(a) 
            sum = 0 
            for _sf in range(len(S_)):
                sum += P[_sf][_s][_a]*V[_sf]
            V[_s] = R[_s][_a] + GAMMA * sum
            delta = max(delta,abs(v-V[_s]))

        if(delta < THRESHOLD):
            break
        else:
            print(f"not done yet: delta = {delta}")

def improvement(is_stable: bool,A_,S_,V,pi,R,P,GAMMA):
    print("improving policy...")

    is_stable = True

    n_a = len(A_) 
    n_s = len(S_)

    for _s in range(len(S_)):
        a_past = pi[_s]
        _a_past = action_index(a_past)
        _max_arg = _a_past
        max_sum = V[_s] 
        for _a in range(n_a):
            a = A_[_a]
            x,y = S_[_s]
            if(not(valid_action(a,x,y))):
                continue
            sum = 0
            for _sf in range(n_s):
                sum += V[_sf]*P[_sf][_s][_a]
            sum = R[_s][_a] + sum * GAMMA
            if(sum > max_sum):
                print(f"sum = {sum} max_sum = {max_sum}")
                _max_arg = _a
                max_sum = sum
                print(f"changed!: pi({(x,y)}) = {pi[_s]} --> {A_[_max_arg]}")
        pi[_s] = A_[_max_arg]
        if(a_past != pi[_s]):
            is_stable = False
    return is_stable
