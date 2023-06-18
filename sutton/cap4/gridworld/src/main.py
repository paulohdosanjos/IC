import numpy as np
import policy
import grid as problem

def main():
    S_ = problem.S() #mapping to states
    pi = problem.compute_pi() #policy
    R = problem.compute_R() #expected rewards
    P = problem.compute_P() #transitions probs

    GAMMA = problem.get_gamma()
    THRESHOLD = problem.get_threshold()
    n_s = problem.get_number_states()
    V = np.zeros(n_s)

    V = policy.evaluation(S_,V,pi,R,P,GAMMA,THRESHOLD)

    problem.print_V(V)
        
    return 

if __name__ == "__main__":
    main()
