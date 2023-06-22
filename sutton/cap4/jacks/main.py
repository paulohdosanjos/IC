import numpy as np
import policy
import jacks as problem

def main():
    S_ = problem.S() #mapping to states
    #pi = problem.compute_pi() #policy
    pi = problem.testing_pi() #policy
    R = problem.compute_R() #expected rewards
    P = problem.compute_P() #transitions probs

    GAMMA = problem.gamma()
    THRESHOLD = problem.threshold()
    n_s = problem.number_states()
    V = np.zeros(n_s)

    V = policy.evaluation(S_,V,pi,R,P,GAMMA,THRESHOLD)

    problem.print_V(V)
        
    return 

if __name__ == "__main__":
    main()
