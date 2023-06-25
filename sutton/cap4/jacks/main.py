import numpy as np
import policy
import jacks as problem

def main():
    S_ = problem.state_map() #mapping to states
    A_ = problem.action_map() #mapping to actions
    #pi = problem.testing_pi() #policy
    P = problem.compute_P() #transitions probs
    R = problem.compute_R() #expected rewards

    GAMMA = problem.gamma()
    THRESHOLD = problem.threshold()
    n_s = problem.number_states()
    V = np.zeros(n_s)
    pi = np.zeros(n_s, dtype = int)

    is_stable = False
    iteration = 0

    while(not(is_stable)):
        iteration += 1
        policy.evaluation(S_,V,pi,R,P,GAMMA,THRESHOLD)
        print(f"interation {iteration}")
        print(V)
        is_stable = policy.improvement(is_stable,A_,S_,V,pi,R,P,GAMMA)
        if(not(is_stable)):
            print("policy improved!")
        else:
            print("policy NOT improved!")
        print(pi)


    problem.print_V(V)
    problem.print(pi)
    return 

if __name__ == "__main__":
    main()
