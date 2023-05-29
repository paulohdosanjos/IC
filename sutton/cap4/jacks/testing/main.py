import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

THRESHOLD = 0.1
PROB_THRESHOLD = 0.0001
INFTY = 1000000000
MAX_ARG = 10 # percent point function (0.999) for poisson distribution
MAX_CARS = 20 # maximum number of cars in any location at a given time
MAX_MOVES = 5 #maximum number of cars moves per night
GAMMA = 0.9
R_MIN = -2*MAX_MOVES
R_MAX = 20*MAX_ARG+2*MAX_MOVES
A_MIN = -MAX_MOVES
A_MAX = MAX_MOVES
ld1 = 3
ld2 = 2
lp1 = 3
lp2 = 4

file = open("output.txt", "r")
P = file.read()


S = [] #set of states
for i in range(MAX_CARS+1):
    for j in range(MAX_CARS+1):
        S.append([i,j])

def p(sf,r,s,a):
    

def policy_evaluation(V,pi):
    n = len(V)

    delta = 0
    while(delta > THRESHOLD):
        for i in range(n):
            for j in range(n):
                v = V[i][j]
                
                #updates V(s)
                sum = 0
                for ii in range(n):
                    for jj in range(n):
                        for r in R: 
                            sum += P[ii,jj,r,i,j,pi[i][j]]*(r+GAMMA*V[ii][jj])
                V[i][j] = sum

                delta = max(delta, abs(v-V[i][j]))
    print("policy evaluated!")
    print(V)

    return

def policy_improvement(V,pi):
    print("improving policy...")

    n = len(V)
    policy_stable = True

    for i in range(n):
        for j in range(n):
            print(f"V({i})({j})")
            a = pi[i][j]
            max_arg = a
            max_sum = -INFTY
            for a in range(A_MIN,A_MAX+1):
                sum = 0
                for ii in range(n):
                    for jj in range(n):
                        for r in R:
                            print(f"({ii},{jj},{r})")
                            sum += P[ii,jj,r,i,j,pi[i][j]]*(r+GAMMA*V[ii][jj])
                            #print(f"p([{ii},{jj}],{r}|[{i},{j}],{pi[i][j]})={p([ii,jj],r,[i,j],pi[i][j])}")

                if(sum > max_sum):
                    max_arg = a

            pi[i][j] = max_arg
            if(a != pi[i][j]):
                policy_stable = False


    if(not(policy_stable)):
        policy_evaluation(V,pi)
    else:
        print("policy improved!")
        print(pi)
        return

def main():
    n = 1 + MAX_CARS
    V = np.zeros((n,n)) #inicialize value function with zeroes
    pi = np.zeros((n,n)) #inicialize policy with a = 20 for all s


    policy_evaluation(V,pi)
    policy_improvement(V,pi)

    print(V)
    print(pi)

    return

if __name__ == "__main__":
    main()

