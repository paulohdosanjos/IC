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


def p1(p1,x,a):
    if(p1 < x-a):
        return poisson.pmf(p1,lp1)
    elif (p1 == x-a):
        return 1 - poisson.cdf(p1,lp1) + poisson.pmf(p1,lp1) #pr{p1 >= p1}
    else:
        return 0

def p2(p2,y,a):
    if(p2 < y+a):
        return poisson.pmf(p2,lp2)
    elif (p2 == y+a):
        return 1 - poisson.cdf(p2,lp2) + poisson.pmf(p2,lp2) #pr{p2 >= p2}
    else:
        return 0

def d1(d1,x,a,p1):
    if(d1 < max_cars - (x-a-p1)):
        return poisson.pmf(d1,ld1)
    elif(d1 == MAX_CARS - (x-a-p1)):
        return 1 - poisson.cdf(d1,ld1) + poisson.pmf(d1,ld1)
    else:
        return 0

def D2(d2,y,a,p2):
    if(d2 < MAX_CARS - (y+a-p2)):
        return poisson.pmf(d2,ld2)
    elif(d2 == MAX_CARS - (y+a-p2)):
        return 1 - poisson.cdf(d2,ld2) + poisson.pmf(d2,ld2)
    else:
        return 0

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
    P = {} #dictonary with non-zero probabilities
    total = 0

    for s in S:
        x,y = s
        for a in range(-min(y,MAX_MOVES),min(x,MAX_MOVES)+1):
            sum = 0

            tmp_p1 = []
            for p1 in range(min(x-a,MAX_ARG)+1): 
                tmp_p1.append(P1(p1,x,a))

            r_max = -2*abs(a)+10*(min(x-a,MAX_ARG)+min(y+a,MAX_ARG))
            for r in range(r_max+1): #a lot of invalid r's
                num_request = (r+2*abs(a))//10

                if(10*num_request != r + 2*abs(a)): #num_request not integer: invalid r
                    continue

                tmp_p2 = []

                for p1 in range(min(x-a,MAX_ARG)+1): 
                    P2_ = P2(num_request-p1,y,a)
                    tmp_p2.append(tmp_p1[p1]*P2_)

                for xf in range(MAX_CARS+1):
                    tmp_d1 = [] 
                    for p1 in range(min(x-a,MAX_ARG)+1):
                        D1_ = D1(xf-x+a+p1,x,a,p1)
                        tmp_d1.append(tmp_p2[p1]*D1_)

                    for yf in range(MAX_CARS+1):
                        if(xf-x+a >= MAX_ARG or yf-y-a >= MAX_ARG):
                            continue
                        oti = xf+yf-x-y+num_request
                        if(oti < 0 or oti > 1.2*MAX_ARG):
                            continue
                        prob_p = 0 
                        #print(tmp)
                        for p1 in range(min(x-a,MAX_ARG)+1): 
                            D2_ = D2(num_request-p1+yf-y-a,y,a,num_request-p1)
                            prob_p += D2_*tmp_d1[p1]

                        sf = [xf,yf]
                        #prob = p(sf,r,s,a)
                        #if(abs(prob-prob_p)>PROB_THRESHOLD):
                        #    print("hell no")
                             #print(f"p[({sf[0]},{sf[1]}),{r},({s[0]},{s[1]}),{a}]={prob},p_paralelo[({sf[0]},{sf[1]}),{r},({s[0]},{s[1]}),{a}]={prob_p}")
                            #return
                        #print(f"p[({sf[0]},{sf[1]}),{r},({s[0]},{s[1]}),{a}]={prob_p}")
                        #sum += prob
                        #print(f"p[({sf[0]},{sf[1]}),{r},({s[0]},{s[1]}),{a}]={prob}")
                        if(prob_p > PROB_THRESHOLD):
                            sum += prob_p
                            #print(f"p[({sf[0]},{sf[1]}),{r},({s[0]},{s[1]}),{a}]={prob}")
                            P[(sf[0],sf[1],r,s[0],s[1],a)] = prob_p
        #print(f"sum_prob({s}) = {sum}")
        total += 1
        #print(f"{total}/{((MAX_CARS+1)*(MAX_CARS+1))}")
    print(P)

if __name__ == "__main__":
    main()

