import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

THRESHOLD = 0.1
PROB_THRESHOLD = 0.0001
INFTY = 1000000000
MAX_ARG = 15 # percent point function (0.999) for poisson distribution
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

R = [] #possible values of rewards
for p in range(0,2*MAX_ARG + 1):
    for a in range(A_MIN,A_MAX+1):
        r = 10*p - 2*abs(a)
        if(not(r in R)):
            R.append(r)

S = [] #set of states
for i in range(MAX_CARS+1):
    for j in range(MAX_CARS+1):
        S.append([i,j])




def p(sf,r,s,a):
    """
        p(s',r|s,a)
    """

    xf, yf = sf
    x , y = s

    #print(f"s' = {sf}, r = {r}, s = {s}, a = {a}")

    sum = 0

    num_request = (r+2*abs(a))/10

    #print(f"número total de requests = {tot_requests}")

    for d1 in range(MAX_ARG):
        #print(f"d1 = {d1}")

        D1 = poisson.pmf(d1,ld1)
        D2 = poisson.pmf(-d1 + xf - x + yf - y + num_request, ld2)
        P1 = poisson.pmf(d1 + x - xf - a, lp1)
        P2 = poisson.pmf(-d1 + xf - x + a + num_request, lp2)

        #print(f"D1 = {D1}, D2 = {D2}, P1 = {P1}, P2 = {P2}")

        sum += D1*D2*P1*P2 

        #print(f"+= {sum}")

    return sum


#calculating num_request range set with non tiny probabilities 
NUM_REQUEST_SET = {}
for n in range(0,2*MAX_ARG+1): #all possible values
    prob = 0
    for p1 in range(0,MAX_ARG + 1):
        prob += poisson.pmf(n-p1,lp2)*poisson.pmf(p1,lp1)
    if prob > PROB_THRESHOLD:
        NUM_REQUEST_SET[n] = prob
#print(NUM_REQUEST_SET)






P = {} #dictonary with non-zero probabilities

for s in S:
    x,y = s
    sum = 0
    for a in range(-min(y,MAX_MOVES),min(x,MAX_MOVES)+1):
        for num_request in NUM_REQUEST_SET.keys(): #melhorar esse range
            r = 10*num_request -2*abs(a)
            for xf in range(x-a-num_request,x-a+MAX_ARG+1):
                for yf in range(y+a-num_request,y+a+MAX_ARG+1):
                    sf = [xf,yf]
                    prob = p(sf,r,s,a)
                    sum += prob
                    if(xf > MAX_ARG or xf < 0 or yf > MAX_ARG or yf < 0):
                        continue
                    if(x-xf+y-yf > num_request):
                        continue
                    sf = [xf,yf]
                    prob = p(sf,r,s,a)
                    print(f"({sf},{r},{s},{a}):")
                    if(prob > PROB_THRESHOLD):
                        P[(sf[0],sf[1],r,s[0],s[1],a)] = prob
                        print(f"P[({sf[0]},{sf[1]}),{r},({s[0]},{s[1]}),{a}]={prob}")
    print(f"sum_prob({s}) = {sum}")
print(P)


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
                            sum += p([ii,jj],r,[i,j],pi[i][j])*(r+GAMMA*V[ii][jj])
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
                            sum += p([ii,jj],r,[i,j],pi[i][j])*(r+GAMMA*V[ii][jj])
                            print(f"p([{ii},{jj}],{r}|[{i},{j}],{pi[i][j]})={p([ii,jj],r,[i,j],pi[i][j])}")

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
    #sf = [8,9]
    #r = 44
    #s = [10,4]
    #a = 3
    #print(f"p(s' = {sf}, r = {r} | s = {s}, a = {a}) = {p(sf,r,s,a)}")

    n = 1 + MAX_CARS
    V = np.zeros((n,n)) #inicialize value function with zeroes
    pi = np.zeros((n,n)) #inicialize policy with a = 20 for all s


    policy_evaluation(V,pi)
    policy_improvement(V,pi)

    return

#if __name__ == "__main__":
#    main()
