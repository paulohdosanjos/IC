import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

THRESHOLD = 0.1
PROB_THRESHOLD = 0.0000001
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


def P1(p1,x,a):
    if(p1 < x-a):
        return poisson.pmf(p1,lp1)
    elif (p1 == x-a):
        return 1 - poisson.cdf(p1,lp1) + poisson.pmf(p1,lp1) #Pr{P1 >= p1}
    else:
        return 0

def P2(p2,y,a):
    if(p2 < y+a):
        return poisson.pmf(p2,lp2)
    elif (p2 == y+a):
        return 1 - poisson.cdf(p2,lp2) + poisson.pmf(p2,lp2) #Pr{P2 >= p2}
    else:
        return 0

def D1(d1,x,a,p1):
    if(d1 < MAX_CARS - (x-a-p1)):
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

    xf, yf = sf
    x , y = s

    prob = 0

    num_request = (r+2*abs(a))//10

    if(10*num_request != r + 2*abs(a)): #num_request not integer: invalid r
        return 0

    if(not(a >=-min(y,MAX_MOVES) and a <= min(x,MAX_MOVES))): #condition (2)
       return 0

    for p1 in range(min(x-a,MAX_ARG)+1): #condition (1)

        P1_ = P1(p1,x,a)

        P2_ = P2(num_request-p1,y,a)

        D1_ = D1(xf-x+a+p1,x,a,p1)

        D2_ = D2(num_request-p1+yf-y-a,y,a,num_request-p1)

        prob += P1_*P2_*D1_*D2_ 

    return prob

def main():
    P = {} #dictonary with non-zero probabilities

    for s in S:
        x,y = s
        for a in range(-min(y,MAX_MOVES),min(x,MAX_MOVES)+1):
            sum = 0
            for num_request in range(2*MAX_ARG+1): 
                r = 10*num_request - 2*abs(a)
                for sf in S:
                    prob = p(sf,r,s,a)
                    sum += prob
                    #print(f"P[({sf[0]},{sf[1]}),{r},({s[0]},{s[1]}),{a}]={prob}")
                    if(prob > PROB_THRESHOLD):
                        P[(sf[0],sf[1],r,s[0],s[1],a)] = prob
        print(f"sum_prob({s}) = {sum}")
    print(P)

if __name__ == "__main__":
    main()


