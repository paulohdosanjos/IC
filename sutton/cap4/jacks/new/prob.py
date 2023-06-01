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


def P1_(p1,x,a):
    if(p1 < x-a):
        return poisson.pmf(p1,lp1)
    elif (p1 == x-a):
        return 1 - poisson.cdf(p1,lp1) + poisson.pmf(p1,lp1) #Pr{P1 >= p1}
    else:
        return 0

def P2_(p2,y,a):
    if(p2 < y+a):
        return poisson.pmf(p2,lp2)
    elif (p2 == y+a):
        return 1 - poisson.cdf(p2,lp2) + poisson.pmf(p2,lp2) #Pr{P2 >= p2}
    else:
        return 0

def D1_(d1,x,a,p1):
    if(d1 < 0): return 0
    if(d1 < MAX_CARS - (x-a-p1)):
        return poisson.pmf(d1,ld1)
    elif(d1 == MAX_CARS - (x-a-p1)):
        return 1 - poisson.cdf(d1,ld1) + poisson.pmf(d1,ld1)
    else:
        return 0

def D2_(d2,y,a,p2):
    if(d2 < MAX_CARS - (y+a-p2)):
        return poisson.pmf(d2,ld2)
    elif(d2 == MAX_CARS - (y+a-p2)):
        return 1 - poisson.cdf(d2,ld2) + poisson.pmf(d2,ld2)
    else:
        return 0

def calc_p1():
    """
    calculate p(x'|x,a) matrix the right way, i hope
    """
    n = MAX_CARS+1
    P1 = np.zeros((n,n,2*MAX_MOVES+1))

    for x in range(n):
        i = 0 
        for a in range(-MAX_MOVES,MAX_MOVES+1):
            if(x-a > MAX_CARS or a > x): 
                i += 1
                continue
            sum = 0
            for xf in range(n):
                if(xf == n-1): 
                    P1[xf][x][i] = 1 - sum
                    print(f"P1({xf}|{x},{a}) = {P1[xf][x][i]}, i = {i}")
                    if( x == 19 and a == -1):
                        print("here")
                        print(P1[xf][x][i])
                        print(i)
                    continue
                P1[xf][x][i] = calc_p1_(xf,x,a)
                print(f"P1({xf}|{x},{a}) = {P1[xf][x][i]}, i = {i}")
                sum += P1[xf][x][i]
            i += 1

    return P1


def calc_p1_(xf,x,a):
    """
    calculate p(x'|x,a)
    """
    sum = 0
    n = MAX_CARS + 1
    for p1 in range(x-a+1): #posso deixar mais rápido usando MAX_ARG
        #print(f"p1 = {p1}")
        A = P1_(p1,x,a)
        #print(f"p1+xf-x+a={p1+xf-x+a},x={x},a={a},p1={p1}")
        B = D1_(p1+xf-x+a,x,a,p1)
        #print(f"P1_ = {A}, D1_ = {B}")
        sum += A*B
    return sum

def calc_p2():
    """
    calculate p(y'|y,a) matrix the right way, i hope
    """
    n = MAX_CARS+1
    P2 = np.zeros((n,n,2*MAX_MOVES+1))

    for y in range(n):
        i = 0 
        for a in range(-MAX_MOVES,MAX_MOVES+1):
            if(y+a > MAX_CARS or a < -y): 
                i += 1
                continue
            sum = 0
            for yf in range(n):
                if(yf == n-1): 
                    P2[yf][y][i] = 1 - sum
                    print(f"P2({yf}|{y},{a}) = {P2[yf][y][i]}, i = {i}")
                    continue
                P2[yf][y][i] = calc_p1_(yf,y,a)
                print(f"P2({yf}|{y},{a}) = {P2[yf][y][i]}, i = {i}")
                sum += P2[yf][y][i]
            i += 1

    return P2


def calc_p2_(yf,y,a):
    """
    calculate p(y'|y,a)
    """
    sum = 0
    n = MAX_CARS + 1
    for p2 in range(y+a+1): #posso deixar mais rápido usando MAX_ARG
        #print(f"p2 = {p2}")
        A = P2_(p2,y,a)
        #print(f"p2+yf-y+a={p1+xf-x+a},x={x},a={a},p1={p1}")
        B = D2_(p2+yf-y-a,y,a,p2)
        #print(f"P1_ = {A}, D1_ = {B}")
        sum += A*B
    return sum

def calc_p(P1,P2):
    """
    calculate p(s'|s,a) matrix
    """
    n = MAX_CARS + 1
    P = np.zeros((n,n,n,n,2*MAX_MOVES+1))
    for xf in range(n):
        for yf in range(n):
            for x in range(n):
                for y in range(n):
                    i = 0
                    for a in range(-MAX_MOVES,MAX_MOVES+1):
                        if(a < -y or a > x):
                            P[xf][yf][x][y][i] = 0
                            print(f"P[{xf}][{yf}][{x}][{y}][{a}] = {P[xf][yf][x][y][i]}")
                            i += 1
                            continue
                        P[xf][yf][x][y][i] = P1[xf][x][a]*P2[yf][y][a]
                        print(f"P[{xf}][{yf}][{x}][{y}][{a}] = {P[xf][yf][x][y][i]}")
                        i += 1
    return P

def main():

    n = 1 + MAX_CARS

    P1 = calc_p1()
    P2 = calc_p2()

    
    print()
    print()

    #x = 19 
    #a = -3 
    #i = MAX_MOVES + a
    #sum = 0
    #for xf in range(n):
    #    prob = P1[xf][x][i] 
    #    sum += prob
    #    print(f"p({xf}|{x},{a}) = {prob}")
    #print(f"sum(p(x',{x},{a})) = {sum}")

    #return

    #for x in range(n):
    #    for a in range(-MAX_MOVES,MAX_MOVES+1):
    #        #dado um x' e a 
    #        i = +MAX_MOVES + a
    #        sum = 0
    #        for xf in range(n):
    #            sum += P1[xf][x][i]
    #        print(f"sum(p(x',{x},{a}))= {sum}")

    #for y in range(n):
    #    for a in range(-MAX_MOVES,MAX_MOVES+1):
    #        #dado um y' e a 
    #        i = +MAX_MOVES + a
    #        sum = 0
    #        for yf in range(n):
    #            sum += P2[yf][y][i]
    #        print(f"sum(p(y',{y},{a}))= {sum}")


    ##print(P1)

    #return

    P = calc_p(P1,P2) #p(s'|s,a)

    #testing P

    return

if __name__ == "__main__":
    main()

