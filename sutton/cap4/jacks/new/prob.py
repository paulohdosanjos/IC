import numpy as np
from scipy.stats import poisson
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://matplotlib-backend-kitty')

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
                    for a in range(-MAX_MOVES,MAX_MOVES+1):
                        i = MAX_MOVES + a
                        if(a < -y or a > x):
                            P[xf][yf][x][y][i] = 0
                            print(f"P(({xf},{yf})|({x},{y}),{a}) = {P[xf][yf][x][y][i]}")
                            continue
                        P[xf][yf][x][y][i] = P1[xf][x][i]*P2[yf][y][i]
                        print(f"P(({xf},{yf})|({x},{y}),{a}) = {P[xf][yf][x][y][i]}")
    return P

def calc_r():
    """
    calculate r(s,a) matrix
    """
    n = MAX_CARS + 1
    R = np.zeros((n,n,2*MAX_MOVES+1))

    for x in range(n):
        for y in range(n):
            for a in range(-MAX_MOVES,MAX_MOVES+1):
                i = MAX_MOVES + 1
                if(y+a > MAX_CARS or x-a > MAX_CARS):
                    R[x][y][i] = 0
                    print(f"r(({x},{y}),{a}) = {R[x][y][i]}")
                    continue
                if(a < -y or a > x):
                    R[x][y][i] = 0
                    print(f"r(({x},{y}),{a}) = {R[x][y][i]}")
                    continue
                R[x][y][i] = calc_r_(x,y,a)
                print(f"r(({x},{y}),{a}) = {R[x][y][i]}")

    return R

def calc_r_(x,y,a):
    sum = 0
    for r in range(-2*abs(a),-2*abs(a)+10*(x+y),10):
        for p1 in range(x-a+1):
            num_request = (r+2*abs(a))//10
            sum += r * P1_(p1,x,a) * P2_(num_request - p1, y,a)
    return sum

def policy_evaluation(V,pi,R,P):
    n = MAX_CARS + 1 

    print("evaluating policy...")
    while(True):
        delta = 0
        total = 0
        for x in range(n):
            for y in range(n):
                v = V[x][y]
                a = int(pi[x][y])
                i = MAX_MOVES + a

                #updates V(s)
                sum = 0
                for xf in range(n):
                    for yf in range(n):
                        sum += V[xf][yf]*P[xf][yf][x][y][i]

                print(f"r(({x},{y}),{a}) = {R[x][y][i]}")
                print(f"sum = {sum}")
                V[x][y] = R[x][y][i] + sum * GAMMA
                delta = max(delta,abs(v-V[x][y]))
                total += 1
                #print(f"{total}/{n*n}")
    
        if(delta < THRESHOLD):
            break
        else:
            print(f"not done yet: delta = {delta}")

    return

def policy_improvement(V,pi,policy_stable,R,P):
    print("improving policy...")

    n = MAX_CARS + 1 
    policy_stable = True
    total = 0

    for x in range(n):
        for y in range(n):
            a_past = pi[x][y]
            max_arg = a_past
            max_sum = -INFTY
            for a in range(-MAX_MOVES,MAX_MOVES+1):
                i = MAX_MOVES + 1
                sum = 0
                for xf in range(n):
                    for yf in range(n):
                        sum += V[xf][yf]*P[xf][yf][x][y][i]
                sum = R[x][y][i] + sum * GAMMA
                if(sum > max_sum):
                    max_arg = a
                    max_sum = sum

            pi[x][y] = max_arg
            if(a_past != pi[x][y]):
                policy_stable = False
            total += 1
            if(total == (n*n)//4): print("25%")
            if(total == (n*n)//2): print("50%")
            if(4*total == 3*(n*n)): print("75%")
            if(total == (n*n)): print("100%")
            #print(f"{total}/{n*n}")
    return

def show(pi):

    n = len(pi)

    cmap = plt.cm.get_cmap('RdYlBu',11)
    norm = plt.Normalize(-5, 5)  # Normalizar os valores de pi(x, y) entre -5 e 5

    plt.imshow(pi,cmap = cmap, extent = [0,n-1,0,n-1])

    cbar = plt.colorbar()
    cbar.set_label('pi(x, y)')

    # Configurar os rótulos dos eixos
    plt.xticks(np.arange(n))
    plt.yticks(np.arange(n))
    plt.xlabel('x')
    plt.ylabel('y')

    # Configurar os limites dos eixos
    plt.xlim(0, n-1)
    plt.ylim(0, n-1)

    plt.show()

def main():

    n = 1 + MAX_CARS

    P1 = calc_p1()
    P2 = calc_p2()

    
    #print()
    #print()

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

    #debuging p(20,20|20,19,5) should be zero because y+a > MAX_CARS
    #xf = 20; yf = 20
    #x = 20; y = 19
    #a = -1
    #i = MAX_MOVES + a

    #print()
    #print()
    #print(f"p({xf},{yf}|{x},{y},{a}) = {P[xf][yf][x][y][i]}")

    #print(f"p(x' = {xf}|x = {x},a = {a}) = {P1[xf][x][i]}")

    #print(f"p(y' = {yf}|y = {y},a = {a}) = {P2[yf][y][i]}")
    #return


    #checking sums

    #for x in range(n):
    #    for y in range(n):
    #        for a in range(-MAX_MOVES,MAX_MOVES+1):
    #            i = MAX_MOVES + a
    #            sum = 0
    #            for xf in range(n):
    #                for yf in range(n):
    #                    sum += P[xf][yf][x][y][i]
    #            print(f"sum(p(s'|({xf},{yf}),{a}))={sum}")

    R = calc_r()



    V = np.zeros((n,n)) #inicialize value function with zeroes
    pi = np.zeros((n,n)) #inicialize policy with a = 0 for all s

    print()
    print(f"pi[0]:\n {pi}")
    #show(pi)

    print()
    print(f"V[0]:\n {V}")

    policy_stable = False
    i = 1
    while(not(policy_stable)):
        policy_evaluation(V,pi,R,P)
        print()
        print("value function evaluated!")
        print()
        print(f"V[{i}]:\n {V}")
        policy_improvement(V,pi,policy_stable,R,P)
        i += 1
        if(not(policy_stable)):
            print()
            print("policy improved!")
            print()
            print(f"pi[{i}]:\n {pi}")
            #show(pi)


    print()
    print("final policy")
    print(f"pi[{i}]:\n {pi}")
    #show(pi)
    print()
    print("final value function")
    print(f"V[{i}]:\n {V}")

    return


if __name__ == "__main__":
    main()

