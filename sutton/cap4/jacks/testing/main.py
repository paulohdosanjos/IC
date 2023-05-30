import json
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

with open("output.txt", "r") as file:
    content = file.read()
    content_corrigido = content.replace("(", "\"(")
    content_corrigido = content_corrigido.replace(")", ")\"")
#print(content_corrigido)
P = json.loads(content_corrigido)

#print(P)

S = [] #set of states
for i in range(MAX_CARS+1):
    for j in range(MAX_CARS+1):
        S.append([i,j])

def p(xf,yf,r,x,y,a):
    s = f"({xf}, {yf}, {r}, {x}, {y}, {a})"
    if(s in P.keys()):
        return P[s]
    else: return 0
    

def policy_evaluation(V,pi):
    n = len(V)

    print("evaluating policy...")
    while(True):
        delta = 0
        total = 0
        for s in S:
            x,y = s
            v = V[x][y]
            a = int(pi[x][y])
            #updates V(s)
            sum = 0
            for sf in S:
                xf,yf = sf
                r_max = -2*abs(a)+10*(min(x-a,MAX_ARG)+min(y+a,MAX_ARG))
                for r in range(r_max+1):  #melhorar esse range
                    #print(f"{p(xf,yf,r,x,y,a)},{(r+GAMMA*V[xf][yf])}")
                    sum += p(xf,yf,r,x,y,a)*(r+GAMMA*V[xf][yf])
            #print(f"sum = {sum}")
            V[x][y] = sum
            delta = max(delta, abs(v-V[x][y]))
            total += 1
            #print(f"{total}/{n*n}")
    
        if(delta < THRESHOLD):
            break
        else:
            print(f"not done yet: delta = {delta}")

    return

def policy_improvement(V,pi,policy_stable):
    print("improving policy...")

    n = len(V)
    policy_stable = True
    total = 0

    for s in S:
        x,y = s
        a_past = pi[x][y]
        max_arg = a_past
        max_sum = -INFTY
        for a in range(-min(y,MAX_MOVES),min(x,MAX_MOVES)+1):
            sum = 0
            for sf in S:
                xf,yf = sf
                r_max = -2*abs(a)+10*(min(x-a,MAX_ARG)+min(y+a,MAX_ARG))
                for r in range(r_max+1):
                    #print(f"({xf},{yf},{r})")
                    sum += p(xf,yf,r,x,y,a)*(r+GAMMA*V[xf][yf])
                    #print(f"p([{ii},{jj}],{r}|[{i},{j}],{pi[i][j]})={p([ii,jj],r,[i,j],pi[i][j])}")

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
    V = np.zeros((n,n)) #inicialize value function with zeroes
    pi = np.zeros((n,n)) #inicialize policy with a = 20 for all s

    print()
    print(f"pi[0]:\n {pi}")
    show(pi)

    print()
    print(f"V[0]:\n {V}")

    policy_stable = False
    i = 1
    while(not(policy_stable)):
        policy_evaluation(V,pi)
        print()
        print("value function evaluated!")
        print()
        print(f"V[{i}]:\n {V}")
        policy_improvement(V,pi,policy_stable)
        i += 1
        if(not(policy_stable)):
            print()
            print("policy improved!")
            print()
            print(f"pi[{i}]:\n {pi}")
            show(pi)


    print()
    print("final policy")
    print(f"pi[{i}]:\n {pi}")
    show(pi)
    print()
    print("final value function")
    print(f"V[{i}]:\n {V}")

    return

if __name__ == "__main__":
    main()

