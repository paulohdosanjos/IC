def evaluation(S_,V,pi,R,P,GAMMA,THRESHOLD):
    print("evaluating policy...")
    while(True):
        delta = 0
        for _s in range(len(S_)):
            v = V[_s]
            #updates V(s)
            sum_a = 0
            for _a in range(len(pi)):
                sum_s = 0 
                for _sf in range(len(S_)):
                    sum_s += P[_sf][_s][_a]*V[_sf]
                sum_a += pi[_a][_s]*R[_s][_a] + pi[_a][_s]*GAMMA*sum_s 
            V[_s] = sum_a
            delta = max(delta,abs(v-V[_s]))

        if(delta < THRESHOLD):
            break
        else:
            print(f"not done yet: delta = {delta}")

    return V
