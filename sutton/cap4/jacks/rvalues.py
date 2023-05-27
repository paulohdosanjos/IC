R = set() 
for p in range(0,21):
    for a in range(-5,6):
        r = 10*p - 2*abs(a)
        R.add(r)
print(len(R))
print(R)
