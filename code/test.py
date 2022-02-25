list1 = [0,1,2,3,2,1]
index = []
for i in set(list1):
    print(i)
    print([j for j,x in enumerate(list1) if x==i])
    index.append([j for j,x in enumerate(list1) if x==i])
print(index)