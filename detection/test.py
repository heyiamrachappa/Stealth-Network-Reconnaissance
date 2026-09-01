def dfs(v):
    seen.add(v)
    print(v,end=" ")
    for x in g[v]:
        if x not in seen: dfs(x)

g={0:[1,2],1:[3],2:[],3:[]}
seen=set()
dfs(0)