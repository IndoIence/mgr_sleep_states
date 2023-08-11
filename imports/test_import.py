def where_only_one_spike(indicies, padding=100):
    hits = []
    prev = indicies[0]
    cur = indicies[1]
    next = indicies[2]
    for i in range(2,len(indicies)-1): # those boundaries don't work for 3 ind list
        if cur - prev > padding and next - cur > padding:
            hits.append(cur)
        prev = cur
        cur = next
        next = indicies[i + 1]
    return hits
