def check(s, filename):
    words = s.split()
    words = sorted(list(set([x.lower() for x in words])))
    vals = [0 for _ in range(0, len(words))]
    
    for word in s.split():
        for i in range(0, len(words)):
            if words[i] == word.lower():
                vals[i] += 1
                break
    
    with open(filename, 'w') as f:
        for i in range(len(words)):
            f.write(words[i] + ' ' + str(vals[i]) + '\n')
