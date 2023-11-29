def find_modified_max_argmax(L, f):
    a = [f(x) for x in L if x==int(x)]
    return ( max(a), max(enumerate(a), key=lambda x: x[1])[0] ) if a else ()
    
