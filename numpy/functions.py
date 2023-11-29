from typing import List


def sum_non_neg_diag(x: List[List[int]]) -> int:
    length = min(len(x), len(x[0]))
    diag = [x[i][i] for i in range(length) if x[i][i] >= 0]
    return sum(diag) if diag else -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    return sorted(x) == sorted(y)


def max_prod_mod_3(x: List[int]) -> int:
    x1 = x[1:]
    x2 = x[:-1]
    prod = [x1[i] * x2[i] for i in range(0, len(x1)) if (x1[i] % 3 == 0) or (x2[i] % 3 == 0)]
    return max(prod) if prod else -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    res = []
    for matrix in image:
        newrow = []
        for row in matrix:
            s = 0
            for i in range(len(row)):
                s += row[i] * weights[i]
            newrow.append(s)
        res.append(newrow)
    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:    
    x = [i[0] for i in x for _ in range(i[1])]
    y = [i[0] for i in y for _ in range(i[1])]
    
    if len(x) != len(y):
        return -1
    
    res = 0
    
    for i in range(len(x)):
        res += x[i] * y[i]

    return res

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    res = []
    
    for i in range(len(X)):
        new_row = []
        for j in range(len(Y)):
            dist, scalar, xnorm, ynorm = [0] * 4
            for k in range(len(X[0])):
                xnorm += X[i][k] ** 2
                ynorm += Y[j][k] ** 2
                scalar += X[i][k] * Y[j][k]
            
            if xnorm and ynorm:
                dist = scalar / (xnorm * ynorm) ** (1/2)
            else:
                dist = 1
            
            new_row.append(dist)
        res.append(new_row)
    
    return res
