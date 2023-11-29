from typing import List


def hello(x = None):
    if x:
        return 'Hello, ' + x + '!'
    else:
        return 'Hello!'
        


def int_to_roman(num):
    res = ''
    while num:
        if num >= 1000: 
            num -= 1000
            res += 'M'
        elif num >= 900:
            num -= 900
            res += 'CM'
        elif num >= 500:
            num -= 500
            res += 'D'
        elif num  >= 400:
            num -= 400
            res += 'CD'
        elif num >= 100:
            num -= 100
            res += 'C'
        elif num >= 90:
            num -= 90
            res += 'XC'
        elif num >= 50:
            num -= 50
            res += 'L'
        elif num >= 40:
            num -= 40
            res += 'XL'
        elif num >= 10:
            num -= 10
            res += 'X'
        elif num >= 9:
            num -= 9
            res += 'IX'
        elif num >= 5:
            num -=5
            res += 'V'
        elif num >= 4:
            num -= 4
            res += 'IV'
        elif num >= 1:
            num -= 1
            res += 'I'
        
    return res



def longest_common_prefix(strs):
    if not strs:
        return ''
    
    for i in range(0, len(strs)):
                   strs[i] = strs[i].strip()
    print(strs)
    i = 1
    res = ''
    running = True
    minlen = min([len(s) for s in strs])
    while running and i <= minlen:
        res = strs[0][0:i]
        for s in strs:
            if s[0:i] != res:
                res = res[0:-1]
                running = False
                break
        i += 1
    return res


def primes():
    i = 2;
    while True:
        flag = True
        for j in range (2, int(i/2 + 1)):
            if i % j == 0:
                flag = False
                break;
        if flag:
            yield i
            
        i += 1


class BankCard:
    def __init__(self, total, limit = -1):
        self.total_sum = total
        self.balance_limit = limit
    
    def put(self, sum_put):
        self.total_sum += sum_put
        print("You put", sum_put, "dollars.")
    
    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            print("Not enough money to spend", sum_spent, "dollars.")
            raise ValueError
        
        self.total_sum -= sum_spent
        print("You spent", sum_spent, "dollars")
    
    def __repr__(self):
        return "To learn the balance call balance."
    
    @property
    def balance(self):
        if self.balance_limit:
            self.balance_limit -= 1
            return self.total_sum
        
        print("Balance check limits exceeded.")
        raise ValueError
        
        
    
    def __add__(self, other):
        if min( self.balance_limit, other.balance_limit ) == -1:
            new_limit = -1
        else:
            new_limit = max( self.balance_limit, other.balance_limit )
        return BankCard(self.total_sum + other.total_sum, new_limit)

