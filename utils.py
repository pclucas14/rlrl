def pprint(arr):
    def fill(x):
        x = '{:.2f}'.format(x)
        for _ in range(8 - len(x)):
            x += ' '    
        return x        

    out = ''
    for line in arr:
        st = '['
        for no in line:
            st += fill(no)
        
        out += st[:-2] + ']\n'

    return out
