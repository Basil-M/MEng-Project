# coding = utf-8

def FreqDict2List(dt):
    return sorted(dt.items(), key=lambda d: d[-1], reverse=True)


def LoadList(fn):
    with open(fn) as fin:
        st = list(ll for ll in fin.read().split('\n') if ll != "")
    return st


def LoadDict(fn, func=str):
    dict = {}
    with open(fn) as fin:
        for lv in (ll.split('\t', 1) for ll in fin.read().split('\n') if ll != ""):
            dict[lv[0]] = func(lv[1])
    return dict
