import fire


def add(x, y):
    return x + y


def mul(**kwargs):
    a = kwargs['a']
    b = kwargs['b']
    return a * b


if __name__ == '__main__':
    fire.Fire()


"""
在该目录下执行
poetry run python example.py add 1 2 # 执行add(1, 2)
poetry run python example.py mul --a=1 --b=2 # 执行mul(a=1, b=2), kwargs={'a':1, 'b':2}
poetry run python example.py add --x=1 --y==2 # 执行add(x=1, y=2)
"""