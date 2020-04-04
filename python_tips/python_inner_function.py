class Student():
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def __repr__(self):
        return 'id = ' + self.id + ', name = ' + self.name

    @classmethod
    def class_f(cls):
        print("classmethod!")
        print(cls)
        print("classmethod!")


if __name__ == "__main__":
    print("---- abs() -----")
    print(abs(-6))

    print("---- all() -----")
    print(all([1, 0, 3, 6]), all([1, 2, 3]))

    print("---- any() ----")
    print(any([0, 0, 0, []]), any([0, 0, 1]))

    print("----ascii()----")
    xiaoming = Student('001', 'xiaoming')
    print([xiaoming], type(xiaoming))
    print([ascii(xiaoming)], type(ascii(xiaoming)))

    print("----bin() 二进制----")
    print(bin(10))

    print("---oct()  八进制----")
    print(oct(10))

    print("----hex()  十六进制----")
    print(hex(10))

    print("---- bool() ----")
    print(bool(1))
    print(bool([1, 1, 0]))
    print(bool([]))
    print(bool([0, 0, 0]))

    print("---- bytes() ----")
    s = "apple"
    print(bytes(s, encoding='utf-8'))

    print("---- str() ----")
    integ = 100
    print([str(integ)])

    print("---- callable() ----")
    print(callable(1))
    print(callable(str))
    print(callable(int))
    print(callable(xiaoming))

    print("---- chr() ---- 十进制对应ASCII字符")
    print(chr(65))
    print(chr(64))
    print(chr(66))

    print("---- ord() ---- ASCII字符对应十进制")
    print(ord('A'))
    print(ord(':'))

    print("---- classmethod() ----")
    print(Student.class_f())

    print("---- compile() exec() ----")
    s = "print('helloworld')"
    r = compile(s, "<string>", "exec")
    print(r)
    exec(r)

    print("---- complex() ----")
    print(complex(1, 2))

    print("---- delattr() ----")
    print(xiaoming, xiaoming.id, xiaoming.name)
    delattr(xiaoming, 'id')
    print(xiaoming.name)
    print(hasattr(xiaoming, 'id'))

    print("---- dict() ----")
    print(dict())
    print(dict(a=1, b=2))
    print(dict(zip(['a', 'b'], [1, 2])))
    print(dict([('a', 1), ('b', 2)]))

    print("---- dir() ----")
    print(dir(xiaoming))

    print("---- divmod() ----")
    print(divmod(10, 3), type(divmod(10, 3)))

    print("---- enumuerate() ----")
    s = ["a", "b", "c"]
    for i, v in enumerate(s, 10):
        print(i, v)

    print("---- eval() ----")
    s = "1 + 3 + 5"
    print(eval(s))

    print("---- filter ----")
    print("---- [ item for item in iterables if function(item)] ----")
    fil = filter(lambda x: x > 10, [1, 11, 5, 6, 23])
    print(fil, list(fil))

    print("---- float ----")
    print(float(3))

    print("---- format() ----")
    print("i am {}, age{}".format("tom", 18))

    print("---- frozenset() ----")
    print(frozenset([1, 1, 3, 2, 3, 4]))
    print(set('set'))




