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


class undergraduate(Student):

    def studyClass(self):
        pass

    def attendActivity(self):
        pass


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

    print("---- getattr() ----")
    print(getattr(xiaoming, 'name'))

    print("---- global() ----")
    # print(globals())

    print("---- hasattr() ----")
    print(hasattr(xiaoming, 'name'))
    print(hasattr(xiaoming, 'id'))

    print("---- hash() ----")
    print(hash(xiaoming))

    print("---- help() ----")
    print(help(xiaoming))

    print("---- id() ----")
    print(id(xiaoming))

    print("---- input() ----")
    # input_ = input()
    # print(input_)

    print("---- int() ----")
    print(int('12', base=10))

    print("---- isinstance() ----")
    print(isinstance(xiaoming, Student))

    print("---- issubclass() ----")
    print(issubclass(undergraduate, Student))
    print(issubclass(Student, object))
    print(issubclass(object, Student))

    print("---- iter() ----")
    lst = [1, 3, 5]
    for i in iter(lst):
        print(i)

    print("---- len(s) ----")
    dic = dict(zip(['a', 'b'], [1, 2]))
    print(len(dic))

    print("---- list() map()----")
    a = list(map(lambda x: x % 2 == 1, [2, 3, 4, 5, 6, 7]))
    b = list(map(lambda x, y: x % 2 == 1 and y % 2 == 0, [1, 3, 2, 4, 1], [3, 2, 1, 2, 4]))
    print(a)
    print(b)

    print("---- max() min() ----")
    print(max(3, 1, 4, 2, 1))
    print(max((), default=0))
    print(max({'a': 3, 'b1': 5, 'c': 4}))
    a = [{'name': 'xiaoming', 'age': 18, 'gender': 'male'}, {'name': 'xiaohong', 'age': 20, 'gender': 'female'}]
    print(max(a, key=lambda x: x['age']))

    print("---- memoryview(obj) ----")

    print("---- next() ----")
    print("---- object() ----")
    print("---- openfile() ----")

    print("---- pow() ----")
    print(pow(3, 2))
    print(pow(3, 2, 4))

    print("----  class property(fget=None, fset=None, fdel=None, doc=None) ----")

    print("---- range(stop) ----")

    print("---- reverse(seq)----")
    print(list(reversed([1, 2, 3, 4, 5])))

    print("---- round() ----")
    print(round(10.123456789, 2))

    print("---- class set([iterable]) ----")
    a = [1, 2, 3, 2, 1]
    print(set(a))

    print("---- class slice(stop) ----")
    a = [1, 4, 2, 3, 1]
    print(a[slice(0, 5, 2)])
    print(a[0:5:2])

    print("---- sorted() ----")
    print(sorted(a, reverse=True))
    a = [{'name': 'xiaoming', 'age': 18, 'gender': 'male'}, {'name': 'xiaohong', 'age': 20, 'gender': 'female'}]
    print(sorted(a, key=lambda x: x['age']))

    print("---- staticmethod ----")
    print("---- vars() ----")
    # print(vars())

    print("---- sum() ----")

    print("---- super() ----")

    print("---- tuple() ----")

    print("---- type() ----")

    print("---- zip() ----")
    x = [3, 2, 1]
    y = [4, 5, 6]
    print(list(zip(y, x)))

    print("---- 列表生成式 ----")
    print(list(range(11)))
    print([x**2 for x in range(11)])
    print([x**2 for x in range(11) if x % 2 == 0])
    a = range(5)
    b = ['a', 'b', 'c', 'd', 'e']
    c = [str(y) + str(x) for x, y in zip(a, b)]
    print(c)

    print("---- Collections ----")

    print("---- NamedTuple ----")
    from collections import namedtuple
    Person = namedtuple('Person', ['age', 'height', 'name'])
    data = [Person(10, 1.4, 'xiaoming'), Person(12, 1.5, 'xiaohong')]
    print(data[0].age)
    print(data[1].height)

    print("---- Counter ----")
    from collections import Counter
    skuPurchaseCount = [3, 8, 3, 10, 3, 3, 1, 3, 7, 6, 1, 2, 7, 0, 7, 9, 1, 5, 1, 0]
    print(Counter(skuPurchaseCount).most_common())

    print("---- itertools ----")
    print("---- chain ----")
    from itertools import chain
    print(list(chain(['I', 'love'], ['python'], ['very', 'much'])))
    print("---- accumulate ----")
    from itertools import accumulate
    print(list(accumulate([1, 2, 3, 4, 5, 6], lambda x, y: x*y)))
    print("---- compress ----")
    from itertools import compress
    print(list(compress('abcdefg', [1, 1, 0, 1, 1])))



