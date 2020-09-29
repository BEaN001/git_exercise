from graphviz import Digraph

# 实例化一个Digraph对象(有向图)，name:生成的图片的图片名，format:生成的图片格式
dot = Digraph(name="MyPicture", comment="the test", format="png")

# 生成图片节点，name：这个节点对象的名称，label:节点名,color：画节点的线的颜色
dot.node(name='a', label='Ming', color='green')
dot.node(name='b', label='Hong', color='yellow')
dot.node(name='c', label='Dong')

# 在节点之间画线，label：线上显示的文本,color:线的颜色
dot.edge('a', 'b', label="ab\na-b", color='red')
# 一次性画多条线，c到b的线，a到c的线
dot.edges(['cb', 'ac'])

# 打印生成的源代码
print(dot.source)

# 画图，filename:图片的名称，若无filename，则使用Digraph对象的name，默认会有gv后缀
# directory:图片保存的路径，默认是在当前路径下保存
dot.view(filename="mypicture", directory=".")

# 跟view一样的用法(render跟view选择一个即可)，一般用render生成图片，不使用view=True,view=True用在调试的时候
dot.render(filename='MyPicture', directory="./My",view=True)