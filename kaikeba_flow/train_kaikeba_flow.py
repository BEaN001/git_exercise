from sklearn.datasets import load_boston
from tqdm import tqdm_notebook
import numpy as np
from kaikeba_flow.nn.core import Placeholder, Linear, Sigmoid, MSE
from kaikeba_flow.utils.utilities import topological_sort_feed_dict


def forward(graph_order, monitor=False):
    for node in graph_order:
        if monitor: print('forward compuiting -- {}'.format(node))
        node.forward()


def backward(graph_order, monitor=False):
    for node in graph_order[::-1]:
        if monitor: print('backward computing -- {}'.format(node))
        node.backward()


def run_one_epoch(graph_order, monitor=False):
    forward(graph_order, monitor)
    backward(graph_order, monitor)


def optimize(graph, learning_rate=1e-2):
    # there are so many other update / optimization methods
    # such as Adam, Mom,
    for t in graph:
        if t.is_trainable:
            t.value += -1 * learning_rate * t.gradients[t]


data = load_boston()
X_, y_ = data['data'], data['target']
X_rm = X_[:, 5]

w1_, b1_ = np.random.normal(), np.random.normal()
w2_, b2_ = np.random.normal(), np.random.normal()
w3_, b3_ = np.random.normal(), np.random.normal()

X, y = Placeholder(name='X', is_trainable=False), Placeholder(name='y', is_trainable=False)
w1, b1 = Placeholder(name='w1'), Placeholder(name='b1')
w2, b2 = Placeholder(name='w2'), Placeholder(name='b2')

# build model
output1 = Linear(X, w1, b1, name='linear-01')
output2 = Sigmoid(output1, name='activation')
# output2 = Relu(output1, name='activation')
y_hat = Linear(output2, w2, b2, name='y_hat')
cost = MSE(y, y_hat, name='cost')

feed_dict = {
    X: X_rm,
    y: y_,
    w1: w1_,
    w2: w2_,
    b1: b1_,
    b2: b2_,
}

graph_sort = topological_sort_feed_dict(feed_dict)

epoch = 1000

batch_num = len(X_rm)

learning_rate = 1e-3

losses = []

for e in tqdm_notebook(range(epoch)):
    loss = 0

    for b in range(batch_num):
        index = np.random.choice(range(len(X_rm)))
        X.value = X_rm[index]
        y.value = y_[index]

        run_one_epoch(graph_sort, monitor=False)

        optimize(graph_sort, learning_rate)

        loss += cost.value

    losses.append(loss / batch_num)