import numpy as np

np.random.seed(1)


def sigmoid(net):
    return 1 / (1 + np.exp(-net))


def relu(net):
    return np.maximum(0, net)


train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')

# 中间层0的权值
wp0 = 2 * np.random.random((50, 55)) - 1

# 中间层0的阈值
b0 = 0.1 * np.ones((55,))

# 中间层1的权值
wp1 = 2 * np.random.random((55, 42)) - 1

# 中间层1的阈值
b1 = 0.1 * np.ones((42,))

# 输出层的权值
wp2 = 2 * np.random.random((42, 40)) - 1

# 输出层的阈值
b2 = 0.1 * np.ones((40, ))

# 训练次数
times = 10000

# 学习率
alpha = 0.9

# 动量因子
momentum = 0.9

# 前一次的delta_w0
last_w0 = np.zeros([50, 55])

# 前一次的delta_w1
last_w1 = np.zeros([55, 42])

# 前一次的delta_w2
last_w2 = np.zeros([42, 40])

for i in range(0, times):

    # alpha *= 0.8

    for j in range(0, 280):

        net0 = np.dot(train_data[j], wp0) + b0
        r0 = sigmoid(net0)

        net1 = np.dot(r0, wp1) + b1
        r1 = sigmoid(net1)

        net2 = np.dot(r1, wp2) + b2
        r2 = sigmoid(net2)
        out = r2

        cost = out - train_label[j]
        print('Error {}'.format(np.mean(np.abs(cost))))

        delta_r2 = cost * r2 * (1 - r2)
        delta_b2 = delta_r2
        delta_w2 = np.dot(np.mat(r1).T, np.mat(delta_r2))

        delta_r1 = np.dot(delta_r2, wp2.T) * r1 * (1 - r1)
        delta_b1 = delta_r1
        delta_w1 = np.dot(np.mat(r0).T, np.mat(delta_r1))

        delta_r0 = np.dot(delta_r1, wp1.T) * r0 * (1 - r0)
        delta_b0 = delta_r0
        delta_w0 = np.dot(np.mat(train_data[j]).T, np.mat(delta_r0))

        last_w0 = alpha * delta_w0 - momentum * last_w0
        wp0 -= last_w0
        b0 -= alpha * delta_b0
        last_w1 = alpha * delta_w1 - momentum * last_w1
        wp1 -= last_w1
        b1 -= alpha * delta_b1
        last_w2 = alpha * delta_w2 - momentum * last_w2
        wp2 -= last_w2
        b2 -= alpha * delta_b2
else:
    count = 0
    for i in range(0, 120):
        net0 = np.dot(test_data[i], wp0) + b0
        r0 = sigmoid(net0)

        net1 = np.dot(r0, wp1) + b1
        r1 = sigmoid(net1)

        net2 = np.dot(r1, wp2) + b2
        r2 = sigmoid(net2)

        x = r2.tolist()
        y = test_label[i].tolist()
        a = x.index(max(x))
        b = y.index(max(y))

        if a == b:
            count += 1

    print('正确个数：')
    print(count)
    np.savez('result.npz', wp0, wp1, wp2, b0, b1, b2)

