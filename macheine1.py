from functools import reduce


# 修改每次的self.weights和bias 进行训练
class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知机，设置输入参数的个数，以及激活函数。激活函数的类型为double->double
        '''
        self.activator = activator
        # 权重向量初始化为0，两个0.0
        self.weights = [0.0 for _ in range(input_num)]  #出现两个0.0
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        输入向量，输出感知机的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3...]
        # 变成[(x1,w1),(x2,w2)...]
        # 然后利用reduce求和
        # print(self.activator(reduce(lambda a_b: a_b[0]+a_b[1],[x*w for x,w in zip(input_vec,self.weights)],0.0)+self.bias))
        # zip 对列表进行操作的时候 有0的话会自动省略
        return self.activator(reduce(lambda a, b: a + b, [x * w for x, w in
                                                          zip(input_vec, self.weights)], 0.0) + self.bias)

    def _on_iteration(self, input_vecs, labels, rate):
        # 每个输入和输出打包在一起，成为样本的列表[(input_vec,label)]
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            # 计算感知机在当前权重下的输出
            output = self.predict(input_vec)
            # output是0或者为1
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据:一组向量、与每个向量对应的label;以及训练轮数、学习率
        '''
        # iteration 次数 循环的次数
        # rate 步长，一步的步率
        # input_vecs 输入的数据 一组向量
        #  lables 每个向量对应的labels
        for i in range(iteration):
            self._on_iteration(input_vecs, labels, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知机规则更新权重
        # 首先计算本次更新的delta
        # 然后把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重更新
        # 最后再把权重更新按元素加到原先的weights[w1,w2,w3,...]上
        '''
        delta = label - output
        print(delta)
        # delta 是标签值-当前的感知器输出的值
        self.weights = [(w + rate * delta * x) for x, w in zip(input_vec, self.weights)]
        print(self.weights)
        self.bias += rate * delta
        print(self.__str__())


def f(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    # 1,1-1   0,0-0    1,0-0    0,1-0
    return input_vecs, labels


def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)

    print('1 and 1 = %d' % and_perception.predict([1, 1]))