from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from model import MyModel


def main():
    # 直接提取tf官网的minst手写数据集
    mnist = tf.keras.datasets.mnist

    # 下载并加载数据； 并输入归一化
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0 # [60000,28,28]

    # 添加一个channel维度
    x_train = x_train[..., tf.newaxis] # [60000,28,28,1]
    x_test = x_test[..., tf.newaxis]

    # 生成dataset。训练集：每10000张图片打乱一次，32张一个batch
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # 加载定义的模型
    model = MyModel()

    # 定义损失函数：稀疏的多类别交叉熵函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # 定义 optimizer
    optimizer = tf.keras.optimizers.Adam()

    # 统计训练过程损失和准确率
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 统计测试过程损失和准确率
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 计算损失、求梯度、计算准确率
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions) # 算损失
        gradients = tape.gradient(loss, model.trainable_variables) # 计算每个参数的梯度
        optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # 更新每个参数

        train_loss(loss) # 计算历史的损失
        train_accuracy(labels, predictions) # 计算历史的准确率

    # 计算测试的损失和准确率
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # 清空历史信息
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:# 遍历训练集图像和标签
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


if __name__ == '__main__':
    main()
