# -*- coding: utf-8 -*-
# @Time    : 2021/1/10 10:00
# @Author  : He Ruizhi
# @File    : flappy_bird_train.py
# @Software: PyCharm

""" 训练前请先浏览一下game/wrapped_flappy_bird.py，尤其要注意130行不要注释掉
注释掉是为了playGame在结束状态显示正常
"""

import os
current_dir = os.path.dirname(__file__)
import sys
sys.path.append(current_dir + '/game')

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform
# import tensorflow.keras.backend as K  # 用于防内存泄露和清理计算图，防止OOM错误
# import gc  # 用于清理内存
import cv2
import random
import numpy as np
import wrapped_flappy_bird as game
from datetime import datetime

# 导入双端队列，之所以用双端队列是由于要维护一个固定大小的Replay Memory
from collections import deque


ACTIONS = 2  # 允许的动作数量，在Flappy Bird游戏中为【0:不跳跃，1:跳跃】
GAMMA = 0.99  # 观测值的衰减率
OBSERVE = 5000.  # 观察期步数
EXPLORE = 500000.  # 探索期步数
FINAL_EPSILON = 0.0001  # 在探索-利用算法中的最后的探索率
INITIAL_EPSILON = 0.1  # 最开始的探索率，在观察期和探索期，探索率会逐渐减小
# INITIAL_EPSILON = 0  # 查看当前训练效果，需要把这一行打开 ##
# OBSERVE = 500000  # 查看当前训练效果，需要把这一行打开 ##
REPLAY_MEMORY = 50000  # 定义REPLAY_MEMORY的大小
BATCH = 32  # batch的大小
FRAME_PER_ACTION = 1  # 定义游戏每一个动作持续几帧
LEARNING_RATE = 1e-4  # 定义学习率


def creatDQN():
    """ 创建DQN，DQN的输入是连续的4游戏画面，输出是状态动作值函数Q(s,a)。
    即有几种状态，DQN就有几种输出，每一种输出对应采用相应动作后获得的回报。
    DQN实际上就是一个深度卷积网络。
    """
    model = tf.keras.Sequential()
    # 添加第一个卷积层，输入为(80,80,4)【通道4是由于采用连续4帧游戏画面】
    model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), use_bias=True,
                            input_shape=[80, 80, 4], activation='relu',
                            padding='same', kernel_initializer=RandomUniform()))
    assert model.output_shape == (None, 20, 20, 32)
    # 添加池化层
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    assert model.output_shape == (None, 10, 10, 32)
    # 添加卷积层
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), use_bias=True,
                            activation='relu', padding='same',
                            kernel_initializer=RandomUniform()))
    assert model.output_shape == (None, 5, 5, 64)
    model.add(layers.Conv2D(64, (3, 3), strides=1, use_bias=True,
                            activation='relu', padding='same',
                            kernel_initializer=RandomUniform()))
    assert model.output_shape == (None, 5, 5, 64)
    # 将输出拉平
    model.add(layers.Flatten())
    # 添加全连接层
    model.add(layers.Dense(512, use_bias=True, activation='relu',
                           kernel_initializer=RandomUniform()))
    # 添加输出层
    model.add(layers.Dense(ACTIONS, use_bias=True,
                           kernel_initializer=RandomUniform()))
    assert model.output_shape == (None, 2)
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model


def trainDQN():
    """ 训练DQN网络"""
    game_state = game.GameState()  # 开启游戏模拟器，会打开一个游戏窗口，实时显示游戏信息
    D = deque(maxlen=REPLAY_MEMORY)  # 创建双端队列，用于存储replay memory

    # 获取游戏的初始状态，设置动作为不执行跳跃，并将初始状态修改成80*80*4大小
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 意思是采用第一个动作，即不跳跃
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # 使用opencv中相关函数将图片大小调整为(80,80)的灰度图
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # 将图片处理成二值图，在这里，像素点值大于1就被赋值为255
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # 将二值图x_t按照像素点值排列顺序拉开，并复制4次，组合成4通道输入数据
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 创建DQN模型
    model = creatDQN()
    # 用于加载和保存网络参数
    model_filename = current_dir + '/model/DQNModel.h5'
    try:
        model.load_weights(model_filename)
        print('加载模型权重成功！')
    except:
        print('加载模型权重失败！')

    # 开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    while True:
        # 获得模型的输出
        state = s_t.astype('float32').reshape(1, 80, 80, 4)
        readout_t = model(state)
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:  # 执行一个随机动作
                # print('-------------执行随机动作-------------')
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:  # 由神经网络计算的Q(s,a)值选择对应的动作
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # 什么也不做

        # 随游戏的进行，不断降低epsilon，减少随机动作
        if epsilon > FINAL_EPSILON and t > OBSERVE:  # 由观察期进入了探索期
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选择的动作，并获得下一状态及回报
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 将状态转移过程存储到D中，用于更新参数时采样
        D.append((s_t, a_t, r_t, s_t1, terminal))

        # 过了观察期，才会进行网络参数的更新
        if t > OBSERVE:
            # 从D中随机采样，用于参数更新
            minibatch = random.sample(D, BATCH)

            # 分别将当前状态、采取的动作、获得的回报、下一状态分组存放
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # 计算Q(s,a)的新值
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH, 80, 80, 4)
            y = model(state_batch).numpy()
            next_state_batch = np.array(s_j1_batch).astype('float32').reshape(BATCH, 80, 80, 4)
            readout_j1_batch = model(next_state_batch)

            for i in range(len(minibatch)):
                terminal = minibatch[i][4]
                # 如果游戏结束，就只有返回值
                if terminal:
                    y[i][np.argmax(a_batch[i])] = r_batch[i]
                else:  # 下面的更新过程详见DQN算法原理
                    y[i][np.argmax(a_batch[i])] = r_batch[i] + GAMMA * (np.max(readout_j1_batch[i]))

            # 使用梯度下降更新网络参数
            state_batch = np.array(s_j_batch).astype('float32').reshape(BATCH, 80, 80, 4)
            model.train_on_batch(state_batch, y)

        # 状态发生改变，用于下次循环
        s_t = s_t1
        t += 1

        # 每进行10000次迭代，保留一下网络参数
        if t % 10000 == 0:
            model.save_weights(model_filename)
            print('保存权重完成。')

        # 打印游戏信息
        if t < OBSERVE:
            state = 'observe'
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if t % 1 == 0:
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon,
                  "/ ACTION", action_index, "/ REWARD", r_t,
                  "/ Q_MAX %e" % np.max(readout_t))

        # K.clear_session()
        # gc.collect()


if __name__ == '__main__':
    trainDQN()
