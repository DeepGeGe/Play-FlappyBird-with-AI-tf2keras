# -*- coding: utf-8 -*-
# @Time    : 2021/1/16 10:11
# @Author  : He Ruizhi
# @File    : playGame.py
# @Software: PyCharm

import os
import sys

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir + '/game')

from wrapped_flappy_bird import GameState
import numpy as np
import cv2
import pygame
from flappy_bird_train import creatDQN


# 首先加载模型
model = creatDQN()
model.load_weights(current_dir + '/model/DQNModel.h5')


class PlayGame(object):
    def __init__(self, level=30):
        self.game_state = GameState()  # 游戏模拟器
        self.level = level  # 游戏的难度等级
        self.s_start = None  # 游戏初始界面状态
        self.game_explain_else = None  # 游戏绘制文字的对象
        self.ai_height = None  # 记录要更新的是否启用AI辅助文字的位置
        self.ai_font = None  # 记录是否启用AI辅助的Rect对象
        self.level_height = None  # 记录要更新的难度等级文字的位置
        self.level_font = None  # 记录难度等级的Rect对象
        self.game_explain_start = None  # 记录游戏绘制文字对象

        # 传入一个表示不跳跃的动作，获得游戏的初始界面
        do_nothing = np.zeros(2)
        do_nothing[0] = 1  # 意思是采用第一个动作，即不跳跃
        x_t, r_0, terminal = self.game_state.frame_step(do_nothing)
        # 使用opencv中相关函数将图片大小调整为(80,80)的灰度图
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        # 将图片处理成二值图，在这里，像素点值大于1就被赋值为255
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        # 将二值图x_t按照像素点值排列顺序拉开，并复制4次，组合成4通道输入数据
        self.s_start = np.stack((x_t, x_t, x_t, x_t), axis=2)

        # 在游戏界面绘制游戏说明信息
        screen = self.game_state.screen
        height = screen.get_height() / 6  # 开始绘制文字的起点
        self.game_explain_start = pygame.font.Font(current_dir + '/assets/fonts/msyh.ttc', 28)
        self.game_explain_else = pygame.font.Font(current_dir + '/assets/fonts/msyh.ttc', 20)  # 保存写游戏说明的字体
        game_explain_other = pygame.font.Font(current_dir + '/assets/fonts/msyh.ttc', 16)
        # 绘制游戏说明文字
        game_explain = self.game_explain_start.render('游戏说明：', True, (255, 255, 255))
        screen.blit(game_explain,
                    ((screen.get_width() - game_explain.get_width()) / 2, height))
        height += game_explain.get_height()  # 接下来绘制的位置
        # 绘制【开始/暂停】文字
        start_or_suspend = self.game_explain_else.render('开始：空格键 暂停：S键', True, (255, 255, 255))
        screen.blit(start_or_suspend,
                    ((screen.get_width() - start_or_suspend.get_width()) / 2, height))
        height += start_or_suspend.get_height()
        # 绘制操作说明
        operate = self.game_explain_else.render('跳跃：空格/↑键', True, (255, 255, 255))
        screen.blit(operate,
                    ((screen.get_width() - operate.get_width()) / 2, height))
        height += operate.get_height()
        # 绘制【AI辅助】文字
        self.ai_height = height  # 记录要更改的AI辅助文字的位置
        self.ai_font = self.game_explain_else.render('AI辅助(A键)：未加载', True, (255, 255, 255))
        # 默认不加载AI
        screen.blit(self.ai_font,
                    ((screen.get_width() - self.ai_font.get_width()) / 2, height))
        height += self.ai_font.get_height()
        # 绘制【难度等级】文字
        self.level_height = height
        self.level_font = self.game_explain_else.render('难度等级(L键)：{}'.format(level), True, (255, 255, 255))
        screen.blit(self.level_font,
                    ((screen.get_width() - self.level_font.get_width()) / 2, height))
        height += self.level_font.get_height() * 3
        # 绘制其它说明
        other_1 = game_explain_other.render('(请忽略这个简陋的启动界面)', True, (255, 255, 255))
        other_2 = game_explain_other.render('(作者：何睿智(DeepGeGe))', True, (255, 255, 255))
        other_3 = game_explain_other.render('(源码见GitHub)', True, (255, 255, 255))
        other_4 = game_explain_other.render('(别忘了点赞哟^v^)', True, (255, 255, 255))
        screen.blit(other_1,
                    ((screen.get_width() - other_1.get_width()) / 2, height))
        height += other_1.get_height()
        screen.blit(other_2,
                    ((screen.get_width() - other_2.get_width()) / 2, height))
        height += other_2.get_height()
        screen.blit(other_3,
                    ((screen.get_width() - other_3.get_width()) / 2, height))
        height += other_3.get_height()
        screen.blit(other_4,
                    ((screen.get_width() - other_4.get_width()) / 2, height))
        pygame.display.update()

    def play_game(self):
        """ 游戏框架：
        将游戏分为四个状态：ready，play，suspend，over
        游戏主循环对应四个状态
        对每种状态进行按键判断及状态切换
        """
        ai_use = False  # 是否使用AI辅助
        state = 'ready'  # 游戏的状态，分为ready，play，suspend，over
        level = self.level
        s_game = self.s_start
        end_font = True  # 用来控制结束界面只绘制文字一次

        # 游戏循环
        while True:
            if state == 'ready':  # 准备状态，在准备界面进行调速，启用AI辅助设置
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        # AI辅助和难度等级文字需要进行更改
                        if event.key == pygame.K_a:  # 是否启用AI辅助
                            # 首先覆盖掉有字的区域
                            self.ai_font.fill('black')
                            self.game_state.screen.blit(self.ai_font,
                                ((self.game_state.screen.get_width() - self.ai_font.get_width()) / 2,
                                 self.ai_height))
                            if ai_use:  # 表明开启了AI辅助
                                self.ai_font = self.game_explain_else.render('AI辅助(A键)：未加载', True, (255, 255, 255))
                                self.game_state.screen.blit(self.ai_font,
                                    ((self.game_state.screen.get_width() - self.ai_font.get_width()) / 2,
                                     self.ai_height))
                                ai_use = False
                            else:  # 表明未开启AI辅助
                                self.ai_font = self.game_explain_else.render('AI辅助(A键)：已加载', True, (255, 255, 255))
                                self.game_state.screen.blit(self.ai_font,
                                    ((self.game_state.screen.get_width() - self.ai_font.get_width()) / 2,
                                     self.ai_height))
                                ai_use = True
                        elif event.key == pygame.K_l:  # 调整等级
                            # 首先覆盖掉有字的区域
                            self.level_font.fill('black')
                            self.game_state.screen.blit(self.level_font,
                                ((self.game_state.screen.get_width() - self.level_font.get_width()) / 2,
                                 self.level_height))
                            level += 10
                            if level > 120:
                                level = 30
                            self.level_font = self.game_explain_else.render('难度等级(L键)：{}'.format(level),
                                                                            True, (255, 255, 255))
                            self.game_state.screen.blit(self.level_font,
                                ((self.game_state.screen.get_width() - self.level_font.get_width()) / 2,
                                 self.level_height))
                        elif event.key == pygame.K_SPACE:
                            state = 'play'

            elif state == 'play':  # 玩游戏状态，需判断是否暂停，调速，以及根据是否启用了AI辅助进行相应的处理
                # 按键判断需和游戏状态更新在同一循环下
                if ai_use:
                    action = np.zeros(2)
                    action[np.argmax(model(s_game.astype('float32').reshape(1, 80, 80, 4)))] = 1
                else:
                    action = np.zeros(2)
                    action[0] = 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_SPACE, pygame.K_UP]:
                            action = np.zeros(2)
                            action[1] = 1
                        elif event.key == pygame.K_s:
                            state = 'suspend'
                        elif event.key == pygame.K_a:
                            if ai_use:
                                ai_use = False
                            else:
                                ai_use = True
                        elif event.key == pygame.K_l:
                            level += 10
                            if level > 120:
                                level = 30

                # 运行游戏
                x_t1_colored, r_t, terminal = self.game_state.frame_step(action, level=level)
                # 更新s_game
                x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
                ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
                x_t1 = np.reshape(x_t1, (80, 80, 1))
                s_t1 = np.append(x_t1, s_game[:, :, :3], axis=2)
                s_game = s_t1

                if terminal:  # 结束
                    state = 'end'

            elif state == 'suspend':  # 暂停状态，需进行AI辅助开关处理，调速处理，暂停退出处理
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_s, pygame.K_SPACE]:
                            state = 'play'
                        elif event.key == pygame.K_a:
                            if ai_use:
                                ai_use = False
                            else:
                                ai_use = True
                        elif event.key == pygame.K_l:
                            level += 10
                            if level > 120:
                                level = 30

            else:  # state == 'end'  # 结束状态，绘制游戏结束，进行是否重启游戏处理
                # 绘制游戏结束
                if end_font:
                    over = self.game_explain_start.render('游戏结束！', True, (255, 255, 255))
                    over_2 = self.game_explain_start.render('按空格键重开游戏！', True, (255, 255, 255))
                    self.game_state.screen.blit(over,
                        ((self.game_state.screen.get_width() - over.get_width()) / 2,
                         (self.game_state.screen.get_height() - over.get_height()) / 5))
                    self.game_state.screen.blit(over_2,
                        ((self.game_state.screen.get_width() - over_2.get_width()) / 2,
                         (self.game_state.screen.get_height() - over_2.get_height()) / 5 + over.get_height()))
                    end_font = False

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            # game_state_new = wfb.GameState()
                            # self.__init__(game_state_new)
                            self.__init__()
                            state = 'ready'
                            ai_use = False
                            level = 30
                            end_font = True
                            s_game = self.s_start
            pygame.display.update()


if __name__ == '__main__':
    PlayGame().play_game()
