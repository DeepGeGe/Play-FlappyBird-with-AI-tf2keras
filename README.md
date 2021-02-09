# Play-FlappyBird-with-AI
对Flappy Bird游戏进行了二次开发，添加了AI辅助以及调整难度等级的设置。该项目同时包含DQN训练部分代码。

参考了https://github.com/yenchenlin/DeepLearningFlappyBird 实现。使用了tf2.0keras接口对代码进行了重写，同时参考了https://github.com/zhouchunyuan/DQN_FlappyBird_tf2.0_keras ，对代码进行了优化，使得训练约20W次就能取得不错的效果。
而且优化了训练速度。

此外，我在尝试将整个项目使用pyinstaller打包成一个.exe文件，但是总会有这样那样的问题，用了一个周末的时间也没能搞定。
如果有成功将整个项目打包成一个.exe的小伙伴，欢迎留言。

(2021年1月18日记录：playGame.py文件中的在屏幕上绘制文字应该用函数封装起来，后面直接调用。不会去修改，但是记录一下。)

(2021年2月9日：成功将该游戏用pyinstaller打包成一个.exe文件，可在任意windows操作系统电脑上运行。
下载地址：https://pan.baidu.com/s/1BypiVi3Bu09lXMY-c9n7Sw 提取码：5hux)

![image](https://github.com/DeepGeGe/Play-FlappyBird-with-AI-tf2keras/blob/main/1.png)
![image](https://github.com/DeepGeGe/Play-FlappyBird-with-AI-tf2keras/blob/main/2.png)
![image](https://github.com/DeepGeGe/Play-FlappyBird-with-AI-tf2keras/blob/main/3.png)
