# phm-diagnosis

#### 介绍

故障诊断

使用上海九章云极公司的开源工具包-ylearn，实现故障诊断任务
ylearn主要依托，华为诺亚的gCastle推理包，torch，networkx完成以上任务，
在传参数时，尽可能使用`**kwargs`，如有不便于理解之处，可以直接阅读ylearn源码及注释。



包括:
- [x] 基于数据,生成故障树
- [x]
- [x]
- [x]
- [x]



#### 软件架构
软件架构说明
```
.
├── build
│   ├── bdist.linux-x86_64
│   └── lib
│       └── phm_feature
│           └── __init__.py
├── dist
│   ├── phm_feature-0.0.2-py3-none-any.whl
│   └── phm_feature-0.0.2.tar.gz
├── LICENSE
├── phm_feature
│   ├── __init__.py
├── phm_feature.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
├── README.md
├── setup.py
└── test.py
```

#### 安装教程

> 使用pip进行安装
```
pip install phm-feature
```

#### 使用说明

1.  获得振动信号的特征值
```
import phm_feature
from phm_feature import *
enable_parallel(processnum=None) # 开启多线程模式
disable_parallel() # 开启单线程模式
feature_t(data) # 获取时间域特征
# feature_f # 获取频率域特征
fft(data, 50) # 快速离散傅里叶变换
power(data, 50) # 功率谱
ifft(data, 50) # 快速离散逆傅里叶变换
cepstrum(data, 50) # 倒谱
envelope(data) # 包络谱
window(data, 'hamming') # 加窗-汉明窗
divide(data, 50, 25) # 分帧
```

#### 参与贡献

1. 2022-07-04 v0.0.1
```
pypi上传初版本
pypi上传phm-feature初版本
```

2. 2022-07-04 v0.0.2
```
新增多线程模式
```

#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)



