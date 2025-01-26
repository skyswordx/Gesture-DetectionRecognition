# Gesture-DetectionRecognition

## 开发记录 log

确认两级流水线的设计方案，第一级为手部检测，第二级为关键点检测和手势分类

第一级（手部检测）：
- 目标：快速找到图像中手的位置，并框出边界框（Bounding Box）
- 输出：手的坐标范围（例如左上角坐标(x1,y1)和右下角坐标(x2,y2)）
- 方案：[google-media-pipe](https://github.com/google-ai-edge/mediapipe)

第二级（关键点+分类）：
- 目标：在检测到的区域中定位手部关键点（如21个关节位置），然后根据关键点判断手势类型
- 输出：关键点坐标 + 手势类别（如“握拳”“比耶”等）
- 方案： SVM

## 相关文档


- [Google-MediaPipe-zh-python-setup](https://ai.google.dev/edge/mediapipe/solutions/setup_python?hl=zh-cn)

- [Google-MediaPipe-zh-user-guide](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=zh-cn#get_started)

    - [配置环境需要 VS 工具集](https://github.com/google-ai-edge/mediapipe/issues/1905)
        - 首先下载 VS 工具集，然后安装重启
        - 之后回到虚拟环境安装 `pip install msvc-runtime`
        - [VS 工具集下载链接](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)


使用方式总结
```shell
conda create -n gestureDR python=3.10 

conda activate gestureDR

pip install mediapipe opencv-python

pip install msvc-runtime
```