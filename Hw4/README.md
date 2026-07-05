<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> <br>
  <a href="https://trendshift.io/repositories/8133" target="_blank"></a>
</div>

## 实验简介
本仓库用于本科生《语音信息处理》课程的实验教学，实验内容围绕开源多语言文本转语音系统 **MeloTTS** 展开，旨在帮助学生从原理分析与工程实践两个层面理解现代 TTS（Text-to-Speech）系统的基本流程。

本次实验改编自 [MeloTTS](https://github.com/myshell-ai/MeloTTS)，MeloTTS 是一个高质量、多语言、多口音的文本转语音系统，支持英语、西班牙语、法语、中文、日语、韩语等语言，并具备较快的推理速度和较好的可用性。

## 下载与用例
- [安装与使用示例](install.md)
- [在自定义数据集上训练](docs/training.md)


## 实验内容概览
本仓库包含四次实验，前 3 次为必做实验，第 4 次为拓展实验（作为加分项）。

| 实验编号 | 实验代码 | 实验主题 | 核心内容 |
| --- | --- | --- | --- |
| Homework 1 | [tts_homework_1.ipynb](tts_homework_1.ipynb) | MeloTTS 安装与 API 使用 | 环境配置、模型加载、基础语音合成、参数理解 |
| Homework 2 | [tts_homework_2.ipynb](tts_homework_2.ipynb) | 多音字语音合成 | 文本前端、TN/G2P、多音字定制控制 |
| Homework 3 | [tts_homework_3.ipynb](tts_homework_3.ipynb) | 韵律控制实验 | VITS 结构、MAS/SDP、时长调整与韵律变化 |
| Homework 4 | [tts_homework_4.ipynb](tts_homework_4.ipynb) | 训练与推理优化 | 自定义音色训练 或 ONNX 推理优化 |

## 实验提交要求
完成 Jupyter Notebook 内的实验内容，将答案和相关代码填写在 [姓名-学号-TTS实验报告模板.docx](姓名-学号-TTS实验报告模板.docx) 中，并和生成的音频文件一起打包提交。


## 说明
本次实验尽可能使用国内网络环境进行资源下载和模型训练，如果遇到网络问题导致无法下载资源或依赖环境，请参考以下解决方案：
1. 使用国内镜像源（如清华大学、阿里云等）替换默认的 PyPI 镜像源，安装依赖时加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 参数。
2. 使用 [HF-Mirror](https://hf-mirror.com/) 等国内镜像站点下载 Hugging Face 上的模型和数据集。
3. 使用 [ModelScope](https://www.modelscope.cn/) 等国内平台提供的预训练模型和数据集。
4. 搜索 MeloTTS 的安装教程和解决方案，参考国内社区（如 CSDN、知乎等）上的相关讨论和经验分享。
5. 如果仍然无法解决网络问题，请尝试使用科学上网工具访问相关资源。


## 致谢
本实验仓库基于以下开源项目与研究工作构建，在此表示感谢：

[MeloTTS](https://github.com/myshell-ai/MeloTTS),
[TTS](https://github.com/coqui-ai/TTS), 
[VITS](https://github.com/jaywalnut310/vits), 
[VITS2](https://github.com/daniilrobnikov/vits2) and 
[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)

同时感谢原项目作者及相关社区开发者提供的代码、模型与文档支持。