import torch.nn as nn
import torch
import time


class BasicModule(nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        :param path:
        :return:
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        :param name:
        :return:
        """
        if name is None:
            prefix = '/Users/binyu/Documents/git_exercise/computer_vision/projects/pytorchbook_best_practice/ckpt/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
