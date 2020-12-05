# import sys
# import os
# os.chdir('/Users/binyu/Documents/git_exercise/computer_vision/projects/pytorchbook_best_practice')
# for file in os.listdir(os.getcwd()):
#     print(file)
# sys.path.append('/Users/binyu/Documents/git_exercise/computer_vision/projects/pytorchbook_best_practice')
import sys
import os
import torch
from torchnet import meter
from torch.autograd import Variable
rootPath = '/Users/binyu/Documents/git_exercise'
sys.path.append(rootPath)
from computer_vision.projects.pytorchbook_best_practice.config import DefaultConfig
from computer_vision.projects.pytorchbook_best_practice.utils.visualize import Visualizer
from computer_vision.projects.pytorchbook_best_practice import model
from computer_vision.projects.pytorchbook_best_practice.data_loader.dataset import DogCat
from torch.utils.data import DataLoader


def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """
    # 根据命令行参数更新配置
    opt = DefaultConfig()
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)

    # step1: 模型
    net = getattr(model, opt.model)(num_classes=2)
    if opt.load_model_path:
        # net.load(opt.load_model_path)
        pass
    if opt.use_gpu:
        net.cuda()

    # step2: 数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = torch.optim.Adam(params=net.parameters(),
                                 lr=lr,
                                 weight_decay=opt.weight_decay)

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    net.train()
    # 训练
    for epoch in range(opt.max_epoch):
        print('--epoch .....' + str(epoch))

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label, img_path) in enumerate(train_dataloader):

            # 训练模型参数
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = net(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(loss.data.item())
            confusion_matrix.add(score.data, target.data)

            print('--step .....' + str(ii))
            print('--loss ' + str(float(loss.data.item())))

            # if ii % opt.print_freq == opt.print_freq - 1:
            #     vis.plot('loss', loss_meter.value()[0])
            #
            #     # 如果需要的话，进入debug模式
            #     if os.path.exists(opt.debug_file):
            #         import ipdb;
            #         ipdb.set_trace()

        net.save()

        # 计算验证集上的指标及可视化
        # val_cm, val_accuracy = val(net, val_dataloader)
        # vis.plot('val_accuracy', val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
        #     .format(
        #     epoch=epoch,
        #     loss=loss_meter.value()[0],
        #     val_cm=str(val_cm.value()),
        #     train_cm=str(confusion_matrix.value()),
        #     lr=lr))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(net, dataloader, opt):
    """
    计算模型在验证集上的准确率等信息，用以辅助训练
    :param model:
    :param dataloader:
    :return:
    """
    # 把模型设为验证模式
    net.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(torch.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(torch.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    """
    测试（inference）
    :param kwargs:
    :return:
    """
    # 根据命令行参数更新配置
    opt = DefaultConfig()
    opt.parse(kwargs)
    # configure model
    net = getattr(model, opt.model)(num_classes=2).eval()
    if opt.load_model_path:
        net.load(opt.load_model_path)
    if opt.use_gpu:
        net.cuda()

    # data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, label, image_path) in enumerate(test_dataloader):
        # input = torch.autograd.Variable(data, volatile=True)
        # if opt.use_gpu:
        #     input = input.cuda()
        # score = net(input)
        # probability = torch.nn.functional.softmax(score)[:, 0].data.tolist()
        # # label = score.max(dim = 1)[1].data.tolist()
        with torch.no_grad():
            score = net(data)
            probability = torch.nn.functional.softmax(score, dim=1)[:, 0].data.tolist()

        batch_results = [(path_, probability_, label_) for path_, probability_, label_ in zip(image_path, probability, label)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'probability', 'label'])
        writer.writerows(results)


def help():
    """
    打印帮助的信息
    :return:
    """
    print('''
        usage : python file.py <function> [--args=value]
        <function> := train | test | help
        example: 
                python {0} train --env='env0701' --lr=0.01
                python {0} test --dataset='path/to/dataset/root/'
                python {0} help
        avaiable args:'''.format(__file__))

    from inspect import getsource
    # 根据命令行参数更新配置
    opt = DefaultConfig()
    # opt.parse(kwargs)
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    # import fire
    # fire.Fire()
    # train()
    # test()
    help()

"""
在该目录下执行
poetry run python main.py train --lr=0.001 --use_gpu=False
"""
