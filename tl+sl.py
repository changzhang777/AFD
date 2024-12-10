from __future__ import print_function
import os
import argparse
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.wideresnet import *
from autoattack import AutoAttack
from models.resnet import *
import logging
import copy
import math
from loss import mmd_rbf

os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8. / 255., type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type=float,
                    help='perturb step size')


parser.add_argument('--alpha', default=0.05, type=float,
                    help='the weight for correct feature channels')
parser.add_argument('--sigma', default=0.25, type=float,
                    help='the weight for features')
parser.add_argument('--gamma', default=25.0, type=float,
                    help='the weight for wrong feature channels')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.025, metavar='LR',# 0.01
                    help='learning rate')
parser.add_argument('--pretrained', default='ResNet18-CIFAR10/newest',# ResNet18-CIFAR10 'ResNet18-PGD-AT-CIFAR10'  WideResNet34-CIFAR10
                    help='directory of model')


parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ours/ResNet18-CIFAR10/',# WideResNet34-CIFAR10
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()
# from datetime import datetime
# + str(datetime.now())
model_dir = args.model_dir + 'pretrained=' + args.pretrained + '/' + time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())  + '-alpha-' + str(args.alpha) \
             + '-sigma-' + str(args.sigma) + '-gamma-' + str(args.gamma) + '-lr-' + str(args.lr) + '-wd-' + \
             str(args.weight_decay) + '-seed-' + str(args.seed) #+'-max-ce+6kl-min-opt1-opt2+0'


if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(model_dir, 'train.log'))
logger.info(args)


use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

class_number = 10


def adjust_learning_rate(args, optimizer, epoch):
    for param_group in optimizer.param_groups:
        if epoch <= 10:
            pass
        elif epoch == 11:
            param_group['lr'] = param_group['lr'] * 4 / 5
        elif (epoch + 1) % 2 == 0:
            param_group['lr'] /= 2







def L1(output_nat, output_rob):
    size = output_nat.shape
    if len(size) >= 3:# 4
        # torch.linalg.norm(output_nat - output_rob, ord=1, dim=list(range(2, len(size)))).mean() #矩阵范数
        # return torch.linalg.norm((output_nat - output_rob).view(size[0], size[1], -1), ord=1, dim=-1).mean()
        return torch.linalg.norm((output_nat - output_rob).view(size[0], size[1], -1).mean(dim=-1), ord=1, dim=-1).mean()
    else:
        return torch.linalg.norm(output_nat - output_rob, ord=1, dim=-1).mean()

def L2(output_nat, output_rob):
    size = output_nat.shape
    if len(size) >= 3:
        return torch.linalg.norm((output_nat - output_rob).view(size[0], size[1], -1).mean(dim=-1), ord=2, dim=-1).mean()
    else:
        return torch.linalg.norm(output_nat - output_rob, ord=2, dim=-1).mean()

def Linf(output_nat, output_rob):
    size = output_nat.shape
    if len(size) >= 3:
        return torch.linalg.norm((output_nat - output_rob).view(size[0], size[1], -1).mean(dim=-1), ord=torch.inf, dim=-1).mean()
    else:
        return torch.linalg.norm(output_nat - output_rob, ord=torch.inf, dim=-1).mean()

def Cosine(output_nat, output_rob):
    size = output_nat.shape
    if len(size) >= 3:
        return 1 - torch.cosine_similarity(output_nat.view(size[0], size[1], -1).mean(dim=-1), output_rob.view(size[0], size[1], -1).mean(dim=-1), dim=-1).mean()
    else:
        return 1 - torch.cosine_similarity(output_nat, output_rob, dim=-1).mean()







def criterion(model,
            teacher_model,
            # model_ema,
            x_natural,
            y,
            optimizer,
            optimizer1,
            optimizer2,
            step_size=0.007,
            epsilon=0.031,
            perturb_steps=10,
            epoch=0):

    model.eval()
    batch_size = len(x_natural)
    # last_model = model_ema.store_model(model)
    x_adv = x_natural.detach() + torch.randn(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            # # pgd
            loss_kl = F.cross_entropy(model(x_adv), y)

            # # trades
            # loss_kl += 6 * F.kl_div(F.log_softmax(model(x_adv), dim=1),
            #                        F.softmax(model(x_natural), dim=1), reduction='batchmean')

            # all
            # (logit_nat, logit_adv), (feat1, feat2) = model(x_adv, logit=True, prejection=True)
            # pred = logit_nat.data.max(1)[1].long()
            # logit = torch.scatter(logit_nat, 1, pred.unsqueeze(1), -100000)
            #
            # # 1.sample
            # # logit = F.softmax(logit, -1)
            # # pred_ = torch.cuda.LongTensor([torch.multinomial(logit[i], 1, replacement=False) for i in range(batch_size)])
            # # 2.max
            # pred_= logit.data.max(1)[1].long()
            #
            # # pred label
            # indicator = (pred != y)
            # pred_[indicator] = pred[indicator]

            # loss_kl = F.cross_entropy(logit_nat, y) + args.sigma * torch.cosine_similarity(feat1, feat2, -1).mean()
            # + args.alpha * F.cross_entropy(logit_adv, pred_)


        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    # last_model.eval()


    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    optimizer1.zero_grad()
    optimizer2.zero_grad()

    (logit_nat, logit_adv), (feat1, feat2) = model(x_adv, prejection=True)
    # (_, _), (feat_clean, _) = model(x_natural, prejection=True)
    pred = logit_nat.data.max(1)[1].long()
    logit = torch.scatter(logit_nat, 1, pred.unsqueeze(1), -100000)
    # logit = F.softmax(logit, -1)
    # pred_ = torch.cuda.LongTensor([torch.multinomial(logit[i], 1, replacement=False) for i in range(batch_size)])
    pred_ = logit.data.max(1)[1].long()
    # pred label
    indicator = (pred != y)
    pred_[indicator] = pred[indicator]

    with torch.no_grad():
        # _, feat1_last = last_model(x_adv, prejection=True)
        # _, feat1_last = last_model(x_natural, prejection=True)
        _, feat1_last = teacher_model(x_natural, prejection=True)
        # logits_nat = teacher_model(x_natural)

    # optimizer1.zero_grad()
    loss_neg = args.alpha * F.cross_entropy(logit_adv, pred_)
    loss_neg.backward(retain_graph=True)
    optimizer1.step()
    # loss_neg = torch.Tensor([-1]).cuda()

    # # optimizer2+optimizer
    # loss_cos = args.sigma * torch.cosine_similarity(feat1, feat2, -1).mean()
    # loss_sl = 1 - torch.cosine_similarity(feat1, feat1_last, -1).mean()
    # # theta = indicator.float().mean() ** 0.5#错误率
    # # loss_sl = 1 - theta * torch.cosine_similarity(feat1, feat1_last, -1).mean() - (1 - theta) * torch.cosine_similarity(feat1, feat_clean, -1).mean()
    # loss_rob = F.cross_entropy(logit_nat, y) + args.gamma * loss_sl
    # loss = loss_rob + loss_cos
    # loss.backward()
    # optimizer.step()

    # optimizer2.zero_grad()
    loss_cos = args.sigma * torch.cosine_similarity(feat1, feat2, -1).mean()
    loss_cos.backward(retain_graph=True)
    optimizer2.step()
    # loss_cos = torch.Tensor([-1]).cuda()

    # optimizer.zero_grad()
    loss_sl = 1 - torch.cosine_similarity(feat1, feat1_last, -1).mean() # cos
    # loss_sl = mmd_rbf(feat1, feat1_last) # mmd
    loss_rob = F.cross_entropy(logit_nat, y)
    loss_rob += + args.gamma * loss_sl
    loss_rob.backward()
    optimizer.step()

    return loss_rob.detach(), loss_neg.detach(), loss_cos.detach()







def train_align_loss(args, model_rob, model_nat, model_ema, train_loader, optimizer, optimizer1, optimizer2, epoch):
    model_rob.train()
    model_nat.eval()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        loss_rob, loss_neg, loss_cos = criterion(model=model_rob,
                         teacher_model=model_nat,
                         # model_ema=model_ema,
                         x_natural=data,
                         y=target,
                         optimizer=optimizer,
                         optimizer1=optimizer1,
                         optimizer2=optimizer2,
                         step_size=args.step_size,
                         epsilon=args.epsilon,
                         perturb_steps=args.num_steps,
                         epoch=epoch)


        model_ema.update_params(model_rob)
        model_ema.apply_shadow()
        # model_rob = copy.deepcopy(model_ema.model).cuda()


        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_Rob: {:.6f}\tLoss_Neg: {:.6f}\tLoss_Cos: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_rob.item(), loss_neg.item(), loss_cos.item()))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_Rob: {:.6f}\tLoss_Neg: {:.6f}\tLoss_Cos: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_rob.item(), loss_neg.item(), loss_cos.item()))






def learn_prior_knowledge(model, train_loader):
    model.eval()
    cost_embedding = torch.zeros((class_number, 512)).cuda()
    count_list = [0 for i in range(class_number)]
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            logit, feature = model(data, prejection=True)
        pred = (logit.data.max(1)[1] == target.data)

        # split by class
        # for cls in range(10):
        #     index = (target == cls) * pred
        #     count_list[cls] += index.sum().int()
        #     cost_embedding[cls] += feature[index].sum(dim=0)

        # split by logit
        for cls in range(class_number):
            count_list[cls] += pred.sum().int()
        logit = F.softmax(logit)
        cost_embedding += torch.einsum("ij,ik->jk", [logit[pred], feature[pred]])


    for cls in range(class_number):
        cost_embedding[cls] = feature[cls] / count_list[cls]
    cost_embedding = cost_embedding / torch.linalg.vector_norm(cost_embedding, ord=2, dim=-1, keepdim=True)
    return cost_embedding











class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]
        # self.first_model = copy.deepcopy(model)
        # self.second_model = copy.deepcopy(model)


    def update_params(self, model):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

    # def store_model(self, model):
    #     last_model = copy.deepcopy(self.second_model)
    #     self.second_model_model = copy.deepcopy(self.first_model)
    #     self.first_model = model
    #     return last_model



# def adjust_learning_rate_cosine(optimizer, epoch, args):
#     lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr
#
#
# for start_ep, tau, new_state_dict in zip(start_wa, tau_list, exp_avgs):
#     if epoch == start_ep:
#         for key, value in model.state_dict().items():
#             new_state_dict[key] = value
#     elif epoch > start_ep:
#         for key, value in model.state_dict().items():
#             new_state_dict[key] = (1 - tau) * value + tau * new_state_dict[key]
#     else:
#         pass


def eval_train(model, train_loader, logit=False):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            if logit:
                _, output = model(data, logit=True)
            else:
                output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    logger.info('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy






def _pgd_whitebox(model,
                  X,
                  y,
                  logit,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    with torch.no_grad():
        if logit:
            _, out = model(X, logit=True)
        else:
            out = model(X)
        err = (out.data.max(1)[1] != y.data).sum().item()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    with torch.no_grad():
        if logit:
            _, out = model(X_pgd, logit=True)
        else:
            out = model(X_pgd)
        err_pgd = (out.data.max(1)[1] != y.data).sum().item()
    return err, err_pgd



def eval_adv_test_whitebox(model, test_loader, logit=False):

    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        # PGD
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, logit)

        robust_err_total += err_robust
        natural_err_total += err_natural

    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1- robust_err_total / len(test_loader.dataset))
    logger.info('natural_acc: {}'.format(1 - natural_err_total / len(test_loader.dataset)))
    logger.info('robust_acc: {} '.format(1 - robust_err_total / len(test_loader.dataset)))




def model_ResNet_transfer(model):
    model1 = ResNet18(num_classes=class_number).cuda()
    model1.load_state_dict(model.state_dict(), strict=False)
    model1.linear.weight = torch.nn.parameter.Parameter(model1.linear.weight @ model.linear1.weight)
    return model1


def model_WideResNet_transfer(model):
    model1 = WideResNet(depth=34, num_classes=class_number).cuda()
    model1.load_state_dict(model.state_dict(), strict=False)
    model1.fc.weight = torch.nn.parameter.Parameter(model1.fc.weight @ model.fc1.weight)
    return model1




def eval_apgd(model, test_loader, transfer=False):

    if transfer:
        # model = model_WideResNet_transfer(model)
        model = model_ResNet_transfer(model)

    model.eval()
    robust_err_total = 0
    # adversary = AutoAttack(model, norm="Linf", eps=args.epsilon,
    #                        log_path=os.path.join(model_dir, 'test-apgd.log'))
    # adversary.attacks_to_run = ['apgd-ce']
    adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, version='standard',
                           log_path=os.path.join(model_dir, 'aa.log'))

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        # APGD
        data_adv = adversary.run_standard_evaluation(data, target, bs=args.test_batch_size)
        with torch.no_grad():
            logits = model(data_adv)
            err_robust = (logits.data.max(1)[1] != target.data).sum().item()

        robust_err_total += err_robust


    print('APGD-CE_acc: ', 1- robust_err_total / len(test_loader.dataset))
    logger.info('APGD-CE_acc: {} '.format(1 - robust_err_total / len(test_loader.dataset)))










def main():
    model_nat = ResNet18(class_number).cuda()
    model_nat.load_state_dict(torch.load('./checkpoint/baseline/ResNet18-Standard-CIFAR10/model-100.pth'), strict=False)

    model_rob = ResNet18_FS(class_number).cuda()
    model_rob.load_state_dict(torch.load('./checkpoint/baseline/ResNet18-Standard-CIFAR10/model-100.pth'), strict=False)



    rest_params1 = filter(lambda x: id(x) not in list(map(id, model_rob.linear2.parameters())), model_rob.parameters())
    optimizer = optim.SGD([{"params": rest_params1}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer1 = optim.SGD(model_rob.linear2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    rest_params2 = filter(lambda x: id(x) not in list(map(id, model_rob.linear2.parameters())) , model_rob.parameters())
    # + list(map(id, model_rob.linear1.parameters()))
    optimizer2 = optim.SGD([{"params": rest_params2}], lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay)


    # model_nat = WideResNet(depth=34, num_classes=class_number).cuda()
    # model_nat.load_state_dict(torch.load('./checkpoint/baseline/WideResNet34-Standard-CIFAR100/model-100.pth'))
    #
    # model_rob = WideResNet_FS(depth=34, num_classes=class_number).cuda()
    # model_rob.load_state_dict(torch.load('./checkpoint/baseline/WideResNet34-Standard-CIFAR100/model-100.pth'), strict=False)
    #
    # model_ema = EMA(model_rob)
    #
    # rest_params1 = filter(lambda x: id(x) not in list(map(id, model_rob.fc2.parameters())), model_rob.parameters())
    # optimizer = optim.SGD([{"params": rest_params1}], lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    # optimizer1 = optim.SGD(model_rob.fc2.parameters(), lr=args.lr, momentum=args.momentum,
    #                        weight_decay=args.weight_decay)
    # rest_params2 = filter(lambda x: id(x) not in list(map(id, model_rob.fc2.parameters())), model_rob.parameters())
    # # # + list(map(id, model_rob.linear1.parameters()))
    # # optimizer2 = optim.SGD([{"params": rest_params2}], lr=args.lr, momentum=args.momentum,
    # #                        weight_decay=args.weight_decay)

    #cost_embedding = learn_prior_knowledge(model_nat, train_loader)

    model_rob = torch.nn.DataParallel(model_rob).cuda()
    model_nat = torch.nn.DataParallel(model_nat).cuda()
    model_ema = EMA(model_rob)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        # adversarial training
        adjust_learning_rate(args, optimizer, epoch)
        adjust_learning_rate(args, optimizer1, epoch)
        adjust_learning_rate(args, optimizer2, epoch)
        train_align_loss(args, model_rob, model_nat, model_ema, train_loader, optimizer, optimizer1, optimizer2, epoch)
        print('using time:', time.time() - start_time)
        logger.info('using time: {}'.format(time.time() - start_time))
        eval_train(model_ema.model, test_loader)

        # # eval_train(model_rob, train_loader)
        # eval_train(model_ema.model, train_loader)
        # eval_train(model_ema.model, train_loader, True)
        #
        # # eval_adv_test_whitebox(model_rob, test_loader)
        # eval_adv_test_whitebox(model_ema.model, test_loader)# 8000
        # eval_adv_test_whitebox(model_ema.model, test_loader, True)

        torch.save(model_rob.state_dict(),
                   os.path.join(model_dir, 'model-%d.pth' % epoch))

        torch.save(model_ema.model.state_dict(),
                   os.path.join(model_dir, 'model-ema-%d.pth' % epoch))

        print('================================================================')
        logger.info('================================================================')

    # eval_train(model_rob, train_loader)
    eval_train(model_ema.model, train_loader)
    # eval_train(model_ema.model, train_loader, True)

    # eval_adv_test_whitebox(model_rob, test_loader)
    eval_adv_test_whitebox(model_ema.model, test_loader)  # 8000
    # eval_adv_test_whitebox(model_ema.model, test_loader, True)

    torch.save(model_rob.state_dict(),
               os.path.join(model_dir, 'model-20.pth'))
    torch.save(model_ema.model.state_dict(),
               os.path.join(model_dir, 'model-ema-20.pth'))

    # eval_apgd(model_rob, test_loader)
    eval_apgd(model_ema.model, test_loader)

    # model = WideResNet_FS(depth=34, num_classes=class_number).cuda()
    # model = ResNet18_FS(class_number).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    # # model = ResNet18(class_number).cuda()
    # model.load_state_dict(torch.load('/feature_purification/checkpoint/ours/ResNet18-CIFAR100/pretrained=ResNet18-CIFAR100/newest/2024-03-22 21:03:26.667479-alpha-0.1-sigma-2.0-gamma-50.0-lr-0.025-wd-0.0005-seed-1' + '/model-ema-20.pth'))
    # # eval_adv_test_whitebox(model, test_loader)
    # eval_apgd(model, test_loader)






if __name__ == '__main__':
    main()
