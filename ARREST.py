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
from datetime import datetime
from models.wideresnet import *
from autoattack import AutoAttack
from models.resnet import *
import logging
import copy
import math




os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8. / 255., type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2. / 255., type=float,
                    help='perturb step size')

parser.add_argument('--NR', action='store_true', default=False)
parser.add_argument('--RGKD', action='store_true', default=False)
parser.add_argument('--beta', default=50.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--theta', default=30.0, type=float,
                    help='theta')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,#5e-4
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.025, metavar='LR',# 0.01
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ARREST/ResNet18-CIFAR100/',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()


model_dir = args.model_dir + str(datetime.now()) + '-beta-' + str(args.beta) + '-theta-' + str(args.theta) + '-lr-' + \
            str(args.lr) + '-epochs-' + str(args.epochs) + '-NR-' + str(args.NR) + '-RGKD-' + str(args.RGKD) + \
            '-seed-' + str(args.seed)


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
trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

class_number=100



def adjust_learning_rate(args, optimizer, epoch):
    for param_group in optimizer.param_groups:
        if epoch <= 10:
            pass
        elif epoch == 11:
            param_group['lr'] = 0.02
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
    if len(size) >= 4:
        return 1 - torch.cosine_similarity(output_nat.view(size[0], size[1], -1).mean(dim=-1), output_rob.view(size[0], size[1], -1).mean(dim=-1), dim=-1).mean()
        # return 1 - torch.cosine_similarity(output_nat.view(size[0], -1), output_rob.view(size[0], -1), dim=-1).mean()
    else:
        return 1 - torch.cosine_similarity(output_nat, output_rob, dim=-1).mean()





def ARREST(model,
           model_nat,
           x_natural,
           y,
          optimizer,
          step_size,
          epsilon,
          perturb_steps,
          epoch,
          beta=50.0,
          theta=30):
    model.eval()

    if args.NR:
        with torch.no_grad():
            _, output_nat = model_nat(x_natural, prejection=True)
            _, output_rob = model(x_natural, prejection=True)
            distance = Cosine(output_nat, output_rob) # mean
    else:
        distance = 0.

    x_adv = x_natural.detach() + torch.rand(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()

    if (distance >= 1 - math.cos(math.radians(theta))) and (epoch <= 10):# epoch<=10且距离过大的时候不攻击
        pass
    else:
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    with torch.no_grad():
        _, output_nat = model_nat(x_natural, prejection=True)
    with torch.enable_grad():
        logit_rob, output_rob = model(x_adv, prejection=True)

    loss_ce = F.cross_entropy(logit_rob, y)
    if args.RGKD:
        loss_rgkd = Cosine(output_nat, output_rob)
    else:
        loss_rgkd = 0.

    loss = loss_ce + beta * loss_rgkd
    return loss








def train_align_loss(args, model_nat, model_rob, train_loader, optimizer, epoch):
    model_rob.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        loss = ARREST(model=model_rob,
                         model_nat=model_nat,
                         x_natural=data,
                         y=target,
                         optimizer=optimizer,
                         step_size=args.step_size,
                         epsilon=args.epsilon,
                         perturb_steps=args.num_steps,
                         epoch=epoch,
                         beta=args.beta,
                         theta=args.theta)

        loss.backward()
        optimizer.step()



        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))










def eval_train(model, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
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
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):
    with torch.no_grad():
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
        err_pgd = (model(X_pgd).data.max(1)[1] != y.data).sum().item()
    return err, err_pgd



def eval_adv_test_whitebox(model, test_loader):

    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        # PGD
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)

        robust_err_total += err_robust
        natural_err_total += err_natural

    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1- robust_err_total / len(test_loader.dataset))
    logger.info('natural_acc: {}'.format(1 - natural_err_total / len(test_loader.dataset)))
    logger.info('robust_acc: {} '.format(1 - robust_err_total / len(test_loader.dataset)))




def eval_apgd(model, test_loader):
    model.eval()
    robust_err_total = 0
    # adversary = AutoAttack(model, norm="Linf", eps=args.epsilon,
    #                        log_path=os.path.join(model_dir, 'test-apgd.log'))
    # adversary.attacks_to_run = ['apgd-ce']
    adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, version='standard',
                           log_path=os.path.join(model_dir, 'autoattack.log'))

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        # APGD
        data_adv = adversary.run_standard_evaluation(data, target, bs=data.shape[0])
        with torch.no_grad():
            logits = model(data_adv)
            err_robust = (logits.data.max(1)[1] != target.data).sum().item()

        robust_err_total += err_robust


    print('AA-CE_acc: ', 1- robust_err_total / len(test_loader.dataset))
    logger.info('AA-CE_acc: {} '.format(1 - robust_err_total / len(test_loader.dataset)))










def main():
    # init model, ResNet18() can be also used here for training
    # model_nat = WideResNet(depth=34, num_classes=class_number).cuda()
    # model_nat.load_state_dict(torch.load('./checkpoint/baseline/WideResNet34-Standard-CIFAR100/model-100.pth'))
    model_nat = ResNet18(class_number).cuda()
    model_nat.load_state_dict(torch.load('./checkpoint/baseline/ResNet18-Standard-CIFAR100/model-100.pth'), strict=False)

    model_rob = copy.deepcopy(model_nat)
    optimizer = optim.SGD(model_rob.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)#

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        # adversarial training
        adjust_learning_rate(args, optimizer, epoch)
        train_align_loss(args, model_nat, model_rob, train_loader, optimizer, epoch)
        print('using time:', time.time() - start_time)
        logger.info('using time: {}'.format(time.time() - start_time))

        # evaluation on natural examples
        eval_train(model_rob, train_loader)

        # eval_test(model, test_loader)
        eval_adv_test_whitebox(model_rob, test_loader)

        # save checkpoint
        torch.save(model_rob.state_dict(),
                   os.path.join(model_dir, 'model-{}.pth'.format(epoch)))


        if epoch == args.epochs :
            eval_apgd(model_rob, test_loader)


        print('================================================================')
        logger.info('================================================================')




if __name__ == '__main__':
    main()
