from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from autoattack import AutoAttack
#from models import resnet_transition
# from models import wideresnet
# from models import resnet
from models.RiFT_resnet import *
from models.RiFT_wideresnet import *
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import logging
import cv2
import torch.fft as fft
import torchvision
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--epsilon', type=float, default=8.0/255.0, help='perturb scale')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for training (default: 500)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model-dir',
                    default='/feature_purification/checkpoint/ours/ResNet18-CIFAR100/pretrained=ResNet18-CIFAR100/newest/2024-03-25 00-10-09-alpha-0.1-sigma-1.0-gamma-50.0-lr-0.025-wd-0.0005-seed-1max-NR/',
                    help='directory of model for saving checkpoint')
# ./checkpoint/ours/ResNet18-CIFAR10/pretrained=ResNet18-CIFAR10  ./checkpoint/baseline/Mart_128_resnet18_cifar100/
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')


args = parser.parse_args()

logger = logging.getLogger(__name__)
logfile = os.path.join(args.model_dir, 'test_aa.log')
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(args.model_dir, 'test_aa.log'))
logger.info(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)




def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    correct = 0.

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy





def imsave(tensor, name):
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    image = tensor.clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    #print(image.shape, image.type)
    image = toPIL(image.float())

    image.save('./pictures/{}.jpg'.format(name))
    np.savetxt('./numbers/{}.csv'.format(name), tensor.view(tensor.size(0), -1).cpu().numpy())
    for i in range(3):
        np.savetxt('./numbers/{}_{}.csv'.format(name, i+1), tensor.cpu().numpy()[i])



def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)







def pgd_attack(model, x_natural, y, step_size=0.003,
                epsilon=8.0/255.0, perturb_steps=10, distance='l_inf', i = 0):
    #imsave(x_natural[0], "x_out/%d" % i, "clean")
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                # _, logits = model(x_adv, logit=True)
                loss_kl = F.cross_entropy(logits, y)

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        batch_size = len(x_natural)
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                logits = model(x_adv)
                # logits = F.softmax(logits, dim=1)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    #imsave(x_adv[0], "x_out/%d" % i, "pgd")
    return x_adv



def test_adv(model, data_adv, label):
    # logits = model(data_adv)
    _, logits = model(data_adv, logit=True)
    correct = (logits.data.max(1)[1] == label.data).sum().item()
    acc_adv_40 = 100 * correct / label.size(0)
    return acc_adv_40



def main():
    # init model, ResNet18() can be also used here for training
    setup_seed(args.seed)

    # setup data loader
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),

    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=2, pin_memory=True)
    class_number = 10
    # from tiny_imagenet import TinyImageNet
    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    # testset = TinyImageNet('../data/tiny-imagenet-200', train=False, transform=transform_test)
    # class_number = 200
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    for path in ['model-ema-40.pth']:#, 'model_120.pth' model-res-epoch120.pt 'model-15.pth', 'model-ema-15.pth'
        model_path = os.path.join(args.model_dir, path)
        print(model_path)
        logger.info(model_path)
        # model = wideresnet.WideResNet_FS(depth=34, num_classes=class_number)
        # model = resnet.ResNet18(class_number).cuda()
        # model = WideResNet_FS(image_size=64, depth=34, num_classes=class_number)
        model = ResNet50_FS(32, class_number).cuda()
        # model = ResNet18_FS(64, class_number).cuda()
        # model = resnet.ResNet18_FS(100).cuda()
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(model_path))
        # model = torch.nn.DataParallel(model).cuda()
        # if 'best' in path:
        #     model = torch.nn.DataParallel(model).cuda()
        #     model.module.load_state_dict(torch.load(model_path))
        # else:
        #     model.load_state_dict(torch.load(model_path))
        #     model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        model.eval()


        Correct_pgd = 0.
        Correct_aa = 0.
        Correct = 0.
        for batch, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)

            # data_adv = pgd_attack(model=model, x_natural=data, y=label, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps)
            # acc_adv_40 = test_adv(model, data_adv, label)
            # print('test_pgd: batch: %d / %d, PGD ACC: %f' % (batch, len(test_loader), acc_adv_40))
            # logger.info('test_pgd: batch: %d / %d, PGD ACC: %f' % (batch, len(test_loader), acc_adv_40))
            # Correct_pgd += acc_adv_40 / len(test_loader)

            acc = test_adv(model, data, label)
            Correct += acc / len(test_loader)

        #     adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, version='standard', log_path=os.path.join(args.model_dir, 'autoattack.log'))
        #     # adversary = AutoAttack(model, norm="Linf", eps=args.epsilon,
        #     #                        log_path=os.path.join(args.model_dir, 'test-apgd.log'))
        #     # adversary.attacks_to_run = ['apgd-ce']
        #
        #     data_adv = adversary.run_standard_evaluation(data, label, bs=len(data))
        #     acc_aa = test_adv(model, data_adv, label)
        #     #acc_aa = 0.
        #     print('test_aa: batch: %d / %d, ACC: %f' % (batch, len(test_loader), acc_aa))
        #     logger.info('test_aa: batch: %d / %d, ACC: %f' % (batch, len(test_loader), acc_aa))
        #     Correct_aa += acc_aa / len(test_loader)
        # print('Test: AA ACC: %f' % (Correct_aa))
        # logger.info('Test: AA ACC: %f' % (Correct_aa))
        print('Test:  ACC: %f' % (Correct))
        logger.info('Test:  ACC: %f' % (Correct))
        # print('Test: PGD-20 ACC: %f' % (Correct_pgd))
        # logger.info('Test: PGD-20 ACC: %f' % (Correct_pgd))
        logger.info('================================================================')
        print('================================================================')

if __name__ == '__main__':
    main()

""" """
