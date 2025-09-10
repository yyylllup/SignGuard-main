import numpy as np # linear algebra
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from data_loader import get_dataset
from running import test_classification, benignWorker, byzantineWorker
from models import CNN, CifarCNN, ResNet18
from aggregators import aggregator
from attacks import attack
from options import args_parser
import tools
import time
import copy

from attacks.poisonedfl_adaptive import PoisonedFLAdaptive
from attacks.poisonedfl_strict import PoisonedFLStrict


# make sure that there exists CUDA，and show CUDA：
# print(device)
#
# attacks : non, random, noise, signflip, label_flip, byzMean.
#           lie, min_max, min_sum, *** adaptive (know defense) ***
#
# defense : Mean, TrMean, Median, GeoMed, Multi-Krum, Bulyan, DnC, SignGuard.

# set training hype-parameters
# arguments dict

# args = {
#     "epochs": 160,
#     "num_users": 50,
#     "num_byzs": 10,
#     "frac": 1.0,
#     "local_iter": 1,
#     "local_batch_size": 50,
#     "optimizer": 'sgd',
#     "agg_rule": 'Mean',
#     "attack": 'non',
#     "lr": 0.2,
#     "dataset": 'cifar',
#     "iid": True,
#     "unbalance": False,
#     "device": device
# }
if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.dataset)
    # load dataset and user groups
    train_loader, test_loader = get_dataset(args)
    # construct model
    if args.dataset == 'cifar':
        global_model = CifarCNN().to(device)
    elif args.dataset == 'fmnist':
        global_model = CNN().to(device)
    else:
        global_model = CNN().to(device)
            
    global_model = global_model.cuda()

    # optimizer
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # number of iterations per epoch
    # iteration = len(train_loader[0].dataset) // (args.local_bs * args.local_iter)
    iteration = max(1, len(train_loader[0].dataset) // max(1, (args.local_bs * args.local_iter)))

    train_loss, train_acc = [], []
    test_acc = []
    byz_rate = []
    benign_rate = []

# =============================改动点1=================================
    # attack method
    Attack = None
    if args.attack == "poisonedfl_adaptive":
        # 初始化 PoisonedFL 攻击器
        Attack = PoisonedFLAdaptive(num_malicious=args.num_byzs,
                                    initial_model=global_model)
    elif args.attack == "poisonedfl_strict":
        Attack = PoisonedFLStrict(num_malicious=args.num_byzs, initial_model=global_model)
    else:
        Attack = attack(args.attack)

# ====================================================================

    # Gradient Aggregation Rule
    GAR = aggregator(args.agg_rule)()


    def train_parallel(args, model, train_loader, optimizer, epoch, scheduler):
        print(f'\n---- Global Training Epoch : {epoch + 1} ----')
        num_users = args.num_users
        num_byzs = args.num_byzs
        device = args.device
        iter_loss = []
        data_loader = []

        for idx in range(num_users):
            data_loader.append(iter(train_loader[idx]))

        for it in range(iteration):
            # 随机抽取m个客户端参与迭代
            m = max(int(args.frac * num_users), 1)
            idx_users = np.random.choice(range(num_users), m, replace=False)
            idx_users = sorted(idx_users)

            local_losses = []
            benign_grads = []
            byz_grads = []

# =============================改动点2================================================================
            if args.attack == "poisonedfl_adaptive" and getattr(Attack, "round", 0) == 0:
                # 恶意端像良性端一样正常训练，得到 byz_grads（用于初始化）
                for idx in idx_users[:num_byzs]:
                    grad, loss = benignWorker(model, data_loader[idx], optimizer, device)
                    byz_grads.append(grad)
                    local_losses.append(loss)
            elif args.attack == "poisonedfl_strict":
                # 严格poisonedfl按你之前的设定：恶意端不做本地训练，由攻击器生成
                pass
            else:
                # 其它攻击：沿用原来的 byzantineWorker 产生“恶意端梯度”
                for idx in idx_users[:num_byzs]:
                    grad, loss = byzantineWorker(model, data_loader[idx], optimizer, args)
                    byz_grads.append(grad)
                    local_losses.append(loss)

# ===================================================================================================

            # 良性客户端梯度
            for idx in idx_users[num_byzs:]:
                grad, loss = benignWorker(model, data_loader[idx], optimizer, device)
                benign_grads.append(grad)
                local_losses.append(loss)

# =============================改动点3================================================================
            # 攻击注入
            if args.attack == "poisonedfl_adaptive":
                # 第1轮：generate_attack 会用 byz_grads 初始化；之后轮次 byz_grads 可为空
                byz_grads = Attack.generate_attack(benign_grads, GAR, byz_grads=byz_grads)
            elif args.attack == "poisonedfl_strict":
                byz_grads = Attack.generate_attack(benign_grads, GAR)
            else:
                byz_grads = Attack(byz_grads, benign_grads, GAR)
# ===================================================================================================

            # 聚合
            local_grads = byz_grads + benign_grads
            global_grad, selected_idx, isbyz = GAR.aggregate(local_grads,
                                                             f=num_byzs,
                                                             epoch=epoch,
                                                             g0=(benign_grads[0] if len(benign_grads) > 0 else None),
                                                             iteration=it)

            # 记录一条“攻击成功率/良性入选率”的统计
            byz_rate.append(isbyz)
            benign_rate.append((len(selected_idx) - isbyz * num_byzs) / (num_users - num_byzs))

            # 更新模型
            tools.set_gradient_values(model, global_grad)
            optimizer.step()

# =============================改动点4================================================================
            # 更新攻击器状态
            if args.attack in ("poisonedfl_adaptive", "poisonedfl_strict"):
                Attack.update_state(model)
# ===================================================================================================

            # 记录损失
            if len(local_losses) > 0:
                loss_avg = sum(local_losses) / len(local_losses)
                iter_loss.append(loss_avg)

                if (it + 1) % 10 == 0:
                    print('[epoch %d, %.2f%%] loss: %.5f' %
                          (epoch + 1, 100 * ((it + 1) / iteration), loss_avg),
                          "--- byz. attack succ. rate:", isbyz,
                          '--- selected number:', len(selected_idx))

        if scheduler is not None:
            scheduler.step()

        return iter_loss

    # for epoch in range(args.epochs):
    #     loss = train_parallel(args, global_model, train_loader, optimizer, epoch, scheduler)
    #     acc = test_classification(device, global_model, test_loader)
    #     print("Test Accuracy: {}%".format(acc))
    # =============================
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        loss = train_parallel(args, global_model, train_loader, optimizer, epoch, scheduler)
        acc = test_classification(device, global_model, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

        print("Test Accuracy: {:.2f}% | Best Test Accuracy: {:.2f}% (epoch {})".format(
            acc, best_acc, best_epoch
        ))
