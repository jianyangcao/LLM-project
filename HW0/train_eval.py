# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier(初始化方法),并且默认跳过embedding层
def init_network(model, method='xavier', exclude='embedding', seed=123):
    #name：参数名字符串（例如‘embedding.weight）；w：张量参数（会被优化器更新的参数）
    for name, w in model.named_parameters():
        if exclude not in name:
            #初始化权重矩阵
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def _collect_type_logits_labels(model, data_iter):
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs_type, _ = model(texts)
            type_labels, _ = labels
            logits_all.append(outputs_type.data.cpu().numpy())
            labels_all.append(type_labels.data.cpu().numpy())
    if logits_all:
        return np.vstack(logits_all), np.vstack(labels_all)
    return np.zeros((0, 0)), np.zeros((0, 0))


def _best_thresholds_per_class(logits, labels, candidates):
    # logits/labels: [N, C]
    if logits.size == 0 or labels.size == 0:
        return np.array([])
    probs = 1.0 / (1.0 + np.exp(-logits))
    num_classes = labels.shape[1]
    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    for c in range(num_classes):
        y_true = labels[:, c].astype(np.int32)
        if y_true.sum() == 0:
            best_thresholds[c] = 0.5
            continue
        best_f1 = -1.0
        best_t = 0.5
        p = probs[:, c]
        for t in candidates:
            y_pred = (p >= t).astype(np.int32)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            if tp == 0 and (fp > 0 or fn > 0):
                f1 = 0.0
            else:
                denom = (2 * tp + fp + fn)
                f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[c] = best_t
    return best_thresholds


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train() #把模型切到训练模式
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=getattr(config, 'weight_decay', 0.0),
    )

    type_loss_fn = nn.BCEWithLogitsLoss(pos_weight=config.type_pos_weight)
    sent_loss_fn = nn.CrossEntropyLoss(weight=config.sentiment_class_weight)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    #TensorBoard记录训练。验证的loss和acc曲线
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        #内层 batch 循环：从 train_iter 一批批取数据并训练
        for i, (trains, labels) in enumerate(train_iter):
            outputs_type, outputs_sent = model(trains)
            type_labels, sent_labels = labels
            model.zero_grad()
            loss_type = type_loss_fn(outputs_type, type_labels)
            loss_sent = sent_loss_fn(outputs_sent, sent_labels)
            loss = loss_type + config.sentiment_loss_weight * loss_sent
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true_type = type_labels.data.cpu().numpy()
                pred_type = (torch.sigmoid(outputs_type.data) >= 0.6).int().cpu().numpy()
                train_type_f1 = metrics.f1_score(true_type, pred_type, average='micro', zero_division=0)
                true_sent = sent_labels.data.cpu()
                pred_sent = torch.max(outputs_sent.data, 1)[1].cpu()
                train_sent_acc = metrics.accuracy_score(true_sent, pred_sent)
                dev_type_f1, dev_sent_acc, dev_loss = evaluate(config, model, dev_iter)#调 evaluate 在验证集上计算
                #如果验证 loss 变好了：保存模型
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                #打印训练日志 &写TensorBoard
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Type F1: {2:>6.2%},  Train Sent Acc: {3:>6.2%},  Val Loss: {4:>5.2},  Val Type F1: {5:>6.2%},  Val Sent Acc: {6:>6.2%},  Time: {7} {8}'
                print(msg.format(total_batch, loss.item(), train_type_f1, train_sent_acc, dev_loss, dev_type_f1, dev_sent_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("f1/type_train", train_type_f1, total_batch)
                writer.add_scalar("f1/type_dev", dev_type_f1, total_batch)
                writer.add_scalar("acc/sent_train", train_sent_acc, total_batch)
                writer.add_scalar("acc/sent_dev", dev_sent_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, dev_iter, test_iter)


def test(config, model, dev_iter, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))#把 config.save_path 里保存的参数加载进模型
    model.eval()
    start_time = time.time()
    # per-class thresholds from dev
    cand = np.arange(0.1, 0.91, 0.05)
    dev_logits, dev_labels = _collect_type_logits_labels(model, dev_iter)
    config.type_thresholds = _best_thresholds_per_class(dev_logits, dev_labels, cand)
    test_type_f1, test_sent_acc, test_loss, type_report, sent_report, sent_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Type F1: {1:>6.2%},  Test Sent Acc: {2:>6.2%}'
    print(msg.format(test_loss, test_type_f1, test_sent_acc))
    print("Type Precision, Recall and F1-Score...")
    print(type_report)
    print("Sentiment Precision, Recall and F1-Score...")
    print(sent_report)
    print("Sentiment Confusion Matrix...")
    print(sent_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

#加载训练过程中保存的“验证集最优模型” → 切到评估模式 → 在测试集上跑一遍 evaluate
# → 打印指标（loss/acc/分类报告/混淆矩阵）+ 耗时
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    type_loss_fn = nn.BCEWithLogitsLoss(pos_weight=getattr(config, 'type_pos_weight', None))
    sent_loss_fn = nn.CrossEntropyLoss(weight=getattr(config, 'sentiment_class_weight', None))
    predict_type_all = []
    labels_type_all = []
    predict_sent_all = np.array([], dtype=int) #模型预测类别
    labels_sent_all = np.array([], dtype=int) #真实类别
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs_type, outputs_sent = model(texts)
            type_labels, sent_labels = labels
            loss_type = type_loss_fn(outputs_type, type_labels)
            loss_sent = sent_loss_fn(outputs_sent, sent_labels)
            loss_total += loss_type + config.sentiment_loss_weight * loss_sent
            type_true = type_labels.data.cpu().numpy()
            type_probs = torch.sigmoid(outputs_type.data).cpu().numpy()
            if getattr(config, 'type_thresholds', None) is not None and len(config.type_thresholds) == type_probs.shape[1]:
                type_pred = (type_probs >= config.type_thresholds).astype(np.int32)
            else:
                type_pred = (type_probs >= 0.5).astype(np.int32)
            labels_type_all.append(type_true)
            predict_type_all.append(type_pred)
            sent_true = sent_labels.data.cpu().numpy()
            sent_pred = torch.max(outputs_sent.data, 1)[1].cpu().numpy()
            labels_sent_all = np.append(labels_sent_all, sent_true)
            predict_sent_all = np.append(predict_sent_all, sent_pred)

    labels_type_all = np.vstack(labels_type_all) if labels_type_all else np.zeros((0, config.num_classes), dtype=int)
    predict_type_all = np.vstack(predict_type_all) if predict_type_all else np.zeros((0, config.num_classes), dtype=int)
    type_f1 = metrics.f1_score(labels_type_all, predict_type_all, average='micro', zero_division=0)
    sent_acc = metrics.accuracy_score(labels_sent_all, predict_sent_all) if len(labels_sent_all) else 0.0
    if test:
        type_report = metrics.classification_report(labels_type_all, predict_type_all, target_names=config.class_list, digits=4, zero_division=0)
        sent_report = metrics.classification_report(labels_sent_all, predict_sent_all, target_names=config.sentiment_class_list, digits=4, zero_division=0)
        sent_confusion = metrics.confusion_matrix(labels_sent_all, predict_sent_all)
        return type_f1, sent_acc, loss_total / len(data_iter), type_report, sent_report, sent_confusion
    return type_f1, sent_acc, loss_total / len(data_iter)
