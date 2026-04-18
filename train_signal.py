import time
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from WGFD import Extractor
from torch.profiler import profile, record_function, ProfilerActivity
from dataloader import *
import random
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置pytorch内置hash函数种子，保证不同运行环境下字典等数据结构的哈希结果一致
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False禁用
class Config:
    def __init__(
            self,
            batch_size: int = 16,
            test_batch_size: int = 8,
            epochs: int = 100,
            lr: float = 0.001,
            save_path: str = 'model_weight/extractor.pth',
            device_num: int = 0,
            rand_num: int = 30,
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.device_num = device_num
        self.rand_num = rand_num

def train(extractor, train_dataloader, optimizer1, epoch, writer, device_num):
    extractor.train()
    device = torch.device("cuda:" + str(device_num))
    correct = 0
    for data_nnl in train_dataloader:
        data, target = data_nnl
        target = target.squeeze().long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        output = extractor(data)
        optimizer1.zero_grad()
        output = F.log_softmax(output, dim=1)
        total_loss = loss(output, target)
        total_loss.backward()
        optimizer1.step()
        total_loss += total_loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    mean_loss = total_loss / len(train_dataloader)
    mean_accuracy = 100 * correct / len(train_dataloader.dataset)
    print(
        'Train Epoch: {} \tLoss: {:.6f}, Accuracy_devices: {}/{} ({:0f}%)\n'.format(
            epoch,
            mean_loss,
            correct,
            len(train_dataloader.dataset),
            mean_accuracy,
        )
    )
    writer.add_scalar('Accuracy/train', mean_accuracy, epoch)
    writer.add_scalar('Loss/train', mean_loss, epoch)  # 用于训练准确率和分类器损失的可视化
    return mean_loss, mean_accuracy

def evaluate(extractor, loss, val_dataloader, epoch, writer, device_num):
    extractor.eval()
    val_loss = 0
    correct = 0
    device = torch.device("cuda:" + str(device_num))
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = extractor(data)
            output = F.log_softmax(output, dim=1)
            val_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_dataloader)
    mean_accuracy = 100 * correct / len(val_dataloader.dataset)
    fmt = '\nValidation set: loss_devices: {:.4f}, Accuracy_devices: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            val_loss,
            correct,
            len(val_dataloader.dataset),
            mean_accuracy,
        )
    )
    writer.add_scalar('Accuracy/validation', mean_accuracy, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    return val_loss, mean_accuracy

def test(extractor, test_dataloader):
    extractor.eval()
    correct = 0
    pred_devices = []
    real_devices = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = extractor(data)
            output = F.log_softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)  # 预测结果
            correct_mask = pred.eq(target.view_as(pred))  # 创建一个正确分类的掩码
            correct += correct_mask.sum().item()
            pred_devices[len(pred_devices):len(target) - 1] = pred.tolist()
            real_devices[len(real_devices):len(target) - 1] = target.tolist()
        pred_devices = np.array(pred_devices)
        real_devices = np.array(real_devices)
    accuracy = correct / len(test_dataloader.dataset)
    print("Accuracy_devices:", accuracy)
    return pred_devices, real_devices
def train_and_evaluate(extractor, loss_function, train_dataloader, val_dataloader, optimizer1, epochs, writer, save_path_extractor, device_num):
    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []
    current_max_val_accuracy = 1  # 初始化当前最小的测试损失为一个较大的值，判断模型是否有改进
    current_loss = 50
    time_start1 = time.time()
    for epoch in range(1, epochs + 1):
        time_start = time.time()
        train_loss, train_acc = train(extractor, train_dataloader, optimizer1, epoch, writer, device_num)
        train_losses.append(train_loss)
        val_loss, val_accuracy = evaluate(extractor, loss_function, val_dataloader, epoch, writer, device_num)
        val_loss = val_loss
        val_accuracy = val_accuracy
        if val_accuracy > current_max_val_accuracy or val_loss < current_loss:
            print("Model improved: ", end="")
            if val_accuracy > current_max_val_accuracy:
                print("Accuracy increased from {:.4f} to {:.4f}".format(current_max_val_accuracy, val_accuracy),
                      end="; ")
                current_max_val_accuracy = val_accuracy
            if val_loss < current_loss:
                print("Loss decreased from {:.6f} to {:.6f}".format(current_loss, val_loss), end="; ")
                current_loss = val_loss
            print("\nNew model weight is saved.")
            torch.save(extractor, save_path_extractor)
        else:
            print("No improvement: Accuracy = {:.4f}, Loss = {:.6f}".format(val_accuracy, val_loss))
        time_end = time.time()
        time_sum = time_end - time_start
        print("time for each epoch is: %s" % time_sum)
        print("------------------------------------------------")
        torch.cuda.empty_cache()  # 每轮训练结束清空未使用的显存
    time_end1 = time.time()
    Ave_epoch_time = (time_end1 - time_start1) / epochs
    print("Avgtime for each epoch is: %s" % Ave_epoch_time)
    return train_losses, train_accies, val_losses, val_accies


if __name__ == '__main__':
    conf = Config()
    writer = SummaryWriter("logs")
    device = torch.device("cuda:" + str(conf.device_num))
    RANDOM_SEED = 300
    set_seed(RANDOM_SEED)
    run_for = 'Train'
    if run_for == 'Train':
        X_train, X_val, Y_train, Y_val  = read_train_data()
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        X_test, Y_test = read_test_data()
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, drop_last=True)#drop_last为True则丢失最后一个不完整的批次
        val_dataloader = DataLoader(val_dataset, batch_size=conf.test_batch_size, shuffle=True, drop_last=False)##数据封装格式修改为和paper6一样，运行速度更快
        test_dataloader = DataLoader(test_dataset)
        extractor = Extractor().to(device)
        print(extractor)
        model = Extractor()
        loss = nn.NLLLoss()
        if torch.cuda.is_available():
            loss = loss.to(device)
        optim_extractor = torch.optim.Adam(extractor.parameters(), lr=conf.lr, weight_decay=0)
        time_start = time.time()
        train_losses, train_accies, val_losses, val_accies = train_and_evaluate(extractor,
                                                                                loss_function=loss,
                                                                                train_dataloader=train_dataloader,
                                                                                val_dataloader=val_dataloader,
                                                                                optimizer1=optim_extractor,
                                                                                epochs=conf.epochs,
                                                                                writer=writer, save_path_extractor=conf.save_path,
                                                                                device_num=conf.device_num,)
        time_end = time.time()
        time_sum = time_end - time_start
        print("total training time is: %s" % time_sum)
        extractor = torch.load('model_weight/extractor.pth')
        pred_devices, real_devices = test(extractor, test_dataloader)

