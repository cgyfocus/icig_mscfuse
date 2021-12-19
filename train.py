# -*- coding:utf-8-*-
import scipy.io as scio
import torch
import torch.nn as nn
from torch.optim import Adam                             # 从 torch.optim 中导出 Adam 优化器
from torch.autograd import Variable                      # 创建变量，变量是篮子，用来存放数据
import get_data                                          # 数据读取
from net import NetWork                                  # 网络模型
import argparse                                          # 解析库
from loss import MEF_SSIM_Loss                           # MEF_SSIM 损失函数



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 1]')
parser.add_argument('--data_ir', default='./data/ir/', help='dataset directory')
parser.add_argument('--data_vis',  default='./data/vis/', help='dataset directory')
parser.add_argument('--loss_interval', type=int, default=500, help='interval numbers')
parser.add_argument('--log_dir', default='loss', help='Log dir [default: log]')
parser.add_argument('--nums', type=int, default=10000, help='Picture Numbers ')
parser.add_argument('--max_epoch', type=int, default=128, help='Epoch to run [default: 128]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
FLAGS = parser.parse_args()


def main():
    # 第一步 对数据进行读取
    ir_path  = FLAGS.data_ir
    vis_path = FLAGS.data_vis

    original_imgs_path_ir = get_data.list_images(ir_path)         # 读取红外图像
    original_imgs_path_vis = get_data.list_images(vis_path)       # 读取可见光图像


    train_num = FLAGS.nums                                 # 10000
    imgs_path_ir  = original_imgs_path_ir[:train_num]      # 对数据集进行切片
    imgs_path_vis = original_imgs_path_vis[:train_num]     # 对数据集进行切片
    print(len(imgs_path_ir))
    train(imgs_path_ir, imgs_path_vis)                                # 训练数据

def train(imgs_path_ir, imgs_path_vis):
    """
    开始训练网络
    """
    # 网络参数       关于网络的参数一律大写
    BATCH_SIZE    = FLAGS.batch_size
    LEARNING_RATE = FLAGS.learning_rate                             # LEARNING_RATE = 0.0001
    MAX_EPOCH     = FLAGS.max_epoch                                 # MAX_EPOCH = 8


    # 网络模型
    # ---------------------------------------------------------------------------------
    net_model = NetWork()
    # ---------------------------------------------------------------------------------

    # 网络优化器 Adam优化器
    optimizer = Adam(net_model.parameters(), LEARNING_RATE, weight_decay=1e-4)

    # 损失函数
    criterion = MEF_SSIM_Loss()
    l1 = nn.L1Loss()

    # GPU 加速训练
    net_model.cuda()

    print('************* Training Begins *************')

    loss_pixel = []
    loss_ssim = []
    loss_all = []
    count = 0

    pixel_batch = []
    ssim_batch = []
    all_batch = []
    for i in range(MAX_EPOCH):
        print('  ============ Epoch %d ============ ' % i)

        # 加载数据
        batches  = get_data.information(imgs_path_ir, BATCH_SIZE)
        path_ir  = get_data.load_dataset(imgs_path_ir)
        path_vis = get_data.load_dataset(imgs_path_vis)
        net_model.train()

        # 计算batch的损失
        pixel_b = 0.0
        ssim_b = 0.0
        all_b = 0.0

        for batch in range(batches):

            ir_path_batch  = path_ir[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]                   # [0，4]
            vis_path_batch = path_vis[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]                 # [0，4]
            img_ir  = get_data.get_train_images(ir_path_batch, BATCH_SIZE)
            img_vis = get_data.get_train_images(vis_path_batch, BATCH_SIZE)

            count += 1

            # 将下一个梯度的变化 delta 置为0
            optimizer.zero_grad()

            # 输入数据
            images_ir  = Variable(img_ir, requires_grad=False)                   # Variable 是篮子， tensor 是鸡蛋
            images_vis = Variable(img_vis, requires_grad=False)                  # Variable 是篮子， tensor 是鸡蛋
            images_ir  = images_ir.cuda()                                        # 数据迁移到 GPU
            images_vis = images_vis.cuda()

            # 读取数据
            outputs = net_model(img_ir=images_ir, img_vis=images_vis)


            # 计算 loss
            ssim_loss = criterion(y1=images_ir, y2=images_vis, yf=outputs)
            l1_loss = l1(outputs, images_vis)

            total_loss = ssim_loss + 2000 * l1_loss

            # batch loss 计算
            pixel_b += l1_loss.item()
            ssim_b += ssim_loss.item()
            all_b += total_loss.item()

            total_loss.backward()                           # 梯度反传播
            optimizer.step()

            loss_interval = 100
            #if (count + 1) % loss_interval == 0:  # 每 100 次输出一次 loss 值
            print('SSIM: %6F, L1: %6f, TOTAL: %6f' % (ssim_loss, l1_loss,  total_loss))
            loss_pixel.append(l1_loss.item())
            loss_ssim.append(ssim_loss.item())
            loss_all.append(total_loss.item())

        # 计算每个 epoch 平均 loss
        pixel_batch.append(pixel_b / batches)
        ssim_batch.append(ssim_b / batches)
        all_batch.append(all_b / batches)

        # 保存模型
        save_model_path = 'model' + '/' + str(i) + 'model.model'
        torch.save(net_model.state_dict(), save_model_path)

    # 保存每个batch的平均 loss
    pixel_path = 'loss' + '/' + 'loss_pixel_batch' + '.mat'
    ssim_path = 'loss' + '/' + 'loss_ssim_batch' + '.mat'
    all_path = 'loss' + '/' + 'loss_total_batch' + '.mat'

    # 保存损失函数 以字典形式保存
    scio.savemat(pixel_path, {'pb':  pixel_batch})
    scio.savemat(ssim_path, {'sb': ssim_batch})
    scio.savemat(all_path, {'ab': all_batch})


    loss_pixel_path = 'loss' + '/' + 'loss_pixel' + '.mat'
    loss_ssim_path = 'loss' + '/' + 'loss_ssim' + '.mat'
    loss_all_path = 'loss' + '/' + 'loss_total' + '.mat'
    scio.savemat(loss_pixel_path, {'pixel': loss_pixel})
    scio.savemat(loss_ssim_path, {'ssim': loss_ssim})
    scio.savemat(loss_all_path, {'all': loss_all})

    # 保存网络模型

if __name__ == '__main__':
    main()