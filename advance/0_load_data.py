"""数据加载和处理"""
""" 1.加载包 """

# from __future__ import print_function, division
import os
import torch
"""
pandas 主要用于处理表格型数据（如DataFrame），适合数据分析和操作；
numpy 主要用于处理数值型数组，适合科学计算和数值操作。
"""
import pandas as pd   
import numpy as np           #用于更容易地进行csv解析
from skimage import io, transform    #用于图像的IO和变换
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
"""
torchvision.transforms 模块主要用于对图像数据进行各种预处理和增强操作，主要包括：

1. 图像格式转换：
   - ToTensor(): 将PIL图像或numpy数组转换为PyTorch张量
   - ToPILImage(): 将张量或numpy数组转换回PIL图像

2. 图像大小调整：
   - Resize(): 调整图像大小
   - CenterCrop(): 中心裁剪
   - RandomCrop(): 随机裁剪

3. 图像翻转和旋转：
   - RandomHorizontalFlip(): 随机水平翻转
   - RandomVerticalFlip(): 随机垂直翻转
   - RandomRotation(): 随机旋转

4. 颜色变换：
   - ColorJitter(): 随机改变亮度、对比度、饱和度和色调
   - Grayscale(): 将图像转换为灰度图
   - RandomGrayscale(): 随机将图像转换为灰度图

5. 标准化和归一化：
   - Normalize(): 用均值和标准差对张量图像进行标准化
   - RandomErasing(): 随机擦除图像区域

6. 组合操作：
   - Compose(): 将多个transform操作组合在一起

这些transform操作可以：
- 提高模型的泛化能力
- 增强数据多样性
- 统一输入数据格式
- 提高训练效率
"""

from torchvision import transforms, utils

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

"""数据集注释，
数据集是按如下规则打包成的csv文件:
image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y
0805personali01.jpg,27,83,27,98, ... 84,134
1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312
"""



"""读取数据集
这段代码主要展示了如何加载和处理面部关键点数据集。让我们逐步解释：

1. 数据集格式：
   - 数据集以CSV文件格式存储
   - 每行代表一张图像及其对应的68个面部关键点坐标
   - 第一列是图像文件名
   - 后续列是68个关键点的x,y坐标，格式为part_0_x,part_0_y,...,part_67_x,part_67_y

2. 代码功能：
   - 使用pandas读取CSV文件到DataFrame
   - 选择第65个样本（索引从0开始）
   - 提取图像文件名和关键点坐标
   - 将关键点坐标转换为numpy数组并reshape为(-1,2)格式，即每行一个点的(x,y)坐标
   - 打印图像文件名、关键点数组形状和前4个关键点坐标

3. 关键点说明：
   - 68个关键点按照标准面部关键点标注顺序排列
   - 每个点代表面部特定位置，如眼角、鼻尖、嘴角等
   - reshape后的数组形状为(68,2)，方便后续处理和可视化

这个数据集可以用于面部关键点检测、面部对齐等计算机视觉任务。

"""


landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65


# 这行代码使用pandas的iloc方法从landmarks_frame DataFrame中提取第n行的数据
# iloc[n, 0]表示获取第n行的第0列数据，即图像文件名
# landmarks_frame是一个包含图像文件名和关键点坐标的DataFrame
# n是我们要选择的样本索引，这里n=65表示选择第66个样本（索引从0开始）
# 这行代码的作用是获取第n个样本对应的图像文件名

img_name = landmarks_frame.iloc[n, 0]
# 这行代码使用pandas的iloc方法从landmarks_frame DataFrame中提取第n行的数据
# iloc[n, 1:]表示获取第n行从第1列开始到最后一列的所有数据，即68个关键点的x,y坐标
# .values将提取的数据转换为numpy数组
# 这样我们就可以得到一个包含136个元素的一维数组，每个元素对应一个坐标值
# 这个数组后续会被reshape为(68,2)的形状，表示68个关键点的(x,y)坐标

landmarks = landmarks_frame.iloc[n, 1:].values  # 将 as_matrix() 替换为 values
# reshape(-1, 2) 的作用是将一维数组重新调整为二维数组，其中：
# -1 表示自动计算该维度的大小，保持总元素数不变
# 2 表示每行包含2个元素，即每个关键点的x,y坐标
# 原始landmarks数组包含136个元素（68个关键点 * 2个坐标）
# 经过reshape后，数组被重新组织为68行，每行2列，表示68个关键点的(x,y)坐标
# 因此最终形状为(68,2)

landmarks = landmarks.astype('float').reshape(-1, 2)


print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


"""
写一个简单的函数来展示一张图片和它对应的标注点作为例子。
"""

# 图像显示后立即消失是因为plt.pause(0.001)只暂停了很短时间
# 我们需要使用plt.show(block=True)来保持图像窗口打开
# 或者可以在show_landmarks函数中添加plt.show()
# 这里我们选择在函数外部使用plt.show()来保持代码结构清晰
def show_landmarks(image, landmarks):
    # 这段代码用于显示带有面部关键点标注的图像
    # plt.imshow(image) 用于显示输入图像
    # plt.scatter() 用于在图像上绘制关键点
    #   - landmarks[:, 0] 获取所有关键点的x坐标
    #   - landmarks[:, 1] 获取所有关键点的y坐标
    #   - s=10 设置点的大小为10
    #   - marker='.' 设置点的形状为圆点
    #   - c='r' 设置点的颜色为红色
    # plt.pause(3) 使图像显示3秒钟，以便观察
    """显示带有地标的图片"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(3)  # pause a bit so that plots are updated


plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),landmarks)
plt.show()


"""
数据集类：
自定义数据集应继承Dataset并覆盖以下方法 
* __len__ 实现 len(dataset) 返还数据集的尺寸。
* __getitem__用来获取一些索引数据，例如 dataset[i] 中的(i)。

so，我们可以自定义数据集类了：
"""
class FaceLandmarksDataset(Dataset):


    """面部标记数据集."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)



    # 这段代码实现了获取数据集中指定索引的样本
    # idx: 要获取的样本索引
    # 1. 首先根据索引从csv文件中获取图像文件名
    # 2. 使用io.imread读取图像文件
    # 3. 从csv文件中获取对应的关键点坐标
    # 4. 将关键点坐标转换为numpy数组并调整形状
    # 5. 将图像和关键点打包成一个字典sample
    # 6. 如果定义了transform，则对sample进行变换
    # 7. 返回处理后的sample
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

"""数据可视化
实例化这个类并遍历数据样本。我们将会打印出前四个例子的尺寸并展示标注的特征点。 
"""


face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',                                    root_dir='data/faces/')
fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


"""
数据变化：
通过上面的例子我们会发现图片并不是同样的尺寸。
绝大多数神经网络都假定图片的尺寸相同。
因此我们需要做一些预处理。
让我们创建三个转换: 
* Rescale：缩放图片 
* RandomCrop：对图片进行随机裁剪。这是一种数据增强操作 
* ToTensor：把numpy格式图片转为torch格式图片 (我们需要交换坐标轴).
"""

class Rescale(object):
    """将样本中的图像重新缩放到给定大小。.

    Args:
        output_size（tuple或int）：所需的输出大小。
        如果是元组，则输出为
        与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
        isinstance(output_size, (int, tuple)) 检查 output_size 是否是 int 类型或 tuple 类型。
        如果是，程序继续执行；如果不是，抛出异常。
    """
    
    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    """
    __call__ 是 Python 中的一个特殊方法，它允许类的实例像函数一样被调用。
    但**__call__ 并不是自动被调用的**，只有在显式地将实例作为函数调用时，它才会被执行。

    1）这里是init被调用
    random_crop = RandomCrop(output_size=(100, 100))
    2）这里是call被调用
    cropped_sample = random_crop(sample)
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """随机裁剪样本中的图像.

    Args:
       output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。         
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # 交换颜色轴因为
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


"""
组合转换：
我们想要把图像的短边调整为256，然后随机裁剪(randomcrop)为224大小的正方形。
也就是说，我们打算组合一个Rescale和 RandomCrop的变换。 
我们可以调用一个简单的类 torchvision.transforms.Compose来实现这一操作。

"""


scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

""""
# 在样本上应用上述的每个变换。
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
"""
"""
迭代数据集整体修改在展示：
"""

transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

"""
一些优化：
但是，对所有数据集简单的使用for循环牺牲了许多功能，
尤其是: * 批量处理数据 * 打乱数据 * 使用多线程multiprocessingworker 并行加载数据。

torch.utils.data.DataLoader是一个提供上述所有这些功能的迭代器。
下面使用的参数必须是清楚的。一个值得关注的参数是collate_fn, 可以通过它来决定如何对数据进行批处理。但是绝大多数情况下默认值就能运行良好。


"""

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# 辅助功能：显示批次
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


if  __name__ == '__main__':

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['landmarks'].size())

        # 观察第4批次并停止。
        if i_batch == 3:

            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break








