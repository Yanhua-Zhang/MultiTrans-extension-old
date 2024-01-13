import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()  # 注意这里转 float

    def _dice_loss(self, score, target):
        target = target.float()     # 注意这里转 float
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)  # 对 prediction 进行归一化
        target = self._one_hot_encoder(target)  # 对 ground True 进行 one_hot_encode
        
        if weight is None:
            weight = [1] * self.n_classes  # 没有 weight 的话，这里为啥要乘以 n_classes ??
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        
        # 利用公式对每一类计算 dice loss 
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes  # 这里为啥要除以 n_classes ??

# 利用医学影像处理专用包 medpy 中的函数，对 binary objects 计算 metric
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)

        # 95th percentile of the Hausdorff Distance.
        # defined as the maximum surface distance between the objects.
        hd95 = metric.binary.hd95(pred, gt)  
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:   
        return 1, 0     # 这里 tmd 写错了吧？？？ 应该是 pred.sum()==0 and gt.sum()==0: 才应该 return 1, 0 吧？其他都是 0, 0 ?
    else:
        return 0, 0




def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    
    # 把 image 去掉 batch dim，然后放到 cpu 上，Tensor ---> numpy
    # 1, N, x, y ---> N, x, y
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        # 如果是 volume

        prediction = np.zeros_like(label)  # N, x, y
        for ind in range(image.shape[0]):  
            slice = image[ind, :, :]    # 取 volume 中的一个 slice。 N, x, y ---> x, y
            x, y = slice.shape[0], slice.shape[1]

            # 将 slice resize 到 patch_size 的大小
            if x != patch_size[0] or y != patch_size[1]:
                # 三次插值法（order=3)、最邻近插值算法（当参数order=0时）
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # x, y ---> 1, 1, x, y ---> float ---> GPU
            
            # 以 slice 的形式进行 inference
            net.eval()
            with torch.no_grad():
                outputs = net(input)

                # 1, classes, x, y ---> 1, 1, x, y ---> 1, x, y
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  # 将 pred 重新 resize 到 x, y 大小
                else:
                    pred = out
                prediction[ind] = pred  # 所有的 slice 的 predict 又重新组合为 volume  # 1, x, y ---> N, x, y
    else:
        # 对 singe slice 进行 inference
        # np ---> Tensor ---> x, y ---> 1, 1, x, y
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        # 因为要利用 binary objects 来计算 
        metric_list.append(calculate_metric_percase(prediction == i, label == i))  # 直接可以计算所有 sequences 的吗？

    # volume image 的保存。
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))  # 将数组转换成sitk图像
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))  # 这 spacing 都是 1 吗？？？
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))

        # 转为 NIfTI格式及其压缩格式进行储存：.nii、.nii.gz
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

# ================================================================================
# 计算 RGB 图像的 mDice, mIoU

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

# 利用医学影像处理专用包 medpy 中的函数，对 binary objects 计算 metric
def calculate_metric_percase_appended(pred, gt):
    # pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)

        # 95th percentile of the Hausdorff Distance.
        # defined as the maximum surface distance between the objects.
        hd95 = metric.binary.hd95(pred, gt)  

        mIoU = mean_iou_np(pred, gt)
        dice2 = mean_dice_np(pred, gt)

        return dice, hd95, mIoU, dice2
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 1
    else:
        return 0, 0, 0, 0

# -----------------------------------------------------------
def test_RGB_image(args, image, label, net, classes, test_save_path=None, case=None):

    # # 在 dataload 时候已经 resize 过了。    
    # # 把 image 去掉 batch dim，然后放到 cpu 上，Tensor ---> numpy
    # # 1, 3, x, y ---> 3, x, y ---> cpu ---> np
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # x, y = image.shape[2], image.shape[3]

    # image = image.transpose(1, 2, 0)  # 3, x, y ---> x, y, 3 
    # image_resized = cv2.resize(image, (patch_size[0], patch_size[1]), interpolation = cv2.INTER_LINEAR)
    # image_resized = image_resized.transpose(2, 0, 1)  # x, y, 3 ---> 3, x, y 
    
    # input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # 3, x, y ---> 1, 3, x, y ---> float ---> GPU

    label = label.squeeze(0).cpu().detach().numpy()  # 通过 Dataloader 加载，因此需要去掉一个维度。
    w, h = label.shape

    input = image.float().cuda()

    # 进行 inference
    net.eval()
    with torch.no_grad():
        outputs = net(input)

        # 1, classes, x, y ---> 1, x, y ---> x, y
        if args.num_classes == 1:
            out = outputs.sigmoid().data.cpu().numpy().squeeze()
            out = 1*(out > 0.5)
        else:
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  
            out = out.cpu().detach().numpy()

        x, y = out.shape[0], out.shape[1]

        # print(out.shape)

        if x !=w or y != h:
            # pred = zoom(out, (x / w, y / h), order=0)  # 将 pred 重新 resize 到与 label 相同大小
            pred = cv2.resize(out, (h, w), interpolation=cv2.INTER_NEAREST)
        else:
            pred = out

        prediction = pred  # 

    metric_list = []
    # for i in range(1, classes):
        # 因为要利用 binary objects 来计算 
    metric_list.append(calculate_metric_percase_appended(prediction, label))  # 直接可以计算所有 sequences 的吗？

    # volume image 的保存。
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))  # 将数组转换成sitk图像
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # img_itk.SetSpacing((1, 1, z_spacing))  # 这 spacing 都是 1 吗？？？
        # prd_itk.SetSpacing((1, 1, z_spacing))
        # lab_itk.SetSpacing((1, 1, z_spacing))

        # 转为 NIfTI格式及其压缩格式进行储存：.nii、.nii.gz
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.png")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.png")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.png")
        
    return metric_list