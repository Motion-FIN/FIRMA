import torch
import torch.nn.functional as F
from utils.config import children
from functools import partial


def gram_matrix(input):
    batch_size, channel, height, width = input.size()
    features = input.view(batch_size * channel, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channel * height * width)

def gram_matrix_loss(input, target):
    input_gram = gram_matrix(input)
    target_gram = gram_matrix(target)
    # print(f'input_gram: {input_gram}, target_gram : {target_gram}')
    loss = F.mse_loss(input_gram*10000, target_gram*10000)
    # loss = F.l1_loss(input_gram*1000, target_gram*1000)
    # print(f'gram_loss: {loss}')
    return loss

def rom_loss(rom_boxes, predict, target):
    gt_losses = []
    k = 0
    for boxes in rom_boxes:
        box_losses = []
        
        # # /// 230807 추가 start
        # # 박스의 크기 계산
        # box_sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]

        # # 크기에 따른 가중치 설정 (크기가 클수록 가중치가 크도록)
        # max_size = max(box_sizes)
        # min_size = min(box_sizes)
        # normalized_sizes = [(size - min_size) / (max_size - min_size) for size in box_sizes]
        # bboxWeights = [1 + 2 * normalized_size for normalized_size in normalized_sizes]
        # print(f'{k}.bboxWeights : {bboxWeights}')
        # # /// 230807 추가 end

        for i, bbox in enumerate(boxes):
            x, y, w, h = bbox
            # print(f'x:{x}, y:{y}, w:{w}, h:{h}')
            # print(f'predict.shape:{predict[k].shape}')
            pred_ = predict[k].permute(1, 2, 0)
            gt_ = target[k].permute(1, 2, 0)
            # print(f'pred_.shape:{pred_.shape} // gt_.shape:{gt_.shape}')
            # plt.imshow(pred_)
            # plt.axis('off')  # Turn off axis
            # plt.show()
            # plt.imshow(gt_)
            # plt.axis('off')  # Turn off axis
            # plt.show()
            # bounding box area -> im0, im1 extraction
            yy = int(y)
            yh = int(y+h)
            xx = int(x)
            xw = int(x+w)
            bbox_pred = pred_[yy:yh, xx:xw, :]
            bbox_gt = gt_[yy:yh, xx:xw, :]
            # print(f'bbox_pred.shape:{bbox_pred.shape} // bbox_gt.shape:{bbox_gt.shape}')
            # print(f'pred_.shape:{bbox_pred} // bbox_gt.shape:{bbox_gt}')
            # result_frame = pred_.copy()
            # cv2.rectangle(result_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            # plt.imshow(result_frame)
            # plt.axis('off')  # Turn off axis
            # plt.show()
            # bbox 별 가중치 bbox_weight : bboxW = 1 부터 0 까지 
            # # /// 이 부분 이전버전 
            # diff_loss = torch.abs(bbox_gt - bbox_pred)
            # if not torch.isnan(diff_loss).any():
            #     # /// 230807 추가 start
            #     # weightedLoss = diff_loss # * bboxWeights[i]
            #     # /// 230807 추가 end
            #     # weightedLoss = diff_loss
            #     Rom_loss = torch.mean(diff_loss)
            #     if not torch.isnan(Rom_loss).any(): 
            #         box_losses.append(Rom_loss)
            # /// 이 부분 수정버전
            diff_loss = F.l1_loss(bbox_pred, bbox_gt)
            if not torch.isnan(diff_loss).any():
                box_losses.append(torch.mean(diff_loss))

        boxLoss_avg = torch.mean(torch.tensor(box_losses))
        if not torch.isnan(boxLoss_avg).any():
            gt_losses.append(boxLoss_avg)

        # print(f'Rom_losses:{gt_losses}')
        k = k+1
    loss = torch.mean(torch.tensor(gt_losses))
    # print(f'loss:{loss}')
    return loss

def _make_loss(losses):
    def f(pred,gt):
        metrics = dict()
        loss = 0
        for tag, L, w in losses:
            tag = 'loss/'+tag
            metrics[tag] = L(pred,gt)
            loss += metrics[tag]*w
        return loss, metrics
    return f

def make_vfi2_loss(cfg):
    loss_fn = {
        'l1': lambda pred, gt, opt : F.l1_loss(pred['final'],gt),
        'l1_tea': lambda pred,gt, opt: F.l1_loss(pred['merged_tea'],gt),
        'distill': lambda pred, gt, opt : pred['loss_distill'],
        'rom': lambda pred, gt, opt : rom_loss(pred['bounding_box'], pred['final'], gt),
        # 'gram': lambda pred, gt, opt : gram_matrix_loss(pred['final'], gt),        
        }
    losses = []
    for tag, opt in children(cfg):
        losses.append((tag,partial(loss_fn[tag],opt=opt),opt.w))
    return _make_loss(losses)