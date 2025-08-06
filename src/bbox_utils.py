import numpy as np
import torch
import numpy as np

def bbox_deduplicate(_surf_pos, padding="zero"):
    """

    Args:
        _surf_pos: Nx10
        max_surf:
        padding:

    Returns:

    """

    if isinstance(_surf_pos, torch.Tensor):
        pass
    elif isinstance(_surf_pos, np.ndarray):
        _surf_pos = torch.from_numpy(_surf_pos)
    else:
        raise NotImplementedError()

    max_surf = _surf_pos.shape[-2]
    dedup_mask = np.zeros((max_surf), dtype=np.bool)  # 记录哪些BBox (为True的部分) 是被dedup掉的

    if padding == "repeat":
        bbox_threshold = 0.08
    elif padding == "zero":
        bbox_threshold = 2e-4

    bboxes = torch.concatenate(
        [_surf_pos[0][:, :6].unflatten(-1, torch.Size([2, 3])), _surf_pos[0][:, 6:].unflatten(-1, torch.Size([2, 2]))]
        , dim=-1).detach().cpu().numpy()

    non_repeat = None
    for bbox_idx, bbox in enumerate(bboxes):
        if padding == "repeat":
            if non_repeat is None:
                non_repeat = bbox[np.newaxis, :, :]
                is_deduped = False
            else:
                diff = np.max(np.max(np.abs(non_repeat - bbox)[..., -2:], -1), -1)  #
                same = diff < bbox_threshold
                bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., -2:], -1), -1)  # [...,-2:]
                same_rev = diff_rev < bbox_threshold
                if same.sum() >= 1 or same_rev.sum() >= 1:
                    is_deduped = True  # 当前BBox是否被去重了
                else:
                    is_deduped = False
                    non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

        if padding == "zero":
            # （2D）判断BBox的大小是否非0
            is_deduped = False
            v = 1
            for h in (bbox[1] - bbox[0])[3:]:
                v *= h
            if v < bbox_threshold:
                is_deduped = True
            # elif non_repeat is not None:  # 去重复的（zero padding 也有概率会产生重复）
            #     bbox_threshold_2 = 0.03
            #     diff = np.max(np.max(np.abs(non_repeat - bbox)[..., :3], -1), -1)  #
            #     same = diff < bbox_threshold_2
            #     bbox_rev = bbox[::-1]  # also test reverse bbox for matching
            #     diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., :3], -1), -1)  # [...,-2:]
            #     same_rev = diff_rev < bbox_threshold_2
            #     if same.sum() >= 1 or same_rev.sum() >= 1:
            #         is_deduped = True  # 当前BBox是否被去重了
            if is_deduped == False:
                if non_repeat is None:
                    non_repeat = bbox[np.newaxis, :, :]
                else:
                    non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

        dedup_mask[bbox_idx] = is_deduped

    bboxes = np.concatenate([non_repeat[:, :, :3].reshape(len(non_repeat), -1), non_repeat[:, :, 3:].reshape(len(non_repeat), -1)], axis=-1)

    return bboxes, dedup_mask


from scipy.optimize import linear_sum_assignment


def get_diff_map(A, B):
    """
    根据 BBox 来做板片的匹配计算（其实也可以用其它的信息，只要符合输入）
    Args:
        A: NxD
        B: NxD
    Returns:
    """
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()

    diff_map = abs(A[:, None, :] - B[None, :, :])

    diff_map = np.abs(np.mean(diff_map, axis=-1))
    # _, min_indices = torch.min(diff_map, dim=1)
    # 匈牙利算法
    row_ind, col_ind = linear_sum_assignment(diff_map)
    # 总代价
    cost = diff_map[row_ind, col_ind]
    cost_total = cost.sum()

    return diff_map, cost, cost_total, row_ind, col_ind


def bbox_3d_iou(pred_bboxes, gt_bboxes):
    """
    计算 3D IoU（向量化版本）。
    pred_bboxes, gt_bboxes: Nx10 数组，前 6 维为 [x_min, y_min, z_min, x_max, y_max, z_max]
    返回: Nx1 的 3D IoU 数组
    """
    # 提取 3D 坐标
    pred_3d = pred_bboxes[:, :6]  # [N, 6]
    gt_3d = gt_bboxes[:, :6]      # [N, 6]

    # 交集坐标
    x_min_inter = np.maximum(pred_3d[:, 0], gt_3d[:, 0])
    y_min_inter = np.maximum(pred_3d[:, 1], gt_3d[:, 1])
    z_min_inter = np.maximum(pred_3d[:, 2], gt_3d[:, 2])
    x_max_inter = np.minimum(pred_3d[:, 3], gt_3d[:, 3])
    y_max_inter = np.minimum(pred_3d[:, 4], gt_3d[:, 4])
    z_max_inter = np.minimum(pred_3d[:, 5], gt_3d[:, 5])

    # 交集体积
    inter_lengths = np.stack([
        x_max_inter - x_min_inter,
        y_max_inter - y_min_inter,
        z_max_inter - z_min_inter
    ], axis=1)
    inter_lengths = np.maximum(inter_lengths, 0)  # 负值置为 0
    inter_volume = np.prod(inter_lengths, axis=1)

    # 预测和真实 bbox 体积
    pred_lengths = pred_3d[:, 3:6] - pred_3d[:, 0:3]
    gt_lengths = gt_3d[:, 3:6] - gt_3d[:, 0:3]
    pred_lengths = np.maximum(pred_lengths, 0)  # 防止负值
    gt_lengths = np.maximum(gt_lengths, 0)
    pred_volume = np.prod(pred_lengths, axis=1)
    gt_volume = np.prod(gt_lengths, axis=1)

    # 并集体积
    union_volume = pred_volume + gt_volume - inter_volume

    # IoU
    iou_3d = np.where(union_volume > 0, inter_volume / union_volume, 0.0)
    return iou_3d


def bbox_2d_iou(pred_bboxes, gt_bboxes):
    """
    计算 2D IoU（向量化版本）。
    pred_bboxes, gt_bboxes: Nx10 数组，indices 指定 2D 坐标 [x_min, y_min, x_max, y_max]
    返回: Nx1 的 2D IoU 数组
    """
    # 提取 2D 坐标
    pred_2d = pred_bboxes  # [N, 4]
    gt_2d = gt_bboxes      # [N, 4]

    # 交集坐标
    x_min_inter = np.maximum(pred_2d[:, 0], gt_2d[:, 0])
    y_min_inter = np.maximum(pred_2d[:, 1], gt_2d[:, 1])
    x_max_inter = np.minimum(pred_2d[:, 2], gt_2d[:, 2])
    y_max_inter = np.minimum(pred_2d[:, 3], gt_2d[:, 3])

    # 交集面积
    inter_lengths = np.stack([
        x_max_inter - x_min_inter,
        y_max_inter - y_min_inter
    ], axis=1)
    inter_lengths = np.maximum(inter_lengths, 0)  # 负值置为 0
    inter_area = np.prod(inter_lengths, axis=1)

    # 预测和真实 bbox 面积
    pred_lengths = pred_2d[:, 2:4] - pred_2d[:, 0:2]
    gt_lengths = gt_2d[:, 2:4] - gt_2d[:, 0:2]
    pred_lengths = np.maximum(pred_lengths, 0)
    gt_lengths = np.maximum(gt_lengths, 0)
    pred_area = np.prod(pred_lengths, axis=1)
    gt_area = np.prod(gt_lengths, axis=1)

    # 并集面积
    union_area = pred_area + gt_area - inter_area

    # IoU
    iou_2d = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou_2d



def evaluate_bboxes_iou(pred_bboxes, gt_bboxes, indices_2d=[6, 7, 8, 9]):
    """
    Example of usage

    pred_bboxes Nx10
    gt_bboxes   Nx10
    """
    raise NotImplementedError

    ious_3d = [bbox_3d_iou(pred_bboxes[i], gt_bboxes[i]) for i in range(len(pred_bboxes))]
    ious_2d = [bbox_2d_iou(pred_bboxes[i][indices_2d], gt_bboxes[i][indices_2d]) for i in range(len(pred_bboxes))]

    mIoU_3d = sum(ious_3d) / len(ious_3d) if ious_3d else 0.0
    mIoU_2d = sum(ious_2d) / len(ious_2d) if ious_2d else 0.0

    return mIoU_3d, mIoU_2d


def bbox_l2_distance(pred_bbox, gt_bbox):
    return np.sqrt(np.sum((pred_bbox - gt_bbox) ** 2))