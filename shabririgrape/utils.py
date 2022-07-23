import torch


def convert_to_soft_label(label: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Convert label to soft label.
    Reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf

    Args:
        label (torch.Tensor): Label.

    Returns:
        torch.Tensor: Soft label.
    """
    classes_idx_list = [i for i in range(num_classes)]
    classes_idx_tensor = torch.tensor([classes_idx_list for _ in range(label.shape[0])])
    label_tensor = torch.tensor([[l for _ in range(num_classes)] for l in label])
    return (-(label_tensor - classes_idx_tensor) ** 2).float().softmax(dim=1)
