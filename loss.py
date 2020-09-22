import torch.nn.functional as F

def binary_cross_entropy(output, target):
    
    # Align the time_steps of output and target
    N = min(output.shape[1], target.shape[1])

    out = F.binary_cross_entropy(
        output[:, 0: N, :],
        target[:, 0: N, :]
    )

    return out