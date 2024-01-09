import torch

def OrthogonalityLoss(src, tgts):
    loss = torch.zeros([1])
    for tgt in tgts:
        for (sv, tv) in zip(src, tgt):
            loss += torch.dot(src, tgt)

    return loss