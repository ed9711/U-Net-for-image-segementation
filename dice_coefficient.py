import torch
from torch.autograd import Function


class DiceCoefficient(Function):

    @staticmethod
    def forward(ctx, input, target):
        ctx.save_for_backward(input, target)
        eps = 0.0001
        # print(np.array(input.view(-1).cpu()).shape, np.array(target.view(-1).cpu().float()).shape)
        ctx.inter = torch.dot(input.view(-1), target.view(-1).float())
        ctx.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * ctx.inter.float() + eps) / ctx.union.float()
        return t.float()

    @staticmethod
    def backward(ctx, grad_output):

        input, target = ctx.saved_variables
        grad_input = grad_target = None

        if ctx.needs_input_grad[0]:
            grad_input = (grad_output * 2 * (target * ctx.union - ctx.inter) \
                          / (ctx.union * ctx.union))  # .float()
        if ctx.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        loss = DiceCoefficient()
        s = s + loss.apply(c[0], c[1])

    return s / (i + 1)
