from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from TripletLoss import TripletLoss


class Loss(loss._Loss):
    def __init__(self, num_triplet_losses):
        super(Loss, self).__init__()
        self.num_triplet_losses = num_triplet_losses

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[0:self.num_triplet_losses]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[self.num_triplet_losses:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss
        loss_triplet = Triplet_Loss.data.cpu().numpy()
        loss_xent = CrossEntropy_Loss.data.cpu().numpy()

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            loss_triplet,
            loss_xent), end=' ')
        return loss_sum, loss_triplet, loss_xent
