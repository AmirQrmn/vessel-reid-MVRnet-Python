import os
import glob
import torch
import torch.optim as optim
from data import Data
from model import PRNet
from utils import detect_latest_checkpoint, delete_older_checkpoints
from tensorboardX import SummaryWriter
from loss import Loss
from metrics import mean_ap, cmc, re_ranking
from tqdm import tqdm
from extract_feature import extract_feature
import numpy as np
from scipy.spatial.distance import cdist

# INPUTS
RESUME_TRAINING = False
TOTAL_EPOCHS = 500
LR_DECAY_EPOCHS = [300, 400]
WEIGHT_DECAY = 5e-4
CUDA_DEVICE_ID = 0
LR_INITIAL = 2e-4
MODEL_SAVE_FREQ = 50

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE_ID)

if os.name == "posix":
    root_dir_dataset = '../VCA-VR'
else:
    root_dir_dataset = '../VCA-VR/'

global current_iter

def evaluate(model, query_loader, test_loader, testset, queryset, writer):
    model.eval()

    print('extract features, this may take a few minutes')
    with torch.no_grad():
        qf = extract_feature(model, tqdm(query_loader)).numpy()
        gf = extract_feature(model, tqdm(test_loader)).numpy()

    def rank(dist):
        r = cmc(dist, queryset.ids, testset.ids,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, queryset.ids, testset.ids)

        return r, m_ap

    # re-rank
    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))
    dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    r, m_ap = rank(dist)
    print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
          .format(m_ap, r[0], r[2], r[4], r[9]))

    # no re-rank
    dist = cdist(qf, gf)

    r, m_ap = rank(dist)

    print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
          .format(m_ap, r[0], r[2], r[4], r[9]))
    writer.add_scalar('non-re-ranked top-1', r[0], current_iter)


def train_single_epoch(train_loader, model, loss, optimizer, writer):
    global current_iter
    model.train()

    for batch, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_val, loss_triplet, loss_xent = loss(outputs, labels)
        loss_val.backward()
        optimizer.step()
        current_iter += inputs.shape[0]
        writer.add_scalar('loss', loss_val, current_iter)
        writer.add_scalar('loss_triplet', loss_triplet, current_iter)
        writer.add_scalar('loss_xent', loss_xent, current_iter)


def run_train():
    global current_iter
    if RESUME_TRAINING is False:
        try:
            for f in glob.glob("./checkpoints/*"):
                os.remove(f)
            for f in glob.glob("./tensorboard_logs/*"):
                os.remove(f)
        except OSError:
            raise OSError("Unexpected exception trying to modify files in out dirs. Check rights.")

    current_iter = 0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    market_dataset = Data(parent_dir=root_dir_dataset)
    data_loader = market_dataset.train_loader
    model = PRNet(num_classes=365).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_INITIAL, weight_decay=WEIGHT_DECAY, amsgrad=False)
    loss = Loss(2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, LR_DECAY_EPOCHS)
    writer = SummaryWriter(log_dir='./tensorboard_logs')

    start_epoch = 1
    if RESUME_TRAINING:
        latest_ckpt_file = detect_latest_checkpoint('./checkpoints')
        checkpoint = torch.load(latest_ckpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        current_iter = checkpoint['current_iter']
        scheduler.step(start_epoch)
        model.train()
        print("Loaded model from {} with epoch {}".format(latest_ckpt_file, checkpoint['epoch']))

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        train_single_epoch(data_loader, model, loss, optimizer, writer)
        print("Done with epoch {}".format(epoch))
        scheduler.step()
        if epoch % MODEL_SAVE_FREQ == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_iter': current_iter,
            }, os.path.join('./checkpoints', 'epoch_%d.pth' % epoch))
            delete_older_checkpoints('./checkpoints')
            evaluate(model, market_dataset.query_loader, market_dataset.test_loader,
                     market_dataset.testset, market_dataset.queryset, writer)


if __name__ == '__main__':
    run_train()
