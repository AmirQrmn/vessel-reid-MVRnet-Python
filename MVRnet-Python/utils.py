import re
import glob
import os
import errno


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(sim_pos, sim_neg):
    pred = (sim_pos - sim_neg).cpu().data
    return (pred < 0).sum().item()*1.0/sim_pos.size()[0]


def detect_latest_checkpoint(path):
    regexp = re.compile('epoch_(\d+).pth')
    all_files = glob.glob(os.path.join(path, '*'))
    all_epochs = [int(regexp.match(os.path.basename(file_path)).groups()[0]) for file_path in all_files]
    latest_full_path = all_files[all_epochs.index(max(all_epochs))]
    return latest_full_path


def delete_older_checkpoints(path):
    regexp = re.compile('epoch_(\d+).pth')
    all_files = glob.glob(os.path.join(path, '*'))
    all_epochs = [int(regexp.match(os.path.basename(file_path)).groups()[0]) for file_path in all_files]
    _ = [os.remove(file) for idx, file in enumerate(all_files) if idx != all_epochs.index(max(all_epochs))]
    return

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
        return dir_path
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
