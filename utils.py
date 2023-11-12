import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# TODO: I think the mean calculation in this function is wrong
# Each row should be divided by the number of positive events
# not the total number of events (which is what I think is happening here)
def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    return loss_cox

def update_optim(optim, epoch, lr_decay):
    '''
    Updates optimizer as proposed in: 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5828433/
    '''
    cur_lr = optim.param_groups[0]['lr']
    new_lr = cur_lr / (1 + (epoch*lr_decay))
    optim.param_groups[0]['lr'] = new_lr

def save_graph(title, dst, x_label, y_label, list1, list1_label, list2=None, list2_label=None):
    plt.figure(figsize=(10, 6)) 
    plt.title(title)
    plt.plot(list1, label=list1_label)

    if list2 is not None:
        plt.plot(list2, label=list2_label)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    plt.savefig(os.path.join(dst, f'{title}.png'))
    plt.close()
