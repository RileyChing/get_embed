from pathlib import Path
import torch
from tqdm import tqdm
from pif.hvp_grad import grad_z
from pif.utils import save_json
from torch.autograd import grad
from collections import OrderedDict
from torch.nn import functional as F, CrossEntropyLoss
import time
import math


def concat_along_channel_dim(x1, x2):
    assert x1.ndim == 4
    assert x2.ndim == 4
    assert x1.shape[0] == x2.shape[0]

    if x1.shape[2] < x2.shape[2]:
        x1 = F.pad(x1, (0, 0, 0, x2.shape[2] - x1.shape[2]))
    if x1.shape[3] < x2.shape[3]:
        x1 = F.pad(x1, (0, x2.shape[3] - x1.shape[3], 0, 0))
    if x1.shape[2] > x2.shape[2]:
        x2 = F.pad(x2, (0, 0, 0, x1.shape[2] - x2.shape[2]))
    if x1.shape[3] > x2.shape[3]:
        x2 = F.pad(x2, (0, x1.shape[3] - x2.shape[3], 0, 0))

    # x = torch.cat([x1, x2], dim=1)
    return x1, x2

def calc_all_grad(config, model, train_loader, test_loader,
                  ntest_start, ntest_end, mode='TC'):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    '''
    config['outdir'] specifies the model name, i.e. different model saves at different out dir with unique name/id/

    '''
    depth, r = config['recursion_depth'], config['r_averaging']

    outdir = Path(config["outdir"])

    # breakpoint()
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    influence_results = {}

    ntrainiter = len(train_loader.dataset)
    ntest_end = len(test_loader.dataset)
    model.eval()
    grad_z_test = ()
    # for i in tqdm(range(ntest_start, ntest_end)):
    for i, batch in enumerate(tqdm(test_loader)):

        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
        else:
            device = torch.device("cpu")

        input_ids, token_type_ids, lm_labels = batch
        # for i in range(len(lm_labels[0])):
        #     if(lm_labels[0,i]<0):
        #         lm_labels[0,i] = 0

        input_ids, token_type_ids, lm_labels = \
            input_ids.to(device), token_type_ids.to(device), lm_labels.to(device)
        di, dt, dl, idx, dd = test_loader.dataset[i]

        # for i in range(38):
        #    if(dl[i]<0):
        #        dl[i] = 0

        idx = int(idx)

        if outdir.joinpath(f'did-{idx}.{mode}.json').exists():
            continue

        if mode == 'TC':

            # grad_z_test = grad_z(input_ids, token_type_ids, lm_labels, model)
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                labels=lm_labels
            )
            # ebd_test = outputs[2][0][0]#+outputs[2][0][1]
            ebd_test = outputs[0]

            loss, logits = outputs[0], outputs[1]

            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = lm_labels[..., 1:].contiguous().to(device)
            # pad_id = -100
            # loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
            # siz = shift_logits.size(-1)
            # log = shift_logits.view(-1, siz)
            # lab = shift_labels.view(-1)
            # loss = loss_fct(log, lab)

            # start = time.clock()
            grad_z_test = grad(loss, model.parameters())  # , allow_unused=True)
            # end1 = time.clock()
            # print(end1-start)
            # grad_z_test = pick_gradient(grad_z_test, model)  # pick effective ones
            # end2 = time.clock()
            # print(end2-end1)

        if mode == 'IF':
            s_test = torch.load(config['stest_path'] + f"/did-{int(idx)}_recdep{depth}_r{r}.s_test")
            s_test = [s_t.cuda() for s_t in s_test]

        train_influences = {}

        for j, batch_t in enumerate(train_loader):  # in tqdm(range(ntrainiter)):

            input_ids_t, token_type_ids_t, lm_labels_t = batch_t
            input_ids_t, token_type_ids_t, lm_labels_t = \
                input_ids_t.to(device), token_type_ids_t.to(device), lm_labels_t.to(device)

            # for i in range(len(lm_labels_t[0])):
            #     if (lm_labels_t[0, i] < 0):
            #         lm_labels_t[0, i] = 0

            ti, tt, tl, t_idx, td = train_loader.dataset[j]
            t_idx = int(t_idx)
            outputs = model(
                input_ids=input_ids_t,
                token_type_ids=token_type_ids_t,
                labels=lm_labels_t
            )

            ebd_train = outputs[0]#+outputs[2][0][1]
            loss_t, logits = outputs[0], outputs[1]
            # grad_z_train = grad_z(input_ids_t, token_type_ids_t, lm_labels_t, model, device, pid)
            grad_z_train = grad(loss_t, model.parameters())  # , allow_unused=True)
            # grad_z_train = pick_gradient(grad_z_train, model)  # pick effective ones

            score = 0
            # score = cosine_similarity(ebd_train, ebd_test)
            # sum1 = sum(score)
            # avg = (sum1 / (float)(len(score)))
            if mode == 'IF':
                score = param_vec_dot_product(s_test, ebd_train)#grad_z_train)
            elif mode == 'TC':
                # x1, x2 = concat_along_channel_dim(ebd_test, ebd_train)
                score = param_vec_dot_product(grad_z_test, grad_z_train)
            # if mode == 'TC':
                # print(cos_sim(ebd_train, ebd_test))

                # score = cos_similar(ebd_train, ebd_test)#[3][3]
                # sum1 = sum(score)
                # avg = (sum1 / (float)(len(score)))


                # last1 = len(score[:,:,:,:])
                # print(last1)
            # breakpoint()

            if t_idx not in train_influences:
                train_influences[t_idx] = {'train_dat': (td),
                                          'if': float(score)}

        # train_influences1 = {}
        # train_influences1 = OrderedDict(sorted(train_influences, key=lambda x: x[1]['if'], reverse=True))#, reverse=True
        if idx not in influence_results:
            influence_results[idx] = {'test_dat': (dd),
                                      'ifs': train_influences}
        save_json(influence_results, outdir.joinpath(f'did-{idx}.{mode}.json'))


def param_vec_dot_product(a, b):
    """ dot product between two lists"""
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])
    # breakpoint()


def pick_gradient(grads, model):
    """
    pick the gradients by name.
    Specifically for BERTs, it extracts 10, 11 layer, pooler and classification layers params.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]
            # if 'layer.10.' in n or 'layer.11.' in n
            # or 'classifier.' in n or 'pooler.' in n


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))

def cosine_similarity(vec_a, vec_b):

    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)

def cos_similar(p, q):

    m = p.shape[3]
    n = q.shape[3]

    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    # return sim_matrix.item()
    # sim_matrix = sim_matrix.view(12, m, n)

    list = []
    #start = time.perf_counter()
    for i in range(1):
        for j in range(12):
            for x in range(sim_matrix.shape[2]):
                for y in range(sim_matrix.shape[3]):
                    list.append(float(sim_matrix[i, j, x, y]))

    #end = time.perf_counter()
    #print(end-start)

    return list

import torch
import torch.nn.functional as F

def cos_sim(feature1, feature2):
    # ??????feature1???N*C*W*H??? feature2??????N*C*W*H
    feature1 = feature1.view(feature1.shape[0], -1)  # ??????????????????N*(C*W*H)????????????
    feature2 = feature2.view(feature2.shape[0], -1)
    feature1 = F.normalize(feature1)  # F.normalize??????????????????????????????L2?????????
    feature2 = F.normalize(feature2)
    feature2 = feature2.t()
    distance = feature2.mm(feature1)
    print(distance)

    return distance

# a = torch.rand((64, 23, 50, 200))
# print(a.shape[3])
# list=[]
# # for  i in range(len(a)):
# #     for j in range(len(a[0])):
# #         for k in range(len(a[]))
#
# for i in range(2):
#     for j in range(2):
#         list.append(float(a[0,0,i,j]))
# # print(len(list))
# # b = torch.rand((64, 23, 60, 200))
