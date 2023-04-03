import torch
from sklearn.cluster import KMeans
import numpy as np

omax_dict = {}
imax_dict = {}
DEBUG = False
omax_dict_debug = {}


def layer_omax_hook(m, i, o):
    name = m.name
    if not isinstance(o, torch.Tensor):
        return
    if o.ndim == 3:
        xmax = torch.amax(o, [0, 1])  # shape d
        xmin = torch.amin(o, [0, 1])  # shape d
    elif o.ndim == 2:
        xmax = torch.amax(o, [0])  # shape d
        xmin = torch.amin(o, [0])  # shape d

    if name not in omax_dict:
        omax_dict[name] = (xmax.detach_(), xmin.detach_())
    else:
        omax_dict[name] = (
            torch.max(omax_dict[name][0], xmax).detach_(),
            torch.min(omax_dict[name][1], xmin).detach_(),
        )
        # omax_dict[name] = omax_dict[name][0]*0.99+xmax*0.01,omax_dict[name][1]*0.99+xmin*0.01
    if DEBUG:
        if name not in omax_dict_debug:
            omax_dict_debug[name] = []
        omax_dict_debug[name].append(o)


def layer_i0max_hook(m, i, o):
    name = m.name
    if len(i) == 0 or not isinstance(i[0], torch.Tensor):
        return
    if i[0].ndim == 3:
        xmax = torch.amax(i[0], [0, 1])  # shape d
        xmin = torch.amin(i[0], [0, 1])  # shape d
    elif i[0].ndim == 2:
        xmax = torch.amax(i[0], [0])  # shape d
        xmin = torch.amin(i[0], [0])  # shape d

    if name not in imax_dict:
        imax_dict[name] = xmax.detach_(), xmin.detach_()
    else:
        imax_dict[name] = (
            torch.max(imax_dict[name][0], xmax).detach_(),
            torch.min(imax_dict[name][1], xmin).detach_(),
        )
        # imax_dict[name] = imax_dict[name][0]*0.99+xmax*0.01,imax_dict[name][1]*0.99+xmin*0.01


def qkt_imax_hook(m, i, o):
    name = m.name
    q = i[0]
    kt = i[1]

    bsz, n_heads, q_len, d = q.size()
    q = q.transpose(1, 2).view(-1, n_heads * d)
    xmax = torch.amax(q, [0])  # shape d
    xmin = torch.amin(q, [0])  # shape d
    if name not in imax_dict:
        imax_dict[name + "_q"] = xmax.detach_(), xmin.detach_()
    else:
        imax_dict[name + "_q"] = (
            torch.max(imax_dict[name][0], xmax).detach_(),
            torch.min(imax_dict[name][1], xmin).detach_(),
        )

    bsz, n_heads, d, q_len = kt.size()
    xmax = torch.amax(kt.view(bsz, n_heads * d, q_len), [0, 2])  # shape d
    xmin = torch.amin(kt.view(bsz, n_heads * d, q_len), [0, 2])  # shape d
    if name not in imax_dict:
        imax_dict[name + "_k"] = xmax.detach_(), xmin.detach_()
    else:
        imax_dict[name + "_k"] = (
            torch.max(imax_dict[name][0], xmax).detach_(),
            torch.min(imax_dict[name][1], xmin).detach_(),
        )


def tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None):
    """
    x shape [b,n,d]
    """
    if n_heads is None:
        n_heads = 1

    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1)
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1)
        npdatamax = xmax.view(n_heads, -1, n).cpu().numpy()
        npdatamin = xmin.view(n_heads, -1, n).cpu().numpy()
    else:
        npdatamax = xmax.view(n_heads, -1, 1).cpu().numpy()
        npdatamin = xmin.view(n_heads, -1, 1).cpu().numpy()
    npdata = np.concatenate([npdatamax, npdatamin], -1)

    cnt = 0
    all_index = []
    all_counts = []
    for i, data in enumerate(npdata):
        # for each head
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(data)
        counts = np.bincount(kmeans.labels_)
        # "labels" refer to the assigned cluster membership of each data point in xmax.
        labels = torch.from_numpy(kmeans.labels_).to(xmax.device)
        index = torch.argsort(labels)
        index += cnt
        all_index.append(index)
        all_counts.append(counts)
        cnt += len(data)
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts
