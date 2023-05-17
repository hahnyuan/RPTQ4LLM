import torch
from sklearn.cluster import KMeans
import numpy as np

oc_maxmin_dict = {}
ic_maxmin_dict = {}
DEBUG = False
oc_maxmin_dict_debug = {}


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

    if name not in oc_maxmin_dict:
        oc_maxmin_dict[name] = (xmax.detach_(), xmin.detach_())
    else:
        oc_maxmin_dict[name] = (
            torch.max(oc_maxmin_dict[name][0], xmax).detach_(),
            torch.min(oc_maxmin_dict[name][1], xmin).detach_(),
        )
        # oc_maxmin_dict[name] = oc_maxmin_dict[name][0]*0.99+xmax*0.01,oc_maxmin_dict[name][1]*0.99+xmin*0.01
    if DEBUG:
        if name not in oc_maxmin_dict_debug:
            oc_maxmin_dict_debug[name] = []
        oc_maxmin_dict_debug[name].append(o)
    

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

    if name not in ic_maxmin_dict:
        ic_maxmin_dict[name] = xmax.detach_(), xmin.detach_()
    else:
        ic_maxmin_dict[name] = (
            torch.max(ic_maxmin_dict[name][0], xmax).detach_(),
            torch.min(ic_maxmin_dict[name][1], xmin).detach_(),
        )
        # ic_maxmin_dict[name] = ic_maxmin_dict[name][0]*0.99+xmax*0.01,ic_maxmin_dict[name][1]*0.99+xmin*0.01


def qkt_imax_hook(m, i, o):
    name = m.name
    q = i[0]
    kt = i[1]

    bsz, n_heads, q_len, d = q.size()
    q = q.transpose(1, 2).view(-1, n_heads * d)
    xmax = torch.amax(q, [0])  # shape d
    xmin = torch.amin(q, [0])  # shape d
    qname=name + "_q"
    if (qname) not in ic_maxmin_dict:
        ic_maxmin_dict[qname] = xmax.detach_(), xmin.detach_()
    else:
        ic_maxmin_dict[qname] = (
            torch.max(ic_maxmin_dict[qname][0], xmax).detach_(),
            torch.min(ic_maxmin_dict[qname][1], xmin).detach_(),
        )

    bsz, n_heads, d, q_len = kt.size()
    xmax = torch.amax(kt.view(bsz, n_heads * d, q_len), [0, 2])  # shape d
    xmin = torch.amin(kt.view(bsz, n_heads * d, q_len), [0, 2])  # shape d
    kname=name + "_k"
    if (kname) not in ic_maxmin_dict:
        ic_maxmin_dict[kname] = xmax.detach_(), xmin.detach_()
    else:
        ic_maxmin_dict[kname] = (
            torch.max(ic_maxmin_dict[kname][0], xmax).detach_(),
            torch.min(ic_maxmin_dict[kname][1], xmin).detach_(),
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


def peg_tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None):
    """
    x shape [b,n,d]
    paper: Understanding and Overcoming the Challenges of Efficient Transformer Quantization
    """
    print("use peg to calc reorder")
    if n_heads is None:
        n_heads = 1

    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1)
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1)
        npdatamax = xmax.view(n_heads, -1, n)
        npdatamin = xmin.view(n_heads, -1, n)
        npdata = (npdatamax[:,:,0]-npdatamin[:,:,0]).reshape(n_heads,-1)
    else:
        npdatamax = xmax.view(n_heads, -1, 1)
        npdatamin = xmin.view(n_heads, -1, 1)
    # npdata = np.concatenate([npdatamax, npdatamin], -1)

        npdata = (npdatamax-npdatamin).reshape(n_heads,-1)

    cnt = 0
    all_index = []
    all_counts = []
    for i, data in enumerate(npdata):
        # for each head
        index=torch.argsort(data)
        counts=[len(data)//n_clusters]*n_clusters
        # kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(data)
        # counts = np.bincount(kmeans.labels_)
        # # "labels" refer to the assigned cluster membership of each data point in xmax.
        # labels = torch.from_numpy(kmeans.labels_).to(xmax.device)
        # index = torch.argsort(labels)
        index += cnt
        all_index.append(index)
        # breakpoint()
        all_counts.append(np.array(counts))
        cnt += len(data)
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts

# tensor_calc_reorder_index=peg_tensor_calc_reorder_index