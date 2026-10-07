"""Graph WaveNet from the official repo, made runnable on current PyTorch without editing it.

Upstream builds gate/residual/skip layers as nn.Conv1d with (1, k) kernels and feeds them 4D
tensors. PyTorch 1.x expanded the 1-element stride/padding/dilation to 2D and ran conv2d; current
PyTorch raises instead. We swap each such layer for an nn.Conv2d with the same weight tensor
(shape [out, in, 1, k]) and dilation (1, d). Parameter names are unchanged, so state dicts load
both ways.
"""
import sys
from pathlib import Path

import torch.nn as nn

GWN_DIR = Path(__file__).resolve().parents[1] / "third_party" / "Graph-WaveNet"
sys.path.insert(0, str(GWN_DIR))
import model as gwn_model  # noqa: E402  (upstream model.py)

HEAD = ("end_conv_1", "end_conv_2")


def _conv1d_to_conv2d(m):
    c = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=1,
                  dilation=(1, m.dilation[0]), bias=m.bias is not None)
    assert m.padding == (0,) and m.stride == (1,) and m.groups == 1
    c.weight, c.bias = m.weight, m.bias
    return c


def fix_conv1d(net):
    for lst in (net.gate_convs, net.residual_convs, net.skip_convs):
        for i, m in enumerate(lst):
            if type(m) is nn.Conv1d:
                lst[i] = _conv1d_to_conv2d(m)
    return net


def patch_upstream():
    """Make every gwnet built afterwards (including inside upstream train.py/engine.py) runnable."""
    cls = gwn_model.gwnet
    if getattr(cls, "_patched", False):
        return
    orig = cls.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        fix_conv1d(self)

    cls.__init__, cls._patched = __init__, True


def build(device="cuda:0", ckpt=None, adj_pkl=None):
    """gwnet with the paper configuration (README flags), in eval() mode."""
    import torch
    import util  # upstream util.py
    patch_upstream()
    adj_pkl = adj_pkl or GWN_DIR.parents[1] / "data" / "sensor_graph" / "adj_mx.pkl"
    _, _, adj = util.load_adj(str(adj_pkl), "doubletransition")
    supports = [torch.tensor(a).to(device) for a in adj]
    # same arguments as engine.trainer with train.py defaults and --randomadj (aptinit=None)
    net = gwn_model.gwnet(device, 207, 0.3, supports=supports, gcn_bool=True, addaptadj=True,
                          aptinit=None, in_dim=2, out_dim=12, residual_channels=32,
                          dilation_channels=32, skip_channels=256, end_channels=512)
    net.to(device)
    if ckpt:
        net.load_state_dict(torch.load(ckpt, map_location=device))
    return net.eval()


def head_params(net):
    return [p for n, p in net.named_parameters() if n.split(".")[0] in HEAD]


if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    patch_upstream()
    net = gwn_model.gwnet("cpu", 207, supports=None, aptinit=None).eval()
    x = torch.randn(2, 2, 207, 13)
    y = net(x)
    assert y.shape == (2, 12, 207, 1), y.shape
    # reference: what PyTorch 1.x computed for a Conv1d with (1, k) kernel = conv2d with dilation (d, d)
    m = nn.Conv1d(32, 32, kernel_size=(1, 2), dilation=4)
    h = torch.randn(2, 32, 207, 13)
    ref = nn.functional.conv2d(h, m.weight, m.bias, dilation=(4, 4))
    assert torch.allclose(_conv1d_to_conv2d(m)(h), ref)
    print("ok, head params:", sum(p.numel() for p in head_params(net)))
