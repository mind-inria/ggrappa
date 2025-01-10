import torch
import logging

from . import GRAPPAReconSpec
from .utils import pinv, get_src_tgs_blocks

logger = logger = logging.getLogger(__name__)


def estimate_grappa_kernel(acs,
                           af,
                           kernel_size=[4,4,5],
                           delta=0,
                           lambda_=1e-4,
                           cuda=False,
                           cuda_mode="estimation",
                           isGolfSparks=False,
                           quiet=False,
                           dtype=torch.complex64,
) -> GRAPPAReconSpec:

    if len(af) == 1:
        af = [af[0], 1]

    if isGolfSparks:
        if len(acs.shape) == 3:
            acs = acs[:, :, None, :]

        acs = acs.to(dtype)
    
    nc, acsny, acsnz, acsnx = acs.shape

    logger.debug("GRAPPA Kernel size: ", kernel_size)
    logger.debug("lambda: ", lambda_)

    if kernel_size:
        pat = torch.zeros([((k-1) * af[i]*[1, (1 if delta == 0 else af[1]//delta)][i==0] + 1) for i, k in enumerate(kernel_size[:2])])

    cnt=0
    for y in range(0, pat.shape[0], af[0]):
        pat[y,cnt::af[1]] = 1
        cnt = (cnt+delta)%af[1]

    tbly = af[0] * (1 if delta == 0 else af[1]//delta)
    tblz = af[1]
    tblx = 1

    sbly = min(pat.shape[0], acsny)
    sblz = min(pat.shape[1], acsnz)
    sblx = min(kernel_size[-1], acsnx)

    xpos = (sblx-1)//2
    ypos = (sbly-tbly)//2
    zpos = (sblz-tblz)//2

    idxs_src = (pat[:sbly, :sblz] == 1)
    idxs_src = idxs_src.unsqueeze(-1).expand(*idxs_src.size(), sblx)

    idxs_tgs = torch.zeros(sbly, sblz, sblx, dtype=torch.bool)
    idxs_tgs[ypos:ypos+tbly, zpos:zpos+tblz, xpos:xpos+1] = True

    nsp = idxs_src.sum()

    if not quiet:
        logger.info("GRAPPA Kernel estimation...")

    # Keep an eye on torch.nn.Unfold (and its functional) for 3D sliding block support
    # for future replacement.
    # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html
    blocks = acs.unfold(dimension=1, size=sbly, step=1)
    blocks = blocks.unfold(dimension=2, size=sblz, step=1)
    blocks = blocks.unfold(dimension=3, size=sblx, step=1)
    if isGolfSparks:
        src, tgs = get_src_tgs_blocks(blocks, idxs_src, idxs_tgs)
    else:
        src = blocks[..., idxs_src].reshape(nc, -1, nsp)
        tgs = blocks[..., idxs_tgs].reshape(nc, -1, idxs_tgs.sum())

    src = src.permute(1,0,-1).reshape(-1, nc*nsp)
    tgs = tgs.permute(1,0,-1).reshape(-1, nc*idxs_tgs.sum())

    src = src.cuda() if cuda and cuda_mode in ["all", "estimation"] else src
    tgs = tgs.cuda() if cuda and cuda_mode in ["all", "estimation"] else tgs

    grappa_kernel = pinv(src, lambda_) @ src.H @ tgs

    return GRAPPAReconSpec(weights=grappa_kernel,
                           af=af,
                           delta=delta,
                           pos=[ypos, zpos, xpos],
                           sbl=[sbly, sblz, sblx],
                           tbl=[tbly, tblz, tblx],
                           idxs_src=idxs_src)
