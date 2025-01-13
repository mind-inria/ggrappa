import torch
import logging

from typing import Union

from . import GRAPPAReconSpec
from .estimation import estimate_grappa_kernel
from .application import apply_grappa_kernel
from .utils import extract_sampled_regions


logger = logging.getLogger(__name__)


def GRAPPA_Recon(
        sig: torch.Tensor,
        acs: torch.Tensor,
        af: Union[list[int], tuple[int, ...]],
        delta: int = 0,
        kernel_size: Union[list[int], tuple[int, ...]] = (4,4,5),
        lambda_: float = 1e-4,
        batch_size: int = 1,
        grappa_recon_spec: GRAPPAReconSpec = None,
        mask: torch.Tensor = None,
        isGolfSparks=False,
        cuda: bool = True,
        cuda_mode: str = "all",
        return_kernel: bool = False,
        quiet=False,

) -> torch.Tensor:
    """Performs GRAPPA reconstruction.

    Parameters
    ----------
    sig : torch.Tensor
        Complex 4D Tensor of shape: (nc, ky, kz, kx).
    acs : torch.Tensor
        Complex 4D Tensor of shape: (nc, acsky, acskz, acskx).
    af : Union[list[int], tuple[int, ...]]
        Acceleration factors. [afy, afz].
    delta : int, optional
        CAIPIRINHA shift. Default: `0`.
    kernel_size : Union[list[int], tuple[int, ...]], optional
        GRAPPA kernel size. Default `(4,4,5)`
    lambda_ : float, optional
        Regularization parameter of the pseudo-inverse. Default: `1e-4`
    batch_size : int, optional
        Size of the batch of `windows` to process by iteration in the kernel application phase. Default: `1`.
    grappa_kernel : torch.Tensor, optional
        GRAPPA kernel to be used. If `None`, the GRAPPA kernel weights will be computed. Default: `None`.
    mask : torch.Tensor, optional
        Binary mask for masked kernel application. Shape: (ky, kz, kx). Default: `None`
    cuda : bool, optional
        Whether to use GPU or not. Default: `True`.
    cuda_mode : str, optional
        CUDA operation mode (GPU):
            * "all" - Both kernel estimation and kernel application . Memory intensive.
            * "estimation" - Only use CUDA for GRAPPA kernel estimation.
            * "application" - Only use CUDA for GRAPPA kernel application.
        Default: `all`.
    quiet : bool, optional
        Enable printings and tqdm bars. Default: `True`.
    isGolfSparks : bool, optional
        Whether the input data is from the GoLF-SPARKLING sequence. Default: `False`.
    """
    if acs is None and grappa_recon_spec is None:
        acs = extract_sampled_regions(sig, acs_only=True)
    if grappa_recon_spec is None:
        grappa_recon_spec = estimate_grappa_kernel(acs,
                                                   af=af,
                                                   kernel_size=kernel_size,
                                                   delta=delta,
                                                   lambda_=lambda_,
                                                   cuda=cuda,
                                                   cuda_mode=cuda_mode,
                                                   isGolfSparks=isGolfSparks,
                                                   quiet=quiet)

    return apply_grappa_kernel(sig,
                               grappa_recon_spec,
                               batch_size=batch_size,
                               cuda=cuda,
                               cuda_mode=cuda_mode,
                               mask=mask,
                               isGolfSparks=isGolfSparks,
                               return_kernel=return_kernel,
                               quiet=quiet)