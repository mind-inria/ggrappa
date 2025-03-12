import torch
import numpy as np
import scipy as sp


def rss(data, axis=0):
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))


def get_indices_from_mask(mask):
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()
    nonzero_indices = np.nonzero(mask)

    min_indices = np.min(nonzero_indices, axis=1)
    max_indices = np.max(nonzero_indices, axis=1)

    cube_dimensions = max_indices - min_indices + 1

    return min_indices, cube_dimensions

def get_src_tgs_blocks(blocks, idxs_src, idxs_tgs, check_type='acs'):
    """Extracts source and target blocks from the given blocks tensor based on specified indices.
    
    Parameters
    ----------
    blocks : torch.Tensor
        A tensor containing the blocks from which to extract source and target blocks.
    idxs_src : torch.Tensor
        A tensor containing the indices of the source blocks to be extracted.
    idxs_tgs : torch.Tensor
        A tensor containing the indices of the target blocks to be extracted.
    check_type : str, optional
        The type of check to perform when selecting blocks. Options are 'acs' and 'all_sampled_srcs'.
        Default is 'acs'.
        If 'acs', the function will select blocks where all samples are present.
        If 'all_sampled_srcs', the function will select blocks where all samples are present in the source locations.
        
    Returns
    -------
    select_blocks_src : torch.Tensor
        A tensor containing the selected source blocks.
    select_blocks_tgs : torch.Tensor
        A tensor containing the selected target blocks.
    """
    
    if check_type == 'acs':
        samples_per_block = (blocks.sum(dim=0)!=0).sum(dim=(-3, -2, -1))
        locy, locz, locx = torch.nonzero(samples_per_block == idxs_src.numel(), as_tuple=True)
    elif check_type == 'all_sampled_srcs':
        #TODO: Test if this works
        srcs_per_block = blocks[..., idxs_src].sum(dim=-1)
        tgs_per_block = blocks[..., idxs_tgs].sum(dim=-1)
        locy, locz, locx = torch.nonzero(
            (srcs_per_block.abs() == torch.sum(idxs_src)) and (tgs_per_block.abs() < torch.sum(idxs_tgs)),
            as_tuple=True,
        )
    select_blocks = blocks[:, locy, locz, locx]
    return select_blocks[..., idxs_src], select_blocks[..., idxs_tgs]

def get_grappa_filled_data_and_loc(sig, rec, params):
    #rec[:, np.abs(sig).sum(axis=0)!=0] = 0
    sampled_mask = np.abs(rec).sum(axis=0) != 0
    extra_data = rec[:, sampled_mask]
    rec_loc = np.nonzero(sampled_mask)
    rec_loc = np.asarray(rec_loc).T
    extra_loc = rec_loc / params['img_size'] - 0.5
    return extra_loc, extra_data
     

def get_cart_portion_sparkling(kspace_shots, traj_params, kspace_data, calc_osf_buffer=10):
    """Extracts and resamples the Cartesian portion of k-space data from the given k-space shots.

    Parameters
    ----------
    kspace_shots : numpy.ndarray
        The k-space trajectory shots.
    traj_params : dict
        Dictionary containing trajectory parameters, including 'img_size'.
    kspace_data : numpy.ndarray
        The k-space data corresponding to the shots.
    calc_osf_buffer:  float, optional
        The oversampling factor for resampling, by default 1.

    Returns
    -------
    numpy.ndarray
        The gridded k-space data with the Cartesian portion resampled and placed in the appropriate locations.
    """
    grads = np.diff(kspace_shots, axis=1)
    re_kspace_data = kspace_data.reshape(kspace_data.shape[0], *kspace_shots.shape[:2])
    mask = grads[..., 1] == 0
    pad_mask = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    mask = np.diff(pad_mask*1)
    starts = np.argwhere(mask == 1)
    ends = np.argwhere(mask == -1)
    osf = 1/np.mean(np.diff(kspace_shots[kspace_shots.shape[0]//2, kspace_shots.shape[1]//2-calc_osf_buffer:kspace_shots.shape[1]//2+calc_osf_buffer, 0])*traj_params['img_size'][0])
    max_length = np.zeros(grads.shape[0]) + osf # To ensure we have atleast one point after resampling
    locs = np.ones((grads.shape[0], 2))*-1
    sampled_loc = [[],] * grads.shape[0]
    cart_loc = [[],] * grads.shape[0]
    new_kspace_data = []
    new_kspace_loc = []
    gridded_data = np.zeros((kspace_data.shape[0], *traj_params['img_size']), dtype=np.complex64)
    for start, end in zip(starts, ends):
        row, start_col = start
        _, end_col = end
        length = end_col - start_col
        if length > max_length[row]:
            max_length[row] = length
            locs[row, 0] = start_col
            locs[row, 1] = end_col
            cart_loc[row] = np.copy(kspace_shots[row, start_col:end_col])
            sampled_loc[row] = [start_col, end_col]
    for row, (locs, s_loc) in enumerate(zip(cart_loc, sampled_loc)):
        if not len(locs) or int(
                (s_loc[1]-s_loc[0])*
                np.diff(kspace_shots[row, s_loc[0]:s_loc[0]+2, 0])*traj_params['img_size'][0]
            ) == 0:
            new_kspace_data.append(re_kspace_data[:, row])
            new_kspace_loc.append(kspace_shots[row, ...])
            continue
        data = sp.signal.resample(
            re_kspace_data[:, row, s_loc[0]: s_loc[1]],
            int(
                (s_loc[1]-s_loc[0])*
                np.diff(kspace_shots[row, s_loc[0]:s_loc[0]+2, 0])*traj_params['img_size'][0]
            ),
            axis=-1,
        )
        new_kspace_data.append(np.hstack([
            re_kspace_data[:, row, :s_loc[0]],
            re_kspace_data[:, row, s_loc[1]:]
        ]))
        new_kspace_loc.append(np.concatenate([
            kspace_shots[row, :s_loc[0]],
            kspace_shots[row, s_loc[1]:],
        ], axis=0))
        locs += 0.5
        locs *= np.asarray(traj_params["img_size"]).T
        rounded_locs = locs.round(0).astype('int')
        gridded_data[:, rounded_locs[0][0]:rounded_locs[0][0]+len(data[0]), rounded_locs[0][1], rounded_locs[0][2]] = data
    return gridded_data, np.hstack(new_kspace_data), np.concatenate(new_kspace_loc, axis=0)
    
def pad_back_to_size(sig, vol_shape, start_loc, end_loc):
    """Pads a given signal tensor back to a specified volume shape.
    
    Parameters
    ----------
    sig : torch.Tensor
        The input signal tensor to be padded.
    vol_shape : tuple of int
        The shape of the volume to pad the signal tensor to (ky, kz, kx).
    start_loc : tuple of int
        The starting location (ky, kz, kx) for padding.
    end_loc : tuple of int
        The ending location (ky, kz, kx) for padding.
    
    Returns
    -------
    torch.Tensor
        The padded signal tensor with the specified volume shape.
    """
    ky, kz, kx = vol_shape
    start_ky = ky // 2
    start_kz = kz // 2
    start_kx = kx // 2
    rec = torch.zeros((sig.shape[0], *vol_shape), dtype=sig.dtype)
    rec[:,   start_ky-start_loc[0]+1:start_ky+end_loc[0]-1,
                   start_kz-start_loc[1]+1:start_kz+end_loc[1]-1,
                   start_kx-start_loc[2]+1:start_kx+end_loc[2]-1] = sig
    return rec

def extract_sampled_regions(sig, acs_only=True):
    """Extracts the Auto-Calibration Signal (ACS) region from the input signal.
    This is a generic function which finds all the regions that is continuously sampled 
    in the center of k-space.

    Parameters
    ----------
    sig : torch.Tensor
        A 4D tensor with shape (coils, ky, kz, kx) representing the input signal.
    acs_only : bool, optional
        If True, the function will return the ACS region only. Default: True.
        If False, the function will return a region within which sampling starts.

    Returns
    -------
    torch.Tensor
        A 4D tensor containing the extracted ACS region from the input signal.
    """
    _, ky, kz, kx = sig.shape

    start_ky = ky // 2
    start_kz = kz // 2
    start_kx = kx // 2

    torch_fn = torch.min
    if acs_only:
        sig_ = (abs(sig) == 0)
    else:
        sig_ = (abs(sig) != 0)
        torch_fn = torch.max


    left_start_ky = torch_fn(torch.nonzero(sig_[0, start_ky:, start_kz, start_kx], as_tuple=False)).item()
    left_end_ky = torch_fn(torch.nonzero(sig_[0, :start_ky+1, start_kz, start_kx].flip(0), as_tuple=False)).item()

    left_start_kz = torch_fn(torch.nonzero(sig_[0, start_ky, start_kz:, start_kx], as_tuple=False)).item()
    left_end_kz = torch_fn(torch.nonzero(sig_[0, start_ky, :start_kz+1, start_kx].flip(0), as_tuple=False)).item()

    left_start_kx = torch_fn(torch.nonzero(sig_[0, start_ky, start_kz, start_kx:], as_tuple=False)).item()
    left_end_kx = torch_fn(torch.nonzero(sig_[0, start_ky, start_kz, :start_kx+1].flip(0), as_tuple=False)).item()

    center = sig[:,   start_ky-left_start_ky+1:start_ky+left_end_ky-1,
                    start_kz-left_start_kz+1:start_kz+left_end_kz-1,
                    start_kx-left_start_kx+1:start_kx+left_end_kx-1]
    if acs_only:
        return center
    start_loc = (left_start_ky, left_start_kz, left_start_kx)
    end_loc = (left_end_ky, left_end_kz, left_end_kx)
    return center, start_loc, end_loc
    

def pinv_batch(M, lambda_=1e-4, cuda=True):
    if cuda: M = M.cuda()
    MM = M.H @ M
    del M
    #torch.cuda.empty_cache()

    # Might also consider Power Iteration methods to speedup the process for large M matrix?
    S = torch.linalg.eigvalsh(MM)[-1].item()

    MM = MM.cpu()
    regularizer = (lambda_**2) * abs(S) * torch.eye(MM.shape[0], device=MM.device)
    del S
    #torch.cuda.empty_cache()
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    del MM
    return reg_pinv

def pinv(M, lambda_=1e-4):
    if M.shape[0] > M.shape[1]:
        MM = M.H @ M
        finalTranspose = False
    else:
        MM = M @ M.H
        finalTranspose = True
    S = torch.linalg.eigvalsh(MM)[-1].item()
    regularizer = (lambda_**2) * abs(S) * torch.eye(MM.shape[0], device=M.device)
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    return reg_pinv.H if finalTranspose else reg_pinv


def pinv_linalg_batch(A, lamdba_=1e-4, cuda=True):
    if cuda: A = A.cuda()
    AA = A.H@A
    del A
    #torch.cuda.empty_cache()
    S = torch.linalg.eigvalsh(AA)[-1].item() # Largest eigenvalue
    lambda_sq = (lamdba_**2) * abs(S)
    del S
    #torch.cuda.empty_cache()
    I = torch.eye(AA.shape[0], dtype=AA.dtype, device=AA.device)
    regularized_matrix = AA + I * lambda_sq
    del I, AA, lambda_sq
    #torch.cuda.empty_cache()
    A = A.cpu()
    regularized_matrix = regularized_matrix.cpu()
    return torch.linalg.solve(regularized_matrix, A.H)


def pinv_linalg(A, lamdba_=1e-4):
    m,n = A.shape
    if n > m:
        AA = A@A.H
    else:
        AA = A.H@A
    S = torch.linalg.eigvalsh(AA)[-1].item()
    lambda_sq = (lamdba_**2) * abs(S)

    I = torch.eye(AA.shape[0], dtype=A.dtype, device=A.device)

    regularized_matrix = AA + I * lambda_sq

    return torch.linalg.solve(regularized_matrix, A.H)
