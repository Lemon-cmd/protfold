import sys, time, argparse, matplotlib.pyplot as plt
import torch, einops, sidechainnet as scn, numpy as np
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F

from math import pi
from tqdm import tqdm
from collections import defaultdict

from os.path import exists
from einops import rearrange
from einops.layers.torch import Rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_seq_features(batch):
    '''
        prot-ids
        seq-lengths
        one-hot seqs
        evos/pssm infos
        angs: phi, psi
        maskings
    '''

    pids = batch.pids
    seqs, evos, angs, msks = batch.seqs, batch.evos, batch.angs[:, :, 0:2], batch.msks

    # compute distance map (l2) : we only care about C-beta atom
    m, l, _ = seqs.shape
    cb_crds = batch.crds.reshape(m, 14, l, -1)[:, 3]
    dmaps = torch.cdist(cb_crds, cb_crds, p = 2)

    # compute contact map : any distance > 8A is 0
    cmaps = dmaps
    cmaps[cmaps > 8] = 0.0

    # compute masks
    msks = msks[:, :, None] * msks[:, None, :]

    return seqs, evos, angs, dmaps, msks, cmaps, pids, batch.lengths


class CropsDataset(torch.utils.data.IterableDataset):
    def __init__(self, seqs, evos, angs, dmaps, msks, cmaps, pids,
                 crop_sz=64, stride_sz=32, bins_sz=64, angs_sz=1295, batch_sz=32, test=False) -> None:

        self._test, self._batch_sz = test, batch_sz
        self.seqs, self.evos, self.angs = seqs, evos, angs
        self.msks, self.dmaps, self.cmaps, self.pids = msks, dmaps, cmaps, pids
        self._crop_sz, self._stride_sz, self._bins_sz = crop_sz, stride_sz, bins_sz

        # create distance bins
        bin_w = (22 - 2) / self._bins_sz
        self.dist_bins = torch.arange(2 + bin_w, 22, bin_w)

        from math import pi
        self.angs_bins = torch.arange(-pi, pi, 2 * pi / angs_sz)

        # if test set, then create place holders for the entire NxN contact map
        if self._test:
            B, L, _ = self.seqs.shape
            self.cnts = torch.zeros((B, L, L))
            self.cmaps_pred = torch.zeros((B, self._bins_sz, L, L))

        L = self.seqs.size(1)
        fn = Rearrange('b (h w) d -> b d h w', h = L)

        self.seqs = fn(self.seqs.repeat(1, L, 1)).float()
        self.evos = fn(self.evos.repeat(1, L, 1)).float()
        self.angs = fn(self.angs.repeat(1, L, 1)).float()
        
        if self.seqs.size(1) % self._crop_sz > 0:
            self._pad_tensors()
        
        self._start_pos = self._get_start_pos()

        self.angs_bins = torch.searchsorted(self.angs_bins, self.angs.contiguous())
        self._dmap_bins = torch.searchsorted(self.dist_bins, self.dmaps.contiguous())

    def _pad_img(self, x, crop_sz, val=None) -> torch.Tensor:
        '''pad square tensors'''
        m, c, h, w = x.shape
        assert h == w, 'tensor size must be squared.'

        val = val if val != None else 0

        d = w + crop_sz
        d = d + (crop_sz - d % crop_sz) - w

        # pad width
        x = F.pad(x, (0, d), 'constant', val)

        # pad height
        x = F.pad(x, (0, 0, 0, d), 'constant', val)

        return x

    def _pad_tensors(self) -> None:
        '''pad all tensors by pad_sz in each relevant dim'''
        self.seqs = self._pad_img(self.seqs, self._crop_sz)
        self.evos = self._pad_img(self.evos, self._crop_sz)
        self.angs = self._pad_img(self.angs, self._crop_sz)

        self.cmaps = self._pad_img(self.cmaps.unsqueeze(1),
                                   self._crop_sz).squeeze(1)

        self.msks = self._pad_img(self.msks.unsqueeze(1),
                                  self._crop_sz).squeeze(1)

        self.dmaps = self._pad_img(self.dmaps.unsqueeze(1),
                                   self._crop_sz, val=self._crop_sz).squeeze(1)

    def _get_start_pos(self) -> list:
        '''Create a set of start positions si, sj for the batch

           During training, si is chosen in (0, pad_sz) with sj > si,
              followed by strides in both si and sj directions

           During testing, si, sj = 0, 1
        '''
        if self._test:
            si, sj = 0, 1
        else:
            si = np.random.randint(0, 32)
            sj = np.random.randint(si + 1, si + 32)

        N = self.seqs.shape[2]

        # now generate other start pos si, and sj using strides
        PI = np.arange(si, N - self._crop_sz + 1, self._stride_sz)
        PJ = np.arange(sj, N - self._crop_sz + 1, self._stride_sz)

        # list of all si, sj pairs
        return [(i,j) for i in PI for j in PJ if i < j]

    def add_pred_crop(self, crop_preds, ij_pos):
        assert(self._test)

        '''Used during testing to convert the predicted output from
        each crop into probabilties for the 64 distance bins, and
        adding this to the appropriate "tile" in the predicted cmap
        '''

        si, sj = ij_pos
        L = self.cnts.size(1)
        B, nbins, n, n = crop_preds.shape
        
        if sj > L or si > L:
            return
        
        n1 = n2 = 0
        sin = si + n
        sjn = sj + n
        
        if si == L:
            n1 = 1
            sin = si + n1
        elif sin > L:
            n1 = L - si
            sin = si + n1
         
        if sj == L:
            n2 = 1
            sjn = sj + n2
        elif sjn > L:
            n2 = L - sj
            sjn = sj + n2
        
        n1 = n if n1 == 0 else n1
        n2 = n if n2 == 0 else n2
        
        # compute softmax along the distance bins axis: dim 1
        predicted_probs = F.softmax(crop_preds.permute(0, 2, 3, 1), dim=-1)
        
        # update the cmaps info and counts
        self.cnts[:, si:sin, sj:sjn] += torch.ones(B, n1, n2)
        self.cmaps_pred[:, :, si:sin, sj:sjn] += predicted_probs[:, :, : n1, : n2].cpu()

    def get_cmap_data(self):
        '''return cmap/cnt info during testing'''
        return self.cmaps_pred, self.cnts, self.cmaps

    def get_dmat_data(self):
        return self._dmap_bins, self.msks

    def get_dist_bins(self):
        '''return the distance bins'''
        return self.dist_bins

    def __len__(self):
        '''how many si, sj pairs are there?'''
        return len(self._start_pos)

    def __iter__(self):
        '''return one crop starting at si, sj *for all* sequences in a batch
           so the crop will the B x f x 64 x 64, with features in the 2nd dim
        '''
        for si, sj in self._start_pos:
            seq_crops = self.seqs[:, :, si:si+self._crop_sz, sj:sj+self._crop_sz]
            evo_crops = self.evos[:, :, si:si+self._crop_sz, sj:sj+self._crop_sz]

            feats = torch.cat([seq_crops, evo_crops], dim = 1)

            msk_crops = self.msks[:, si:si+self._crop_sz, sj:sj+self._crop_sz]
            dmap_crops = self._dmap_bins[:, si:si+self._crop_sz, sj:sj+self._crop_sz]
            ang_crops = self.angs_bins[:, :, si:si+self._crop_sz, sj:sj+self._crop_sz]

            feats = torch.split(feats, self._batch_sz, 0)
            angs = torch.split(ang_crops, self._batch_sz, 0)
            msks = torch.split(msk_crops, self._batch_sz, 0)
            dmaps = torch.split(dmap_crops, self._batch_sz, 0)

            for i in range(len(feats)):
                yield(feats[i], angs[i], dmaps[i], msks[i], (si, sj))


def reload_model(model_type, par_state_dict):
    from collections import OrderedDict

    state_dict = OrderedDict()
    for k, v in par_state_dict.items():
        # remove module
        name = k[7:]

        #if name[:name.find('.')] == 'mlp_head':
            #v = v.fill_(0.0)

        state_dict[name] = v

    model_type.load_state_dict(state_dict)
    return model_type


def save_checkpoint(fname, bidx, e, running_loss, model, optimizer):
    checkpoint = {
        'batch': bidx,
        'epoch': e,
        'loss': running_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, fname)


def load_checkpoint(fname, model, optimizer, device):
    checkpoint = torch.load(fname)
    model = reload_model(model, checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    running_loss = checkpoint['loss']
    bidx = checkpoint['batch']
    e = checkpoint['epoch']

    return e, bidx, running_loss, model, optimizer


def train(model, data, crop_sz, stride_sz, pad_sz, bins_sz, batch_sz, fname="", epochs=10):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-6, amsgrad=True)

    last_epoch, loss = 0, 0
    if exists(fname) and fname != "":
        last_epoch, last_bidx, loss, model, optimizer = load_checkpoint(fname, model, optimizer, device)

    model = nn.DataParallel(model)
    
    model.train()
    for e in range(1, epochs + 1):
        running_loss, running_len = 0, 1

        trainer = tqdm(enumerate(data['train']), total=len(data['train']), unit='batch')
        trainer.set_description(f'Epoch {e}')

        for bidx, batch in trainer:
            seqs, evos, angs, dmaps, msks, cmaps, pids, lengths = get_seq_features(batch)

            trainset = CropsDataset(
                seqs, evos, angs, dmaps, msks, cmaps, pids,
                crop_sz=crop_sz, stride_sz=stride_sz, bins_sz=bins_sz, angs_sz=1295,
                batch_sz=torch.cuda.device_count() * batch_sz, test=False
            )

            for cidx, (feats, angs, dmaps, dmsks, ij_pos) in enumerate(trainset):
                feats = feats.to(device)
                dmaps, dmsks = dmaps.to(device), dmsks.to(device)

                dmaps_pred = model(feats)

                # compute entropy loss per crop element
                loss = F.cross_entropy(dmaps_pred, dmaps, reduction='none')

                # consider only valid positions based on crop_mask
                loss = torch.sum(loss * dmsks) / feats.size(0)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    running_len += dmsks.sum().item()
                    running_loss += loss.item() * feats.size(0)

            running_len = 1 if running_len == 0 else running_len
            trainer.set_postfix(loss = running_loss / running_len)

            if (bidx + 1) % (len(data['train']) // 4) == 0:
                save_checkpoint(fname, bidx, last_epoch + e, running_loss / running_len, model, optimizer)

    return model, optimizer
