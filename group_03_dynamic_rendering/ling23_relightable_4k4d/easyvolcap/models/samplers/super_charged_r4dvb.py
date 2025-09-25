"""
Realtime4DV with background
Performs simple composition rendering with alpha blending and not fancy resizing
Potentially optimizable up to the rendering speed difference of high-res and low-res ones
"""
# This is a inference sampler for the turbo-charged point-planes model.
from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Tuple
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset

import os
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from types import MethodType

from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS, EMBEDDERS, REGRESSORS
from easyvolcap.models.samplers.uniform_sampler import UniformSampler
from easyvolcap.models.samplers.r4dv_sampler import R4DVSampler
from easyvolcap.models.samplers.r4dvb_sampler import R4DVBSampler
from easyvolcap.models.samplers.super_charged_r4dv import SuperChargedR4DV, average_single_frame, load_state_dict_kwargs
from easyvolcap.models.samplers.point_planes_sampler import PointPlanesSampler
from easyvolcap.models.networks.noop_network import NoopNetwork

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.cuda_utils import register_memory
from easyvolcap.utils.net_utils import make_buffer, VolumetricVideoModule
from easyvolcap.utils.data_utils import DataSplit, UnstructuredTensors, load_resize_undist_ims_bytes, load_image_from_bytes, as_torch_func, to_cuda, to_cpu, to_tensor, export_pts, load_pts, decode_crop_fill_ims_bytes, decode_fill_ims_bytes

from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
from easyvolcap.models.networks.embedders.kplanes_embedder import KPlanesEmbedder


def forward_bg(self: SuperChargedR4DV,
               batch: dotdict,
               return_frags: bool = True):
    # Get corresponding indices for sampling
    index = [0]  # manual index for this
    xyz = torch.stack([self.pcds[l] for l in index])  # B, N, 3
    rad = torch.stack([self.rads[l] for l in index])  # B, N, 3
    occ = torch.stack([self.occs[l] for l in index])  # B, N, 3
    sh = torch.stack([self.shs[l] for l in index])  # B, N, 3

    index, time = self.sample_index_time(batch)
    rgbw = self.fetch(index, [self.rgbws])  # will initiate copy for both rgbw and sh
    rgbw = torch.stack([v[0] for v in rgbw])
    cent = torch.stack([self.cents[l] for l in index])  # B, S, 3

    is_relight_mode = rgbw.shape[-1] > 4
    if self.skip_shs or is_relight_mode:
        sh[:] = 0
    if self.skip_base:
        sh = sh.abs()
        rgbw[..., :3] = 0

    rgb = self.get_rgb(batch.R.half(), batch.T.half(), xyz, sh, rgbw, cent, self.n_srcs, self.n_shs, self.ibr_resd_limit)

    if is_relight_mode:
        basecolor = self.parse_rgb(xyz, rgb, batch, 'basecolor')
        normal = self.parse_rgb(xyz, rgb, batch, 'normal')
        metallic = self.parse_rgb(xyz, rgb, batch, 'metallic')
        roughness = self.parse_rgb(xyz, rgb, batch, 'roughness')
        relight_0 = self.parse_rgb(xyz, rgb, batch, 'relight_0')
        relight_1 = self.parse_rgb(xyz, rgb, batch, 'relight_1')
        relight_2 = self.parse_rgb(xyz, rgb, batch, 'relight_2')
        relight_3 = self.parse_rgb(xyz, rgb, batch, 'relight_3')
        relight_4 = self.parse_rgb(xyz, rgb, batch, 'relight_4')
        rgb = relight_0
    else:
        rgb = self.parse_rgb(xyz, rgb, batch, 'rgb')
        basecolor = None
        normal = None
        metallic = None
        roughness = None
        relight_0 = None
        relight_1 = None
        relight_2 = None
        relight_3 = None
        relight_4 = None

    if return_frags:
        return None, xyz, rgb, rad, occ, basecolor, normal, metallic, roughness, relight_0, relight_1, relight_2, relight_3, relight_4

    rgb, acc, dpt = self.render_points(xyz, rgb, rad, occ, batch)  # almost always use render_cudagl
    if basecolor is not None:
        basecolor, _, _ = self.render_points(xyz, basecolor, rad, occ, batch)
    if normal is not None:
        normal, _, _ = self.render_points(xyz, normal, rad, occ, batch)
    if metallic is not None:
        metallic, _, _ = self.render_points(xyz, metallic, rad, occ, batch)
    if roughness is not None:
        roughness, _, _ = self.render_points(xyz, roughness, rad, occ, batch)
    if relight_0 is not None:
        relight_0, _, _ = self.render_points(xyz, relight_0, rad, occ, batch)
    if relight_1 is not None:
        relight_1, _, _ = self.render_points(xyz, relight_1, rad, occ, batch)
    if relight_2 is not None:
        relight_2, _, _ = self.render_points(xyz, relight_2, rad, occ, batch)
    if relight_3 is not None:
        relight_3, _, _ = self.render_points(xyz, relight_3, rad, occ, batch)
    if relight_4 is not None:
        relight_4, _, _ = self.render_points(xyz, relight_4, rad, occ, batch)

    self.store_output(None, xyz, rgb, acc, dpt, batch, basecolor, normal, metallic, roughness, relight_0, relight_1, relight_2, relight_3, relight_4)
    return None


@SAMPLERS.register_module()
class SuperChargedR4DVB(VolumetricVideoModule):
    def __init__(self,
                 network: NoopNetwork,
                 fg_sampler_cfg: dotdict = dotdict(type=SuperChargedR4DV.__name__),
                 bg_sampler_cfg: dotdict = dotdict(type=SuperChargedR4DV.__name__),
                 should_release_memory: bool = True,
                 **kwargs,
                 ):
        super().__init__(network, **kwargs)
        self.fg_sampler: SuperChargedR4DV = SAMPLERS.build(fg_sampler_cfg, network=network, **kwargs)
        self.bg_sampler: SuperChargedR4DV = SAMPLERS.build(bg_sampler_cfg, network=network, n_frames=1, frame_sample=[0, 1, 1], **kwargs)
        self.fg_sampler.should_release_memory = False
        self.bg_sampler.should_release_memory = False
        self.should_release_memory = should_release_memory

        self.fg_sampler.post_handle.remove()
        self.bg_sampler.post_handle.remove()

        self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)
        self.bg_sampler.forward = MethodType(forward_bg, self.bg_sampler)

        self.bg_sampler.streams: List[torch.cuda.Stream] = [torch.cuda.Stream() for _ in self.fg_sampler.pcds]  # data moving streams
        self.bg_sampler.cache: List[torch.Tensor] = [None for _ in self.fg_sampler.pcds]

    @property
    def pts_per_pix(self):
        return self.bg_sampler.pts_per_pix

    @pts_per_pix.setter
    def pts_per_pix(self, v: int):
        self.fg_sampler.pts_per_pix = v  # UNUSED
        self.bg_sampler.pts_per_pix = v

    @property
    def bg_brightness(self):
        return self.bg_sampler.bg_brightness

    @bg_brightness.setter
    def bg_brightness(self, v: float):
        self.fg_sampler.bg_brightness = v  # UNUSED
        self.bg_sampler.bg_brightness = v

    @property
    def volume_rendering(self):
        return self.bg_sampler.volume_rendering

    @volume_rendering.setter
    def volume_rendering(self, v: bool):
        self.fg_sampler.volume_rendering = v  # UNUSED
        self.bg_sampler.volume_rendering = v

    @property
    def use_cudagl(self):
        return self.bg_sampler.use_cudagl

    @use_cudagl.setter
    def use_cudagl(self, v: bool):
        self.fg_sampler.use_cudagl = v  # UNUSED
        self.bg_sampler.use_cudagl = v

    @property
    def use_diffgl(self):
        return self.bg_sampler.use_diffgl

    @use_diffgl.setter
    def use_diffgl(self, v: bool):
        self.fg_sampler.use_diffgl = v  # UNUSED
        self.bg_sampler.use_diffgl = v

    @property
    def use_pulsar(self):
        return self.bg_sampler.use_pulsar

    @use_pulsar.setter
    def use_pulsar(self, v: bool):
        self.fg_sampler.use_pulsar = v  # UNUSED
        self.bg_sampler.use_pulsar = v

    @property
    def skip_shs(self):
        return self.bg_sampler.skip_shs

    @skip_shs.setter
    def skip_shs(self, v: bool):
        self.fg_sampler.skip_shs = v  # UNUSED
        self.bg_sampler.skip_shs = v

    @property
    def skip_base(self):
        return self.bg_sampler.skip_base

    @skip_base.setter
    def skip_base(self, v: bool):
        self.fg_sampler.skip_base = v  # UNUSED
        self.bg_sampler.skip_base = v

    @property
    def render_gs(self):
        return self.bg_sampler.render_gs

    @render_gs.setter
    def render_gs(self, v: bool):
        self.fg_sampler.render_gs = v  # UNUSED
        self.bg_sampler.render_gs = v

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        PointPlanesSampler.render_imgui(self, viewer, batch)

        from imgui_bundle import imgui_toggle
        toggle_ios_style = imgui_toggle.ios_style(size_scale=0.2)
        self.skip_shs = imgui_toggle.toggle('Skip SHs', self.skip_shs, config=toggle_ios_style)[1]
        self.skip_base = imgui_toggle.toggle('Skip base', self.skip_base, config=toggle_ios_style)[1]
        self.render_gs = imgui_toggle.toggle('Render GS', self.render_gs, config=toggle_ios_style)[1]

    @torch.no_grad()
    def _load_state_dict_post_hook(self: SuperChargedR4DVB, module: SuperChargedR4DV, incompatible_keys):
        # Prepare the dataset to be loaded in post hook
        store = dotdict()
        dataset: ImageBasedDataset = cfg.runner.val_dataloader.dataset

        # After preparing the dataset and the module, can load state dict
        store.cameras = dataset.cameras
        store.frame_sample = dataset.frame_sample
        dataset.cameras = {k: [v[0]] for k, v in dataset.cameras.items()}
        dataset.frame_sample = [0, 1, 1]
        self.bg_sampler._load_state_dict_post_hook(module, incompatible_keys)
        del dataset.vhull_bounds  # no bbox for fullscreen dataset
        dataset.cameras = store.cameras
        dataset.frame_sample = store.frame_sample

        # Prepare dtype
        self.fg_sampler.type(self.fg_sampler.compute_dtype)
        self.bg_sampler.type(self.bg_sampler.compute_dtype)

        # Load all images into self.bg_sampler.rgbws
        runner: 'VolumetricVideoRunner' = cfg.runner  # assume the runner has all we need now
        dataset: 'ImageBasedDataset' = runner.val_dataloader.dataset
        kwargs = load_state_dict_kwargs(self.bg_sampler.pcds[0].device)
        times = kwargs.times
        n_views = kwargs.n_views
        n_latents = kwargs.n_latents
        b, e, s = kwargs.b, kwargs.e, kwargs.s
        tb, te, ts = kwargs.tb, kwargs.te, kwargs.ts

        # Preparing functions
        xyz = self.bg_sampler.pcds[0][None].detach()
        t = times[0]
        xyz_t = t[None, None].expand(*xyz.shape[:-1], 1)  # B, N, 1
        xyz_feat: torch.Tensor = self.bg_sampler.xyz_embedder(xyz, xyz_t).detach()  # same time

        @torch.no_grad()
        def l_forward_for_xyz_feat(i: int):
            return xyz, xyz_feat  # this is without batch dimension

        l_average_single_frame = partial(average_single_frame, forward_for_xyz_feat=l_forward_for_xyz_feat, sampler=self.bg_sampler, dataset=dataset, runner=runner, **kwargs)

        # Preparing buffers using prepared functions
        self.bg_sampler.rgbws = [None for _ in self.fg_sampler.pcds]  # HUGE
        self.bg_sampler.cents = nn.ParameterList([None for _ in self.fg_sampler.pcds])  # OK
        for i in tqdm(range(n_latents), desc=f'Caching rgbw and center'):
            rgbw, cent = l_average_single_frame(i)
            rgbw = rgbw.to(self.bg_sampler.dtype).view(self.bg_sampler.memory_dtype).detach().cpu(memory_format=torch.contiguous_format)  # MARK: SYNC
            torch.cuda.empty_cache()
            rgbw = register_memory(rgbw)
            self.bg_sampler.rgbws[(b + i * s) // ts - tb] = make_buffer(rgbw[0])
            self.bg_sampler.cents[(b + i * s) // ts - tb] = make_buffer(cent[0])

        for i, s in enumerate(self.bg_sampler.shs):
            self.bg_sampler.shs[i] = s.to(self.bg_sampler.pcds[0].device, non_blocking=True).view(self.bg_sampler.dtype)

        # Precrop image to bytes
        store.Ks, store.Hs, store.Ws = dataset.Ks.clone(), dataset.Hs.clone(), dataset.Ws.clone()  # avoid inplace modification
        store.ims_bytes, store.mks_bytes = dataset.ims_bytes.clone(), dataset.mks_bytes.clone()
        if dataset.bcs_bytes is not None:
            store.bcs_bytes = dataset.bcs_bytes.clone()
        if dataset.sns_bytes is not None:
            store.sns_bytes = dataset.sns_bytes.clone()
        if dataset.mts_bytes is not None:
            store.mts_bytes = dataset.mts_bytes.clone()
        if dataset.rgs_bytes is not None:
            store.rgs_bytes = dataset.rgs_bytes.clone()
        store.src_ixts, store.src_exts = dataset.src_ixts.clone(), dataset.src_exts.clone()

        ims_bytes = []
        mks_bytes = []
        if dataset.bcs_bytes is not None:
            bcs_bytes = []
        else:
            bcs_bytes = None
        if dataset.sns_bytes is not None:
            sns_bytes = []
        else:
            sns_bytes = None
        if dataset.mts_bytes is not None:
            mts_bytes = []
        else:
            mts_bytes = None
        if dataset.rgs_bytes is not None:
            rgs_bytes = []
        else:
            rgs_bytes = None
        for i in range(dataset.n_views):
            for j in range(dataset.n_latents):
                ims_bytes.append(dataset.ims_bytes[i * dataset.n_latents + j])
                mks_bytes.append(dataset.mks_bytes[i * dataset.n_latents + j])
                if bcs_bytes is not None:
                    bcs_bytes.append(dataset.bcs_bytes[i * dataset.n_latents + j])
                if sns_bytes is not None:
                    sns_bytes.append(dataset.sns_bytes[i * dataset.n_latents + j])
                if mts_bytes is not None:
                    mts_bytes.append(dataset.mts_bytes[i * dataset.n_latents + j])
                if rgs_bytes is not None:
                    rgs_bytes.append(dataset.rgs_bytes[i * dataset.n_latents + j])

        bounds = [dataset.get_bounds(i) for i in range(dataset.n_latents)]  # N, 2, 3
        bounds = torch.stack(bounds)[None].repeat(dataset.n_views, 1, 1, 1)  # V, N, 2, 3
        dataset.ims_bytes, dataset.mks_bytes, dataset.Ks, dataset.Hs, dataset.Ws, _, _ = \
            decode_crop_fill_ims_bytes(ims_bytes, mks_bytes, dataset.src_ixts.numpy(), dataset.src_exts[..., :3, :3].numpy(), dataset.src_exts[..., :3, 3:].numpy(), bounds.numpy(), f'Cropping msks imgs for {blue(dataset.data_root)}, split {magenta(dataset.split.name)}')
        if dataset.bcs_bytes is not None:
            dataset.bcs_bytes, _, _, _, _, _, _ = \
                decode_crop_fill_ims_bytes(bcs_bytes, mks_bytes, dataset.src_ixts.numpy(), dataset.src_exts[..., :3, :3].numpy(), dataset.src_exts[..., :3, 3:].numpy(), bounds.numpy(), f'Cropping bcs imgs for {blue(dataset.data_root)}, split {magenta(dataset.split.name)}')
        if dataset.sns_bytes is not None:
            dataset.sns_bytes, _, _, _, _, _, _ = \
                decode_crop_fill_ims_bytes(sns_bytes, mks_bytes, dataset.src_ixts.numpy(), dataset.src_exts[..., :3, :3].numpy(), dataset.src_exts[..., :3, 3:].numpy(), bounds.numpy(), f'Cropping sns imgs for {blue(dataset.data_root)}, split {magenta(dataset.split.name)}')
        if dataset.mts_bytes is not None:
            dataset.mts_bytes, _, _, _, _, _, _ = \
                decode_crop_fill_ims_bytes(mts_bytes, mks_bytes, dataset.src_ixts.numpy(), dataset.src_exts[..., :3, :3].numpy(), dataset.src_exts[..., :3, 3:].numpy(), bounds.numpy(), f'Cropping mts imgs for {blue(dataset.data_root)}, split {magenta(dataset.split.name)}')
        if dataset.rgs_bytes is not None:
            dataset.rgs_bytes, _, _, _, _, _, _ = \
                decode_crop_fill_ims_bytes(rgs_bytes, mks_bytes, dataset.src_ixts.numpy(), dataset.src_exts[..., :3, :3].numpy(), dataset.src_exts[..., :3, 3:].numpy(), bounds.numpy(), f'Cropping rgs imgs for {blue(dataset.data_root)}, split {magenta(dataset.split.name)}')
        dataset.Ks = torch.as_tensor(dataset.Ks)
        dataset.Hs = torch.as_tensor(dataset.Hs)
        dataset.Ws = torch.as_tensor(dataset.Ws)
        dataset.ims_bytes = UnstructuredTensors(dataset.ims_bytes)
        dataset.mks_bytes = UnstructuredTensors(dataset.mks_bytes)
        if dataset.bcs_bytes is not None:
            dataset.bcs_bytes = UnstructuredTensors(dataset.bcs_bytes)
        if dataset.sns_bytes is not None:
            dataset.sns_bytes = UnstructuredTensors(dataset.sns_bytes)
        if dataset.mts_bytes is not None:
            dataset.mts_bytes = UnstructuredTensors(dataset.mts_bytes)
        if dataset.rgs_bytes is not None:
            dataset.rgs_bytes = UnstructuredTensors(dataset.rgs_bytes)
        dataset.load_source_params()  # only update src_ixts
        dataset.src_exts = store.src_exts

        # After preparing the dataset and the module, can load state dict
        self.fg_sampler._load_state_dict_post_hook(module, incompatible_keys)  # MARK: where the real work is done
        dataset.Ks, dataset.Hs, dataset.Ws = store.Ks, store.Hs, store.Ws
        dataset.ims_bytes, dataset.mks_bytes = store.ims_bytes, store.mks_bytes
        if dataset.bcs_bytes is not None:
            dataset.bcs_bytes = store.bcs_bytes
        if dataset.sns_bytes is not None:
            dataset.sns_bytes = store.sns_bytes
        if dataset.mts_bytes is not None:
            dataset.mts_bytes = store.mts_bytes
        if dataset.rgs_bytes is not None:
            dataset.rgs_bytes = store.rgs_bytes
        dataset.load_source_params()

        # Always perform the clean up
        if self.should_release_memory:
            dataset.ims_bytes = None
            dataset.mks_bytes = None
            if dataset.bcs_bytes is not None:
                dataset.bcs_bytes = None
            if dataset.sns_bytes is not None:
                dataset.sns_bytes = None
            if dataset.mts_bytes is not None:
                dataset.mts_bytes = None
            if dataset.rgs_bytes is not None:
                dataset.rgs_bytes = None
            self.bg_sampler.release_memory()
            self.fg_sampler.release_memory()

        # Restore dtype
        self.fg_sampler.type(self.fg_sampler.dtype)
        self.bg_sampler.type(self.bg_sampler.dtype)

        # Give OpenGL some breathing room
        torch.cuda.empty_cache()
        del dataset.vhull_bounds  # no bbox for fullscreen dataset

    @torch.no_grad()
    def construct_from_runner(self, runner: 'VolumetricVideoRunner'):
        sampler: R4DVBSampler = runner.model.sampler
        b, e, s = sampler.fg_sampler.frame_sample
        n_frames = sampler.fg_sampler.n_frames

        # Construct foreground sampler
        fg_runner = dotdict()
        fg_runner.model = dotdict()
        fg_runner.model.sampler = sampler.fg_sampler
        self.fg_sampler.construct_from_runner(fg_runner)  # this should be pretty smooth

        # Construct background sampler
        bg_runner = dotdict()
        bg_runner.model = dotdict()
        bg_runner.model.sampler = sampler.bg_sampler
        self.bg_sampler.construct_from_runner(bg_runner)  # this should be pretty smooth

    def forward(self, batch: dotdict):
        _, fg_xyz, fg_rgb, fg_rad, fg_occ, fg_basecolor, fg_normal, fg_metallic, fg_roughness, fg_relight_0, fg_relight_1, fg_relight_2, fg_relight_3, fg_relight_4 = self.fg_sampler.forward(batch=batch, return_frags=True)
        _, bg_xyz, bg_rgb, bg_rad, bg_occ, bg_basecolor, bg_normal, bg_metallic, bg_roughness, bg_relight_0, bg_relight_1, bg_relight_2, bg_relight_3, bg_relight_4 = self.bg_sampler.forward(batch=batch, return_frags=True)

        xyz = torch.cat([fg_xyz, bg_xyz], dim=-2)
        rgb = torch.cat([fg_rgb, bg_rgb], dim=-2)
        rad = torch.cat([fg_rad, bg_rad], dim=-2)
        occ = torch.cat([fg_occ, bg_occ], dim=-2)
        if fg_basecolor is not None and bg_basecolor is not None:
            basecolor = torch.cat([fg_basecolor, bg_basecolor], dim=-2)
        else:
            basecolor = None
        if fg_normal is not None and bg_normal is not None:
            normal = torch.cat([fg_normal, bg_normal], dim=-2)
        else:
            normal = None
        if fg_metallic is not None and bg_metallic is not None:
            metallic = torch.cat([fg_metallic, bg_metallic], dim=-2)
        else:
            metallic = None
        if fg_roughness is not None and bg_roughness is not None:
            roughness = torch.cat([fg_roughness, bg_roughness], dim=-2)
        else:
            roughness = None
        if fg_relight_0 is not None and bg_relight_0 is not None:
            relight_0 = torch.cat([fg_relight_0, bg_relight_0], dim=-2)
        else:
            relight_0 = None
        if fg_relight_1 is not None and bg_relight_1 is not None:
            relight_1 = torch.cat([fg_relight_1, bg_relight_1], dim=-2)
        else:
            relight_1 = None
        if fg_relight_2 is not None and bg_relight_2 is not None:
            relight_2 = torch.cat([fg_relight_2, bg_relight_2], dim=-2)
        else:
            relight_2 = None
        if fg_relight_3 is not None and bg_relight_3 is not None:
            relight_3 = torch.cat([fg_relight_3, bg_relight_3], dim=-2)
        else:
            relight_3 = None
        if fg_relight_4 is not None and bg_relight_4 is not None:
            relight_4 = torch.cat([fg_relight_4, bg_relight_4], dim=-2)
        else:
            relight_4 = None

        # Perform points rendering
        rgb, acc, dpt = self.bg_sampler.render_points(xyz, rgb, rad, occ, batch)  # B, HW, C
        if basecolor is not None:
            basecolor, _, _ = self.bg_sampler.render_points(xyz, basecolor, rad, occ, batch)
        if normal is not None:
            normal, _, _ = self.bg_sampler.render_points(xyz, normal, rad, occ, batch)
        if metallic is not None:
            metallic, _, _ = self.bg_sampler.render_points(xyz, metallic, rad, occ, batch)
        if roughness is not None:
            roughness, _, _ = self.bg_sampler.render_points(xyz, roughness, rad, occ, batch)
        if relight_0 is not None:
            relight_0, _, _ = self.bg_sampler.render_points(xyz, relight_0, rad, occ, batch)
        if relight_1 is not None:
            relight_1, _, _ = self.bg_sampler.render_points(xyz, relight_1, rad, occ, batch)
        if relight_2 is not None:
            relight_2, _, _ = self.bg_sampler.render_points(xyz, relight_2, rad, occ, batch)
        if relight_3 is not None:
            relight_3, _, _ = self.bg_sampler.render_points(xyz, relight_3, rad, occ, batch)
        if relight_4 is not None:
            relight_4, _, _ = self.bg_sampler.render_points(xyz, relight_4, rad, occ, batch)
        self.bg_sampler.store_output(None, xyz, rgb, acc, dpt, batch, basecolor, normal, metallic, roughness, relight_0, relight_1, relight_2, relight_3, relight_4)
