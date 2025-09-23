import torch
from typing import List
from easyvolcap.engine import DATASETS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from easyvolcap.dataloaders.datasets.volumetric_video_inference_dataset import VolumetricVideoInferenceDataset


@DATASETS.register_module()
class ImageBasedInferenceDataset(VolumetricVideoInferenceDataset):
    def __init__(self,
                 n_srcs_list: List[int] = [3],  # MARK: repeated global configuration
                 n_srcs_prob: List[int] = [1.0],  # MARK: repeated global configuration
                 append_gt_prob: float = 1.0,
                 extra_src_pool: int = 1,
                 supply_decoded: bool = False,
                 barebone: bool = False,
                 skip_loading_images: bool = False,

                 #  closest_using_t: bool = False,
                 #  src_view_sample: List[int] = [0, None, 1],  # use these as input source views

                 **kwargs,
                 ):
        # NOTE: This file inherits from VolumetricVideoInferenceDataset instead of the ImageBasedDataset
        # Thus functions reusing implementation from that class should explicit define this
        # self.closest_using_t = closest_using_t  # MARK: transpose
        # self.src_view_sample = src_view_sample
        call_from_cfg(super().__init__, kwargs, skip_loading_images=skip_loading_images)  # will have prepared other parts of the dataset (interpolation or orbit)
        if self.src_view_sample != [0, None, 1] and self.view_sample != [0, None, 1]: log(yellow(f'Using `src_view_sample = {self.src_view_sample}` when `view_sample = {self.view_sample}` is not default'))

        # ImageBasedDataset.load_source_params(self)  # no extra dependencies
        ImageBasedDataset.load_source_indices(self)  # no extra dependencies
        self.n_srcs_list = n_srcs_list
        self.n_srcs_prob = n_srcs_prob
        self.extra_src_pool = extra_src_pool
        self.append_gt_prob = append_gt_prob  # manually assign values

        # src_inps will come in as decoded bytes instead of jpegs
        self.supply_decoded = supply_decoded
        self.barebone = barebone

    def load_interpolations(self):
        ImageBasedDataset.load_source_params(self)  # remember things

        # Actual interpolation
        super().load_interpolations()

        # For physical to virtual indexing (ibr inference uses physical indexing)
        # While the original inference dataset uses virtual indexing
        # fmt: off
        self.Hs    = self.Hs  .expand(-1, self.src_ixts.shape[1], *self.Hs.shape[2:]  )
        self.Ws    = self.Ws  .expand(-1, self.src_ixts.shape[1], *self.Ws.shape[2:]  )
        self.Ks    = self.Ks  .expand(-1, self.src_ixts.shape[1], *self.Ks.shape[2:]  )
        self.Rs    = self.Rs  .expand(-1, self.src_ixts.shape[1], *self.Rs.shape[2:]  )
        self.Ts    = self.Ts  .expand(-1, self.src_ixts.shape[1], *self.Ts.shape[2:]  )
        self.Cs    = self.Cs  .expand(-1, self.src_ixts.shape[1], *self.Cs.shape[2:]  )
        self.c2ws  = self.c2ws.expand(-1, self.src_ixts.shape[1], *self.c2ws.shape[2:])
        self.w2cs  = self.w2cs.expand(-1, self.src_ixts.shape[1], *self.w2cs.shape[2:])
        # fmt: on

    def load_paths(self):
        return VolumetricVideoDataset.load_paths(self)  # store images names

    def load_bytes(self):
        return VolumetricVideoDataset.load_bytes(self)  # store images

    def virtual_to_physical(self, latent_index: int):
        return VolumetricVideoDataset.virtual_to_physical(self, latent_index)

    def physical_to_virtual(self, latent_index: int):
        return VolumetricVideoDataset.physical_to_virtual(self, latent_index)

    def get_objects_bounds(self, latent_index: int):
        return VolumetricVideoDataset.get_objects_bounds(self, latent_index)

    def get_objects_priors(self, output: dotdict):
        return VolumetricVideoDataset.get_objects_priors(self, output)

    def load_source_params(self):
        return ImageBasedDataset.load_source_params(self)

    def __getitem__(self, index: dotdict):
        from easyvolcap.utils.ray_utils import weighted_sample_rays
        from easyvolcap.utils.data_utils import DataSplit, UnstructuredTensors, load_resize_undist_ims_bytes, load_image_from_bytes, as_torch_func, to_cuda, to_cpu, to_tensor, export_pts, load_pts, decode_crop_fill_ims_bytes, decode_fill_ims_bytes
        import cv2  # for undistortion
        from easyvolcap.utils.ray_utils import get_rays

        output = ImageBasedDataset.get_metadata(self, index)
        # try:
        #     rgb, msk, wet, dpt, bkg, norm = self.get_image(output.view_index, output.latent_index)  # H, W, 3
        # except:
        #     rgb, msk, wet, dpt, bkg, norm = self.get_image(0, 0)  # H, W, 3

        # H, W = rgb.shape[:2]

        # # # Maybe crop images
        # # if self.immask_crop:  # these variables are only available when loading gts
        # #     meta = dotdict()
        # #     meta.crop_x = self.crop_xs[output.view_index, output.latent_index]
        # #     meta.crop_y = self.crop_ys[output.view_index, output.latent_index]
        # #     meta.orig_h = self.orig_hs[output.view_index, output.latent_index]
        # #     meta.orig_w = self.orig_ws[output.view_index, output.latent_index]
        # #     output.update(meta)
        # #     output.meta.update(meta)

        # # elif self.imbound_crop:  # crop_x has already been set by imbound_crop for ixts
        # #     x, y, w, h = output.crop_x, output.crop_y, output.W, output.H
        # #     rgb = rgb[y:y + h, x:x + w]
        # #     msk = msk[y:y + h, x:x + w]
        # #     wet = wet[y:y + h, x:x + w]
        # #     if dpt is not None: dpt = dpt[y:y + h, x:x + w]
        # #     if bkg is not None: bkg = bkg[y:y + h, x:x + w]
        # #     if norm is not None: norm = norm[y:y + h, x:x + w]
        # #     H, W = h, w

        # # FIXME: Should add mutex to protect this， for now, multi-process and dataloading doesn't work well with each other
        # # If Moderators are used, should set num_workers to 0 for single-process data loading
        # n_rays = self.n_rays
        # patch_size = self.patch_size
        # render_ratio = self.render_ratio
        # random_crop_size = self.random_crop_size
        # render_center_crop_ratio = self.render_center_crop_ratio

        # # Prepare for a different rendering ratio
        # if (len(render_ratio.shape) and  # avoid length of 0-d tensor error, check length of shape
        #         render_ratio[output.view_index] != 1.0) or \
        #         render_ratio != 1.0:
        #     render_ratio = self.render_ratio[output.view_index] if len(self.render_ratio.shape) else self.render_ratio

        #     output = self.scale_ixts(output, render_ratio)
        #     H, W = output.H.item(), output.W.item()

        #     rgb = as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(rgb)
        #     msk = as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(msk)
        #     wet = as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(wet)
        #     if dpt is not None: as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(dpt)
        #     if bkg is not None: as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(bkg)
        #     if norm is not None: as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(norm)

        # # Prepare for a different rendering center crop ratio
        # if (len(render_center_crop_ratio.shape) and  # avoid length of 0-d tensor error, check length of shape
        #         render_center_crop_ratio[output.view_index] != 1.0) or \
        #         render_center_crop_ratio != 1.0:
        #     render_center_crop_ratio = self.render_center_crop_ratio[output.view_index] if len(self.render_center_crop_ratio.shape) else self.render_center_crop_ratio

        #     w, h = int(W * render_center_crop_ratio), int(H * render_center_crop_ratio)
        #     x, y = w // 2, h // 2

        #     # Center crop the target image
        #     rgb = rgb[y: y + h, x: x + w, :]
        #     msk = msk[y: y + h, x: x + w, :]
        #     wet = wet[y: y + h, x: x + w, :]
        #     if dpt is not None: dpt[y: y + h, x: x + w, :]
        #     if bkg is not None: bkg[y: y + h, x: x + w, :]
        #     if norm is not None: norm[y: y + h, x: x + w, :]

        #     # Crop the intrinsics
        #     self.crop_ixts(output, x, y, w, h)

        # should_sample_patch = False
        # should_crop_ixt = False

        # output.rgb = rgb.reshape(-1, 3)  # full image in case you need it
        # output.msk = msk.reshape(-1, 1)  # full mask
        # output.wet = wet.reshape(-1, 1)  # full weights
        # if dpt is not None: output.dpt = dpt.reshape(-1, 1)
        # if bkg is not None: output.bkg = bkg.reshape(-1, 3)
        # if norm is not None: output.norm = norm.reshape(-1, 3)

        # if should_crop_ixt:
        #     # Prepare the resized ixts
        #     self.crop_ixts(output, x, y, w, h)
        # elif should_sample_patch:
        #     # Prepare the full sampling output
        #     H, W = output.H, output.W
        #     K, R, T = output.K, output.R, output.T

        #     # Calculate the pixel coordinates
        #     ray_o, ray_d, coords = get_rays(H, W, K, R, T, z_depth=self.use_z_depth, correct_pix=self.correct_pix, ret_coord=True)  # maybe without normalization
        #     ray_o = ray_o[y: y + h, x: x + w, :]
        #     ray_d = ray_d[y: y + h, x: x + w, :]
        #     coords = coords[y: y + h, x: x + w, :]

        #     # Prepare for computing loss on patch
        #     meta = dotdict()
        #     meta.patch_h = torch.as_tensor(h)
        #     meta.patch_w = torch.as_tensor(w)
        #     output.update(meta)
        #     output.meta.update(meta)

        #     # Store full sampling output
        #     output.ray_o = ray_o.reshape(-1, 3)  # full coords
        #     output.ray_d = ray_d.reshape(-1, 3)  # full coords
        #     output.coords = coords.reshape(-1, 2)  # full coords

        return output

    # Manual polymorphism from ImageBasedDataset
    # NOTE: Refactor will never respect this
    @staticmethod
    def crop_ixts_xywh(size: List[int], output: dotdict):
        return ImageBasedDataset.crop_ixts_xywh(size, output)

    @staticmethod
    def crop_imgs_xywh(size: List[int], output: dotdict):
        return ImageBasedDataset.crop_imgs_xywh(size, output)

    @staticmethod
    def crop_srcs_mask(output: dotdict):
        return ImageBasedDataset.crop_srcs_mask(output)

    @staticmethod
    def crop_tars_mask(output: dotdict):
        return ImageBasedDataset.crop_tars_mask(output)

    def get_viewer_batch(self, batch):
        return ImageBasedDataset.get_viewer_batch(self, batch)

    def get_sources(self, *args, **kwargs):
        return ImageBasedDataset.get_sources(self, *args, **kwargs)
