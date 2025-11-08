import torch
from einops import rearrange
import rp


def unique_pixels(image: torch.Tensor):
    """Return unique pixel values, counts, and index matrix for CHW image."""
    c, h, w = image.shape
    pixels = rearrange(image, "c h w -> (h w) c")
    unique_colors, inverse_indices, counts = torch.unique(
        pixels, dim=0, return_inverse=True, return_counts=True, sorted=False
    )
    u = unique_colors.shape[0]
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

    return unique_colors, counts, index_matrix


def sum_indexed_values(image: torch.Tensor, index_matrix: torch.Tensor):
    """Sum CHW image values grouped by HW index matrix."""
    c, h, w = image.shape
    u = index_matrix.max() + 1
    pixels = rearrange(image, "c h w -> (h w) c")
    output = torch.zeros((u, c), dtype=pixels.dtype, device=pixels.device)
    output.index_add_(0, index_matrix.view(-1), pixels)

    assert image.shape == (c, h, w)
    assert index_matrix.shape == (h, w)
    assert output.shape == (u, c)

    return output


def indexed_to_image(index_matrix: torch.Tensor, unique_colors: torch.Tensor):
    """Convert HW index matrix and UC colors back into CHW image."""
    h, w = index_matrix.shape
    u, c = unique_colors.shape
    assert index_matrix.max() < u
    flattened_image = unique_colors[index_matrix.view(-1)]
    image = rearrange(flattened_image, "(h w) c -> c h w", h=h, w=w)
    assert image.shape == (c, h, w)
    return image


def regaussianize(noise: torch.Tensor):
    """Reintroduce Gaussian statistics into pixelated noise."""
    c, hs, ws = noise.shape
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)
    assert unique_colors.shape == (u, 1)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(noise)
    # alternative: use uniform noise instead
    # foreign_noise = torch.rand_like(noise) * 2 - 1
    assert foreign_noise.shape == noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    # alternative: just use foreign noise directly
    # zeroed_foreign_noise = foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    output = noise / counts_image ** 0.5
    # alternative: just add foreign noise directly
    # output = output + zeroed_foreign_noise
    
    assert output.shape == noise.shape == (c, hs, ws)
    return output, counts_image


def resize_noise(noise, size, alpha=None):
    """Downscale Gaussian noise while preserving variance."""
    if rp.is_numpy_array(noise):
        noise = rp.as_torch_image(noise)
        output = resize_noise(noise, size, alpha)
        output = rp.as_numpy_array(output)
        return rearrange(output, "c h w -> h w c")

    if noise.ndim == 4:
        return torch.stack([resize_noise(x, size, alpha) for x in noise])

    assert noise.ndim == 3, "resize_noise expects CHW tensor input"
    _, old_height, old_width = noise.shape

    if rp.is_number(size):
        new_height, new_width = int(old_height * size), int(old_width * size)
    else:
        new_height, new_width = size

    assert new_height <= old_height, "resize_noise supports shrinking only"
    assert new_width <= old_width, "resize_noise supports shrinking only"

    x, y = rp.xy_torch_matrices(old_height, old_width, max_x=new_width, max_y=new_height)

    if alpha is not None:
        assert alpha.ndim == 2
        assert alpha.shape == noise.shape[1:]
        noise = torch.cat((alpha[None], noise))

    resized = rp.torch_scatter_add_image(
        noise,
        x,
        y,
        height=new_height,
        width=new_width,
        interp="floor",
        prepend_ones=alpha is None,
    )

    total, resized = resized[:1], resized[1:]
    adjusted = resized / total ** 0.5
    return adjusted


__all__ = [
    "unique_pixels",
    "sum_indexed_values",
    "indexed_to_image",
    "regaussianize",
    "resize_noise",
]
