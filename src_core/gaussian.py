import numpy as np
from scipy.ndimage import gaussian_filter


def create_gaussian_defect(center, size, sigma, image_shape):
    """
    Create a single gaussian defect at specified location
    
    Args:
        center: (x, y) center position of defect
        size: (height, width) size of defect region
        sigma: gaussian sigma value (can be tuple for different x,y sigma)
        image_shape: (height, width) shape of full image
        
    Returns:
        defect_image: full image with gaussian defect (0-1)
    """
    h, w = image_shape
    defect_image = np.zeros((h, w), dtype=np.float32)
    
    # Create coordinate grids
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Calculate gaussian
    cx, cy = center
    if isinstance(sigma, (list, tuple)):
        sigma_x, sigma_y = sigma
    else:
        sigma_x = sigma_y = sigma
    
    # Gaussian formula
    gaussian = np.exp(-((X - cx)**2 / (2 * sigma_x**2) + (Y - cy)**2 / (2 * sigma_y**2)))
    
    # Apply size constraint (optional, for more control)
    height, width = size
    mask_x = np.abs(X - cx) <= width / 2
    mask_y = np.abs(Y - cy) <= height / 2
    size_mask = mask_x & mask_y
    
    defect_image = gaussian * size_mask
    
    # Normalize to 0-1
    if defect_image.max() > 0:
        defect_image = defect_image / defect_image.max()
    
    return defect_image


def create_binary_mask(defect_image, threshold=0.1):
    """
    Create binary mask from gaussian defect
    
    Args:
        defect_image: gaussian defect image (0-1)
        threshold: threshold value (default 0.1 = 10%)
        
    Returns:
        binary_mask: binary mask (0 or 1)
    """
    return (defect_image > threshold).astype(np.float32)


def apply_defect_to_background(background, defect_image, intensity):
    """
    Apply defect to background image
    
    Args:
        background: background image (0-255)
        defect_image: gaussian defect (0-1)
        intensity: defect intensity (positive=bright, negative=dark)
        
    Returns:
        output: image with defect applied (same dtype as background)
    """
    output = background + defect_image * intensity
    output = np.clip(output, 0, 255)
    # Keep the same dtype as the input background
    return output.astype(background.dtype)


def generate_multiple_defects(image_shape, defect_params_list):
    """
    Generate multiple defects on single image
    
    Args:
        image_shape: (height, width)
        defect_params_list: list of dict with keys:
            - center: (x, y)
            - size: (height, width)
            - sigma: gaussian sigma
            - intensity: defect intensity
            
    Returns:
        combined_defect: combined defect image
        defect_images: list of individual defect images
    """
    h, w = image_shape
    combined_defect = np.zeros((h, w), dtype=np.float32)
    defect_images = []
    
    for params in defect_params_list:
        defect = create_gaussian_defect(
            center=params['center'],
            size=params['size'],
            sigma=params['sigma'],
            image_shape=image_shape
        )
        defect_images.append(defect)
        
        # Combine defects (for now, use maximum for overlapping)
        # This maintains the strongest defect at each pixel
        combined_defect = np.maximum(combined_defect, defect)
    
    return combined_defect, defect_images


def generate_random_defect_params(image_shape, num_defects, 
                                size_range=(10, 30),
                                sigma_range=(3, 10),
                                intensity_range=(-50, 50)):
    """
    Generate random defect parameters
    
    Args:
        image_shape: (height, width)
        num_defects: number of defects to generate
        size_range: (min, max) size range
        sigma_range: (min, max) sigma range
        intensity_range: (min, max) intensity range
        
    Returns:
        defect_params_list: list of defect parameters
    """
    h, w = image_shape
    defect_params_list = []
    
    for _ in range(num_defects):
        # Random size
        size = np.random.uniform(*size_range)
        
        # Random sigma (proportional to size)
        sigma = size / 3  # Rule of thumb: sigma = size/3
        
        # Random center (ensure defect fits in image)
        margin = size / 2
        center_x = np.random.uniform(margin, w - margin)
        center_y = np.random.uniform(margin, h - margin)
        
        # Random intensity
        intensity = np.random.uniform(*intensity_range)
        
        params = {
            'center': (center_x, center_y),
            'size': (size, size),  # Square defect for now (height, width)
            'sigma': sigma,
            'intensity': intensity
        }
        defect_params_list.append(params)
    
    return defect_params_list


def create_local_gaussian_defect(center, size, sigma, patch_shape, patch_offset=None):
    """
    Create gaussian defect only in local region (optimized version)
    
    Args:
        center: (x, y) center position in global coordinates
        size: (height, width) size of defect region
        sigma: gaussian sigma value (can be tuple for different x,y sigma)
        patch_shape: (height, width) shape of the patch
        patch_offset: (x, y) offset of patch in global image
        
    Returns:
        local_defect: defect array only for affected region
        local_bounds: (y_start, y_end, x_start, x_end) in patch coordinates
    """
    if patch_offset is None:
        patch_offset = (0, 0)
    
    h, w = patch_shape
    offset_x, offset_y = patch_offset
    cx, cy = center
    
    # Convert to local coordinates
    local_cx = cx - offset_x
    local_cy = cy - offset_y
    
    # Calculate affected region (3 sigma rule)
    height, width = size
    margin_y = height * 3 if isinstance(sigma, (list, tuple)) else max(height, width) * 3
    margin_x = width * 3 if isinstance(sigma, (list, tuple)) else max(height, width) * 3
    
    y_start = max(0, int(local_cy - margin_y))
    y_end = min(h, int(local_cy + margin_y))
    x_start = max(0, int(local_cx - margin_x))
    x_end = min(w, int(local_cx + margin_x))
    
    # Early exit if defect is completely outside patch
    if y_start >= h or y_end <= 0 or x_start >= w or x_end <= 0:
        return None, None
    
    # Create local coordinate grid
    local_y = np.arange(y_start, y_end)
    local_x = np.arange(x_start, x_end)
    X, Y = np.meshgrid(local_x, local_y)
    
    # Calculate gaussian
    if isinstance(sigma, (list, tuple)):
        sigma_x, sigma_y = sigma
    else:
        sigma_x = sigma_y = sigma
    
    # Gaussian formula
    gaussian = np.exp(-((X - local_cx)**2 / (2 * sigma_x**2) + 
                       (Y - local_cy)**2 / (2 * sigma_y**2)))
    
    # Apply size constraint
    mask_x = np.abs(X - local_cx) <= width / 2
    mask_y = np.abs(Y - local_cy) <= height / 2
    size_mask = mask_x & mask_y
    
    local_defect = gaussian * size_mask
    
    # Normalize to 0-1
    if local_defect.max() > 0:
        local_defect = local_defect / local_defect.max()
    
    return local_defect, (y_start, y_end, x_start, x_end)


def apply_local_defect_to_background(background, local_defect, bounds, intensity):
    """
    Apply local defect to background at specified bounds
    
    Args:
        background: background image (0-255)
        local_defect: local gaussian defect (0-1)
        bounds: (y_start, y_end, x_start, x_end) where to apply defect
        intensity: defect intensity (positive=bright, negative=dark)
        
    Returns:
        output: image with defect applied
    """
    if local_defect is None:
        return background
    
    y_start, y_end, x_start, x_end = bounds
    output = background.copy()
    output[y_start:y_end, x_start:x_end] += local_defect * intensity
    output = np.clip(output, 0, 255)
    return output.astype(background.dtype)