import numpy as np
import numpy.typing as npt
from scipy.interpolate import RBFInterpolator as RBF

def eval_at_xEdges(im: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fits a thin-plate-spline to the input image and evaluates it 
    at the edges of the voxels (in the x/horziontal direction)
    """
    
    assert im.ndim == 2, print(im.shape)
    
    # Create grid of points for interpolation
    rows, cols = im.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Flatten for RBF
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = im.ravel()
    
    # Mask handling
    if np.ma.isMA(im):
        mask = im.mask.ravel()
        points = points[~mask]
        values = values[~mask]
    
    # RBF interpolation
    rbf = RBF(points, values, kernel='thin_plate_spline')
    
    # Evaluate at edges: xEdges are at x - 0.5 and x + 0.5
    xEdges = np.arange(-0.5, cols + 0.5, 1.0)
    X_edges, Y_edges = np.meshgrid(xEdges, y)
    edge_points = np.column_stack((X_edges.ravel(), Y_edges.ravel()))
    
    im_xEdges = rbf(edge_points).reshape(rows, cols + 1)
    
    return im_xEdges

def eval_at_yEdges(im: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fits a thin-plate-spline to the input image and evaluates it 
    at the edges of the voxels (in the y/vertical direction)
    """
    
    assert im.ndim == 2, print(im.shape)
    
    # Create grid of points for interpolation
    rows, cols = im.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Flatten for RBF
    points = np.column_stack((X.ravel(), Y.ravel()))
    values = im.ravel()
    
    # Mask handling
    if np.ma.isMA(im):
        mask = im.mask.ravel()
        points = points[~mask]
        values = values[~mask]
    
    # RBF interpolation
    rbf = RBF(points, values, kernel='thin_plate_spline')
    
    # Evaluate at edges: yEdges are at y - 0.5 and y + 0.5
    yEdges = np.arange(-0.5, rows + 0.5, 1.0)
    X_edges, Y_edges = np.meshgrid(x, yEdges)
    edge_points = np.column_stack((X_edges.ravel(), Y_edges.ravel()))
    
    im_yEdges = rbf(edge_points).reshape(rows + 1, cols)
    
    return im_yEdges

def xGradient(im: npt.NDArray[np.float64], res_x: float) -> npt.NDArray[np.float64]:
    """Computes discrete derivatives of the input image along the x (horizontal) direction
    
    Parameters
    ----------
    im : npt.NDArray[np.float64]
        2D input image
    res_x: float
        Voxel size in the horizontal direction (in m)

    Returns
    -------
    npt.NDArray[np.float64]
        Discrete derivatives of the input image along the x (horizontal) direction (in /m)
    """
    
    assert im.ndim == 2, print(im.shape)
    
    im_xEdges = eval_at_xEdges(im)
    xgradient = np.zeros_like(im)
    
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if not np.ma.is_masked(im[row, col]):
                
                # If this is the leftmost pixel of this row -> don't trust the fit on the left edge (it is extrapolated)
                if col == 0 or np.ma.is_masked(im[row, col-1]): 
                    xgradient[row,col] =  im[row,col+1]- im[row,col] # (im_xEdges[row,col+1] - im[row,col])/0.5
                        
                # If this is the rightmost pixel of this row -> don't trust the fit on the right edge (it is extrapolated)
                elif col == im.shape[1]-1 or np.ma.is_masked(im[row, col+1]): 
                    xgradient[row,col] = im[row,col]- im[row,col-1] # (im[row,col] - im_xEdges[row,col])/0.5
                
                else:
                    xgradient[row,col] = im_xEdges[row,col+1] - im_xEdges[row,col]

        
    if np.ma.isMA(im):
        xgradient = np.ma.masked_where(im.mask, xgradient)
        
    return xgradient/res_x

def yGradient(im: npt.NDArray[np.float64], res_y: float) -> npt.NDArray[np.float64]:
    """Computes discrete derivatives of the input image along the y (vertical) direction
    
    Parameters
    ----------
    im : npt.NDArray[np.float64]
        2D input image
    res_y: float
        Voxel size in the vertical direction (in m)

    Returns
    -------
    npt.NDArray[np.float64]
        Discrete derivatives of the input image along the y (vertical) direction (in /m)
    """
    
    assert im.ndim == 2, print(im.shape)
    
    im_yEdges = eval_at_yEdges(im)
    ygradient = np.zeros_like(im)
    
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if not np.ma.is_masked(im[row, col]):
                
                # If this is the top pixel (assuming row 0 is top) -> don't trust the fit on the top edge
                if row == 0 or np.ma.is_masked(im[row-1, col]): 
                    ygradient[row,col] =  im[row+1,col]- im[row,col] 
                        
                # If this is the bottom pixel -> don't trust the fit on the bottom edge
                elif row == im.shape[0]-1 or np.ma.is_masked(im[row+1, col]): 
                    ygradient[row,col] = im[row,col]- im[row-1,col] 
                
                else:
                    ygradient[row,col] = im_yEdges[row+1,col] - im_yEdges[row,col]

        
    if np.ma.isMA(im):
        ygradient = np.ma.masked_where(im.mask, ygradient)
        
    return ygradient/res_y