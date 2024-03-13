import numpy as np
from scipy import ndimage

def SL_IO_correct_image(image_data, files):

    # For SL only!
    
    # Return an image containing the flatfielded interpolated
    # inter-order light that can be subtracted from a BCD.

    # files is a dictionary that contains necessary info for this process:
    # files = {
    #     'flatfield':,
    #     'SL1_mask':,
    #     'SL2_mask':,
    # }
    
    flatfield = files['flatfield']
    SL1_mask = files['SL1_mask']
    SL2_mask = files['SL2_mask']
    
    def get_row_image(wedgesub_image_data):
        
        SL_corridor_image = np.where((pix>30)&(pix<50)&(flatfield!=flatfield), 
                                     wedgesub_image_data, np.nan)
        PU_corridor_image = np.where((pix>70)&(pix<90)&(flatfield!=flatfield), 
                                     wedgesub_image_data, np.nan)
        
        SL_x, PU_x = 40, 75
        
        SL1_io_im = []
        SL2_io_im = []
        for i in range(128):

            SL_corr_val = np.nanmedian(SL_corridor_image[i])
            PU_corr_val = np.nanmedian(PU_corridor_image[i])

            SL1_line = np.ones(128)*SL_corr_val
            SL1_io_im.append(SL1_line)

            SL2_line = np.ones(128)*np.mean((SL_corr_val,PU_corr_val))
            SL2_io_im.append(SL2_line)

        SL1_io_im = np.asarray(SL1_io_im)*SL1_mask
        SL2_io_im = np.asarray(SL2_io_im)*SL2_mask
        
        def smoother(image):
            
            smoothing = 2
            smoothed_im = []
            for i in pix:
                if i >= smoothing:
                    new_row = np.nanmedian(image[i-smoothing:i+smoothing], axis=0)
                else:
                    new_row = np.nanmedian(image[0:i+smoothing], axis=0)
                smoothed_im.append(new_row)
            out_im = np.asarray(smoothed_im)
            
            return out_im
        
        SL1_io_im = smoother(SL1_io_im)
        SL2_io_im = smoother(SL2_io_im)
        
        SL1_io_im = np.where(SL1_io_im!=SL1_io_im, 0, SL1_io_im)
        SL2_io_im = np.where(SL2_io_im!=SL2_io_im, 0, SL2_io_im)
        row_image = SL1_io_im + SL2_io_im
        
        return row_image
    
    pix = np.arange(0, 128)
    
    row_image = get_row_image(image_data)
    
    total_io_image = row_image
    
    preserving_flatfield = np.where(flatfield!=flatfield, 1, flatfield)
    final_io_image = total_io_image / preserving_flatfield
    
    return final_io_image