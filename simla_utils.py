'''Functions used throughout SIMLA'''

import numpy as np
import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord
from shapely import Polygon, Point
from photutils import background as bg
from photutils.background import BkgZoomInterpolator as bkginterp
from astropy.wcs import WCS
import astropy
from astropy.io import fits
import gc
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
import glob

from simla_variables import SimlaVar
simlavar = SimlaVar()
simlapath = simlavar.simlapath

def run_inputs_loader(inputfile):
    '''Interprets the txt file for full run inputs.'''
    out_dict = {}
    with open(inputfile) as ipdat:
        lines = ipdat.readlines()
        for i in lines:
            splitter = i.split('=')
            var, ip = splitter[0].replace(' ',''), splitter[1].replace(' ','')
            if var == 'run_name': ip = str(ip).replace('\n', '')
            elif var == 'desired_shard_depth': ip = int(ip)
            else: ip = float(ip)
            out_dict[var] = ip
    return out_dict

def DN_to_MJypsr(image, band):
    '''
    Convert a WISE image from native units to MJy/sr.
    
    image: (arr) a loaded WISE image
    band: (int) the WISE band number

    returns the converted image
    
    '''
    pix_area = ((2.75**2)*(2.3504e-11)) # steradians in a WISE pixel
                                        # WISE pixel size: https://wise2.ipac.caltech.edu/docs/release/allsky/
    
    conversions = { # see Table 1 from https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
        '1': 1.9350e-6,
        '2': 2.7048e-6,
        '3': 1.8326e-6,
        '4': 5.2269e-5,
    }
    conv = conversions[str(band)]
    Jy = image*conv
    MJypsr = ((Jy)/(10**6))/pix_area
    return MJypsr

def fmt_scorners(q_result):
    '''
    From a DB query result that contains shard corners,
    return a formatted shard corners list.
    '''
    ra, dec = \
        q_result.filter(like='_R').to_numpy().T, \
        q_result.filter(like='_D').to_numpy().T
    coords = np.asarray([ra, dec]).T
    return coords

def fmt_j2spec(q_result):
    '''
    From a DB query result that contains binned Judge2 spectra,
    return a formatted list.
    '''
    return q_result.filter(like='SPEC').to_numpy()

def make_wise_bg(image_data):
    '''
    Given a loaded (arr) WISE image, 
    return the simple local background in the same units.
    '''
    bg_2d = bg.Background2D(
        image_data, 
        box_size=(2000,2000), 
        exclude_percentile=90, 
        interpolator=bkginterp(order=1)
        ).background.astype('single')
    return bg_2d

def wise_filename_to_coords(wfile):
    '''
    Return the approx. central coords of a WISE image
    from the info contained in the WISE file name.

    Returns (RA, DEC) in decimal degrees.
    '''
    # the CRVAL is the center pixel for WISE
    # the approx. coords of the center is in the filename
    coordpart = wfile.split(SimlaVar().wisepath)[-1].split('/')[2]
    ra = float(coordpart[0:4])/10
    dec_sign = coordpart[4]
    dec = float(coordpart.split('_')[0][5:])/10
    if dec_sign == 'm':
        dec = -dec
    return ra, dec

def angular_separation(coords1, coords2):
    '''
    Return the angular separation in degrees between a pair of coordinates.
    give coordsX as (RA, DEC) in decimal degrees.
    '''
    # coords are (RA, DEC) in decimal degrees
    # returns the separation in degrees
    coords1, coords2 = np.radians(coords1), np.radians(coords2)
    sep = np.arccos(
        (np.sin(coords1[1])*np.sin(coords2[1])) + \
        (np.cos(coords1[1])*np.cos(coords2[1])*np.cos(coords1[0]-coords2[0]))
    )
    return np.degrees(sep)

def zoom_image(zoom_center, zoom_size, imagefile):

    '''
    Astropy wrapper for zooming-in on FITS images

    zoom_center (list): (RA, DEC) in decimal degrees that will be the center of the new image
    zoom_size (int): the edge length in pixels of the new cutout image
    imagefile (str): the path for the FITS image

    returns: [zoomed image, new header, new WCS object]

    '''
    
    image_data = fits.open(imagefile)[0].data
    image_header = fits.open(imagefile)[0].header
    wcs = WCS(image_header)

    zoom_center = SkyCoord(zoom_center[0], zoom_center[1], unit='deg')
    crop_fits = Cutout2D(image_data, astropy.wcs.utils.skycoord_to_pixel(zoom_center,wcs), zoom_size, wcs=wcs)
    wcs = crop_fits.wcs
    image_data = crop_fits.data
    image_header.update(crop_fits.wcs.to_header()) 
    
    return image_data, image_header, wcs

def load_superdark_sets(sd_dir):
    '''
    Load in a library of superdarks to make tailored superdark interpolation more efficient.
    See superdarks/make_superdarks.py and superdarks/make_tailored_superdarks.py.

    sd_dir is the directory that the zodibinned superdarks are contained in.

    returns a dictionary like

    {SL_<RAMPTIME>:
        {'set': array of superdarks for each zodi bin,
         'depth': array of the number of darks that contribute to each pixel in a superdark,
         'fiducial_zodis': array of 12um zodi values that are the center of each zodi bin},
    {SL_<OTHER_RAMPTIME>:...}
    }

    '''
    # If a zodi bin is missing, we just interpolate over it
    superdark_sets = {}
    for ramp in simlavar.sl_ramptimes:
        ramp_superdarks = sorted(glob.glob(sd_dir+'superdarks/*SL*'+str(ramp)+'*'))
        fiducial_zodis = np.asarray(sorted([float(i.split('fidzodi-')[-1].split('_')[0].split('.npy')[0]) for i in ramp_superdarks]))
        superdark_sets['SL_'+str(ramp)] = {'set': np.asarray([np.load(i)[0] for i in ramp_superdarks]),
                                           'depth': np.asarray([np.load(i)[1] for i in ramp_superdarks]),
                                           'fiducial_zodis': fiducial_zodis}
        
    for ramp in simlavar.ll_ramptimes:
        ramp_superdarks = sorted(glob.glob(sd_dir+'superdarks/*LL_*'+str(ramp)+'*'))
        fiducial_zodis = np.asarray(sorted([float(i.split('fidzodi-')[-1].split('_')[0].split('.npy')[0]) for i in ramp_superdarks]))
        superdark_sets['LL_'+str(ramp)] = {'set': np.asarray([np.load(i)[0] for i in ramp_superdarks]),
                                           'depth': np.asarray([np.load(i)[1] for i in ramp_superdarks]),
                                           'fiducial_zodis': fiducial_zodis}
    
    for ramp in simlavar.ll_ramptimes:
        ramp_superdarks = sorted(glob.glob(sd_dir+'superdarks/*LLa*'+str(ramp)+'*'))
        fiducial_zodis = np.asarray(sorted([float(i.split('fidzodi-')[-1].split('_')[0].split('.npy')[0]) for i in ramp_superdarks]))
        superdark_sets['LLa_'+str(ramp)] = {'set': np.asarray([np.load(i)[0] for i in ramp_superdarks]),
                                            'depth': np.asarray([np.load(i)[1] for i in ramp_superdarks]),
                                            'fiducial_zodis': fiducial_zodis}
    return superdark_sets

def interp_superdark(z, superdark_set, return_set=False):
    '''
    Interpolator function for tailored superdarks, see superdarks/make_tailored_superdarks.py

    z: (float) the 12um zodi value for an AOR
    superdark_set: the output of load_superdark_sets()
    if return_set is False, returns the interpolated "tailored" superdark
    if return_set is True, returns [interpolated "tailored" superdark, 
                                    low-zodi superdark used for interpolation,
                                    high-zodi superdark used for interpolation]
    
    '''
    fiducial_zodis = superdark_set['fiducial_zodis']
    superdark_set = superdark_set['set']

    # If input zodi is outside of the interpolation range, just use whatever one is closest
    if z < np.min(fiducial_zodis): z = np.min(fiducial_zodis) + 10**-10
    if z > np.max(fiducial_zodis): z = np.max(fiducial_zodis) - 10**-10

    z_low = np.max(fiducial_zodis[fiducial_zodis<=z])
    z_high = np.min(fiducial_zodis[fiducial_zodis>=z])

    z_diff = z_high - z_low
    low_weight = (z_high - z)/z_diff
    high_weight = (z - z_low)/z_diff
    low_im = superdark_set[fiducial_zodis==z_low][0]
    high_im = superdark_set[fiducial_zodis==z_high][0]
    out_image = np.average((low_im, high_im), axis=0, weights=(low_weight, high_weight))
        
    if not return_set: return out_image
    elif return_set: return out_image, low_im, high_im

def photometry(region, image_data, image_header):

    '''
    Get the average pixel value of an image within a set of input sky coordinates.
    Pixels are clipped exactly.

    region: (arr) a list of [[RA, DEC], [RA, DEC], ...] in decimal degrees for the aperture
    image_data: (arr) the loaded image data from a FITS file
    image_header: the loaded header object from a FITS file

    returns the average pixel value in the same units as image_data

    '''
    
    wcs = WCS(image_header)
    
    image_size = len(image_data)

    pixel_region = [astropy.wcs.utils.skycoord_to_pixel(SkyCoord(p[0],p[1],unit='deg'),wcs) for p in region]
    region_polygon = Polygon(pixel_region)

    # Constrain the clipping to a box around the input region
    xs = [i[0] for i in pixel_region]
    ys = [i[1] for i in pixel_region]
    maxx = int(np.ceil(np.max(xs) + 1))
    minx = int(np.floor(np.min(xs) - 1))
    maxy = int(np.ceil(np.max(ys) + 1))
    miny = int(np.floor(np.min(ys) - 1))
    
    def count_pixel(x, y):

        # Given input pixel coordinates, return the fraction of the pixel
        # value and area that contributes to the region
        
        s = 0.5
        x0, y0 = x - s, y - s
        x1, y1 = x - s, y + s
        x2, y2 = x + s, y + s
        x3, y3 = x + s, y - s
        
        pixel_polygon = Polygon([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
        pixel_overlap = region_polygon.intersection(pixel_polygon).area
        normalized_overlap = pixel_overlap / pixel_polygon.area
        pixel_value = image_data[y, x]*normalized_overlap
        
        return pixel_value, normalized_overlap
    
    pixel_count = 0
    flux = 0
    
    coords_to_check = [[x, y]
                      for x in np.arange(minx, maxx) if 0 <= x <= image_size
                      for y in np.arange(miny, maxy) if 0 <= y <= image_size]

    # Loop over the pixels in the region, compute the fraction of the pixel inside,
    # and add to the overal flux in the region
    for i in coords_to_check:
        try:
            x, y = i[0], i[1]
            p = Point(x, y)
            if region_polygon.exterior.distance(p) >= np.sqrt(2)/2 and region_polygon.contains(p):
                if image_data[y, x] == image_data[y, x]: # handle NaNs
                    pixel_count += 1.0
                    flux += image_data[y, x]
            elif region_polygon.exterior.distance(p) <= np.sqrt(2)/2:
                pixel_value, normalized_overlap = count_pixel(x, y)
                if pixel_value == pixel_value: # handle NaNs
                    pixel_count += normalized_overlap
                    flux += pixel_value
        except IndexError: pass # handles regions larger than the image

    # Get the average
    if pixel_count == 0: pixel_count = np.nan
    average_pixel_value = flux/pixel_count
    
    if average_pixel_value != average_pixel_value:
        average_pixel_value = 0
            
    return average_pixel_value

def shard_corners(bcd_filename, edgetrim, n_shards):

    '''
    For an input IRS BCD, get the sky coordinates of spatially-distinct "shards".
    Sky coordinates take into account the edge trimming (see trim_and_shard_masks.py)

    bcd_filename: (str) the file name for the BCD
    edgetrim: (float) the % of the width of a suborder that is trimmed
    n_shards: (int) the number of shards per suborder

    returns a list of three lists, each having a length of n_shards:
        sky corners, suborders, shard ids
        i.e. [[shard0 sky corners, shard1 sky corners, ...], [1,1,1,1,2,...], [0,1,2,3,0,1,...]]

    '''

    slt_info = { # width and length in degrees.
        'sl1': {
            'width': 3.7 / 3600,
            'length': 57 / 3600, 
        },
        'sl2': {
            'width': 3.6 / 3600,
            'length': 57 / 3600,
        },
        'll1': {
            'width': 10.7 / 3600,
            'length': 168 / 3600,
        },
        'll2': {
            'width': 10.5 / 3600,
            'length': 168 / 3600,
        }
    }

    def make_rect(width, length, pos, slt_type):

        shards = []
        length = length * (1-((edgetrim*1)/100))

        for i in range(n_shards):
    
            # create rectangle at origin
            rect = [[- (width/2),- (length/2/n_shards)],
                    [(width/2),- (length/2/n_shards)],
                    [(width/2),(length/2/n_shards)],
                    [- (width/2), (length/2/n_shards)]]
    
            fullslit_center = SkyCoord(pos[0], pos[1], unit='deg')
            offset_angle = np.degrees(np.radians(position_angle))
            offset_distance = (length/2) - (2*i+1)*(length/(2*n_shards))
            shard_center = fullslit_center.directional_offset_by(offset_angle*u.deg, offset_distance*u.deg)
    
            # rotate rectangle at origin and translate rectangle to position on sky in degrees
            rot_rect = []
            for p in rect:
                r = np.sqrt((p[0]**2)+(p[1]**2))
                a = np.arctan2(p[1], p[0])
                a = 90-np.degrees(a - np.radians(position_angle))
                newcoord = shard_center.directional_offset_by(a*u.deg, r*u.deg)
                rot_rect.append([newcoord.ra.degree, newcoord.dec.degree])

            shards.append(rot_rect)

        return shards
    
    bcd_header = fits.open(bcd_filename)[0].header
    fovname = bcd_header['FOVNAME']
    position_angle = bcd_header['PA_SLT']

    slt_pos = [bcd_header['RA_SLT'], bcd_header['DEC_SLT']]
    xslt_pos = None 

    rectangles = []
    suborders = []
    shard_ids = [i for i in range(n_shards)][::-1]+[j for j in range(n_shards)][::-1]

    # We have to treat different FOVs differently. This handles each case.
    if 'IRS_Short-Lo_1st' in fovname: 
        rectangles.extend(make_rect(slt_info['sl1']['width'], slt_info['sl1']['length'], slt_pos, 'sl1'))
        suborders.extend([1 for i in range(n_shards)])
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.extend(make_rect(slt_info['sl2']['width'], slt_info['sl2']['length'], xslt_pos, 'sl2'))
        suborders.extend([2 for i in range(n_shards)])
    elif 'IRS_Short-Lo_2nd' in fovname: 
        rectangles.extend(make_rect(slt_info['sl2']['width'], slt_info['sl2']['length'], slt_pos, 'sl2'))
        suborders.extend([2 for i in range(n_shards)])
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.extend(make_rect(slt_info['sl1']['width'], slt_info['sl1']['length'], xslt_pos, 'sl1'))
        suborders.extend([1 for i in range(n_shards)])
    elif 'IRS_Short-Lo_Module' in fovname:
        # distances from module center to slit center:
        SL1_dist = 0.01088423350029247 # degrees
        SL2_dist = 0.010884086370068513
        #
        big_rect_center = SkyCoord(bcd_header['RA_FOV'], bcd_header['DEC_FOV'], unit='deg')
        offset = big_rect_center.directional_offset_by(-1*(180-position_angle)*u.deg, SL1_dist*u.deg)
        slt_pos = [offset.ra.degree, offset.dec.degree]
        rectangles.extend(make_rect(slt_info['sl1']['width'], slt_info['sl1']['length'], slt_pos, 'sl1'))
        suborders.extend([1 for i in range(n_shards)])
        xoffset = big_rect_center.directional_offset_by((position_angle)*u.deg, SL2_dist*u.deg)
        xslt_pos = [xoffset.ra.degree, xoffset.dec.degree]
        rectangles.extend(make_rect(slt_info['sl2']['width'], slt_info['sl2']['length'], xslt_pos, 'sl2'))
        suborders.extend([2 for i in range(n_shards)])
        
    elif 'IRS_Long-Lo_1st' in fovname: 
        rectangles.extend(make_rect(slt_info['ll1']['width'], slt_info['ll1']['length'], slt_pos, 'll1'))
        suborders.extend([1 for i in range(n_shards)])
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.extend(make_rect(slt_info['ll2']['width'], slt_info['ll2']['length'], xslt_pos, 'll2'))
        suborders.extend([2 for i in range(n_shards)])
    elif 'IRS_Long-Lo_2nd' in fovname: 
        rectangles.extend(make_rect(slt_info['ll2']['width'], slt_info['ll2']['length'], slt_pos, 'll2'))
        suborders.extend([2 for i in range(n_shards)])
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.extend(make_rect(slt_info['ll1']['width'], slt_info['ll1']['length'], xslt_pos, 'll1'))
        suborders.extend([1 for i in range(n_shards)])
    elif 'IRS_Long-Lo_Module' in fovname:
        # distances from module center to slit center:
        LL1_dist = 0.026609582651535315 # degrees
        LL2_dist = 0.026607916376313438
        #
        big_rect_center = SkyCoord(bcd_header['RA_FOV'], bcd_header['DEC_FOV'], unit='deg')
        offset = big_rect_center.directional_offset_by(-1*(180-position_angle)*u.deg, LL1_dist*u.deg)
        slt_pos = [offset.ra.degree, offset.dec.degree]
        rectangles.extend(make_rect(slt_info['ll1']['width'], slt_info['ll1']['length'], slt_pos, 'll1'))
        suborders.extend([1 for i in range(n_shards)])
        xoffset = big_rect_center.directional_offset_by((position_angle)*u.deg, LL2_dist*u.deg)
        xslt_pos = [xoffset.ra.degree, xoffset.dec.degree]
        rectangles.extend(make_rect(slt_info['ll2']['width'], slt_info['ll2']['length'], xslt_pos, 'll2'))
        suborders.extend([2 for i in range(n_shards)])
  
    return rectangles, suborders, shard_ids

class bcd_spectrum:

    '''Object class for extracting spectra directly from a BCD. Fullslits and shards are supported.'''
    
    def __init__(self, order_limits='strict'):

        '''
        Initialize the extractor. If extractor = bcd_spectrum() is called, all necessary 
        calibration files will be pre-loaded and a large number of spectrum extractions and be done very quickly.
        order_limits are options for trimming the ends of spectral orders.
        '''

        if order_limits == 'strict':
            order_limits = {
                'sl1': (7.63660, 14.59222),
                'sl2': (5.5, 7),
                'sl3': (7.33419, 8.42289),
                'll1': (22, 33),
                'll2': (15, 21.9),
                'll3': (19.40400, 21.26689)
            }
        elif order_limits == 'full':
            order_limits = {
                'sl1': (0, 50),
                'sl2': (0, 50),
                'sl3': (0, 50),
                'll1': (0, 50),
                'll2': (0, 50),
                'll3': (0, 50)
            }
        
        wavsamp_file = [
            simlapath+'calib/wavsamp/irs_b0_WAVSAMPv6.tbl',
            simlapath+'calib/wavsamp/irs_b1_WAVSAMPv9.tbl',
            simlapath+'calib/wavsamp/irs_b2_WAVSAMPv10.tbl',
            simlapath+'calib/wavsamp/irs_b3_WAVSAMPv7.tbl',
        ]
        
        def process_wavsamp(file):

            f = open(file, 'r')
            l = f.readlines()
            for line in l:
                if 'int' in line:
                    startline = l.index(line)+1
            f.close()
            wavsamp_data = np.genfromtxt(file, skip_header=startline).tolist()

            if 'b0' in file: 
                ol1 = order_limits['sl1']
                ol2 = order_limits['sl2']
                ol3 = order_limits['sl3']
            elif 'b1' in file: return None
            elif 'b2' in file: 
                ol1 = order_limits['ll1']
                ol2 = order_limits['ll2']
                ol3 = order_limits['ll3']
            elif 'b3' in file: return None

            included1 = []
            for i in range(len(wavsamp_data)):
                lam = wavsamp_data[i][3]
                if ol1[0] <= lam <= ol1[1] and \
                wavsamp_data[i][0] == 1:
                    included1.append(i)
            wavelengths1 = [i[3] for i in wavsamp_data]
            spec1 = []
            for i in included1:
                spec1.append([wavelengths1[i], i])
            sortspec1 = sorted(spec1)
            wavelengths1 = np.asarray([i[0] for i in sortspec1])
            included1 = np.asarray([i[1] for i in sortspec1])

            included2 = []
            for i in range(len(wavsamp_data)):
                lam = wavsamp_data[i][3]
                if ol2[0] <= lam <= ol2[1] and \
                wavsamp_data[i][0] == 2:
                    included2.append(i)
            wavelengths2 = [i[3] for i in wavsamp_data]
            spec2 = []
            for i in included2:
                spec2.append([wavelengths2[i], i])
            sortspec2 = sorted(spec2)
            wavelengths2 = np.asarray([i[0] for i in sortspec2])
            included2 = np.asarray([i[1] for i in sortspec2])
            
            included3 = []
            for i in range(len(wavsamp_data)):
                lam = wavsamp_data[i][3]
                if ol3[0] <= lam <= ol3[1] and \
                wavsamp_data[i][0] == 3:
                    included3.append(i)
            wavelengths3 = [i[3] for i in wavsamp_data]
            spec3 = []
            for i in included3:
                spec3.append([wavelengths3[i], i])
            sortspec3 = sorted(spec3)
            wavelengths3 = np.asarray([i[0] for i in sortspec3])
            included3 = np.asarray([i[1] for i in sortspec3])

            return [[wavelengths1, included1], [wavelengths2, included2], [wavelengths3, included3]]
        
        wavelength_data = [process_wavsamp(file) for file in wavsamp_file]

        # contribution_cubes = [ # always want to use latest version
        #     np.load(simlapath+'calib/contribution_cubes/cc_b0v6.npy'),
        #     np.load(simlapath+'calib/contribution_cubes/cc_b1v9.npy'),
        #     np.load(simlapath+'calib/contribution_cubes/cc_b2v10.npy'),
        #     np.load(simlapath+'calib/contribution_cubes/cc_b3v7.npy'),
        # ]
        # No hires, the cubes are too big for GH
        contribution_cubes = [ # always want to use latest version
            np.load(simlapath+'calib/contribution_cubes/cc_b0v6.npy'),
            None,
            np.load(simlapath+'calib/contribution_cubes/cc_b2v10.npy'),
            None,
        ]
        contribution_cubes = [np.where(i==0, np.nan, i) for i in contribution_cubes]

        conversion_bcds = [
            fits.getdata(simlapath+'calib/conversion_frames/sl_conversion.fits'),
            fits.getdata(simlapath+'calib/conversion_frames/sh_conversion.fits'),
            fits.getdata(simlapath+'calib/conversion_frames/ll_conversion.fits'),
            fits.getdata(simlapath+'calib/conversion_frames/lla_conversion.fits'),
            fits.getdata(simlapath+'calib/conversion_frames/lh_conversion.fits'),
            fits.getdata(simlapath+'calib/conversion_frames/lha_conversion.fits')
        ]

        zero_subslit_masks = [
            [np.load(simlapath+'calib/shard_masks/SL1.npy'), 
             np.load(simlapath+'calib/shard_masks/SL2.npy'),
             np.load(simlapath+'calib/shard_masks/SL3.npy')],
            None,
            [np.load(simlapath+'calib/shard_masks/LL1.npy'), 
             np.load(simlapath+'calib/shard_masks/LL2.npy'),
             np.load(simlapath+'calib/shard_masks/LL3.npy')],
            None
        ]
        subslit_masks = [
            [[np.where(i==0, np.nan, i) for i in zero_subslit_masks[0][0]],
             [np.where(i==0, np.nan, i) for i in zero_subslit_masks[0][1]],
             [np.where(i==0, np.nan, i) for i in zero_subslit_masks[0][2]]],
            None,
            [[np.where(i==0, np.nan, i) for i in zero_subslit_masks[2][0]],
             [np.where(i==0, np.nan, i) for i in zero_subslit_masks[2][1]],
             [np.where(i==0, np.nan, i) for i in zero_subslit_masks[2][2]]],
            None
        ]

        pixel_counts = [
            [[np.nansum(np.nansum(contribution_cubes[0]*i, axis=1), axis=1) for i in subslit_masks[0][0]],
             [np.nansum(np.nansum(contribution_cubes[0]*i, axis=1), axis=1) for i in subslit_masks[0][1]],
             [np.nansum(np.nansum(contribution_cubes[0]*i, axis=1), axis=1) for i in subslit_masks[0][2]]],
            None,
            [[np.nansum(np.nansum(contribution_cubes[2]*i, axis=1), axis=1) for i in subslit_masks[2][0]],
             [np.nansum(np.nansum(contribution_cubes[2]*i, axis=1), axis=1) for i in subslit_masks[2][1]],
             [np.nansum(np.nansum(contribution_cubes[2]*i, axis=1), axis=1) for i in subslit_masks[2][2]]],
            None,
        ]
        
        zero_fullslit_masks = [[np.load(simlapath+'calib/trimmed_fullslit_masks/SL1.npy'), 
                                np.load(simlapath+'calib/trimmed_fullslit_masks/SL2.npy'), 
                                np.load(simlapath+'calib/trimmed_fullslit_masks/SL3.npy')],
                               [None, None],
                               [np.load(simlapath+'calib/trimmed_fullslit_masks/LL1.npy'), 
                                np.load(simlapath+'calib/trimmed_fullslit_masks/LL2.npy'), 
                                np.load(simlapath+'calib/trimmed_fullslit_masks/LL3.npy')],
                               [None, None]]
        fullslit_masks = [[np.where(zero_fullslit_masks[0][0]==0, np.nan, zero_fullslit_masks[0][0]), 
                           np.where(zero_fullslit_masks[0][1]==0, np.nan, zero_fullslit_masks[0][1]), 
                           np.where(zero_fullslit_masks[0][2]==0, np.nan, zero_fullslit_masks[0][2])],
                          [None, None],
                          [np.where(zero_fullslit_masks[2][0]==0, np.nan, zero_fullslit_masks[2][0]), 
                           np.where(zero_fullslit_masks[2][1]==0, np.nan, zero_fullslit_masks[2][1]), 
                           np.where(zero_fullslit_masks[2][2]==0, np.nan, zero_fullslit_masks[2][2])],
                          [None, None]]
        
        fullslit_pixel_counts = [[np.nansum(np.nansum(contribution_cubes[0]*fullslit_masks[0][0], axis=1), axis=1), 
                                  np.nansum(np.nansum(contribution_cubes[0]*fullslit_masks[0][1], axis=1), axis=1), 
                                  np.nansum(np.nansum(contribution_cubes[0]*fullslit_masks[0][2], axis=1), axis=1)],
                                 [None, None],
                                 [np.nansum(np.nansum(contribution_cubes[2]*fullslit_masks[2][0], axis=1), axis=1), 
                                  np.nansum(np.nansum(contribution_cubes[2]*fullslit_masks[2][1], axis=1), axis=1), 
                                  np.nansum(np.nansum(contribution_cubes[2]*fullslit_masks[2][2], axis=1), axis=1)],
                                 [None, None]]
        
        self.contribution_cubes = contribution_cubes
        self.conversion_bcds = conversion_bcds
        self.subslit_masks = subslit_masks
        self.zero_subslit_masks = zero_subslit_masks
        self.fullslit_masks = fullslit_masks
        self.wavelength_data = wavelength_data
        self.pixel_counts = pixel_counts
        self.zero_fullslit_masks = zero_fullslit_masks
        self.fullslit_masks = fullslit_masks
        self.fullslit_pixel_counts = fullslit_pixel_counts
        
    def conv_select(self, header):
        # Return the appropriate conversion frame
        if header['CHNLNUM'] < 2:
            return self.conversion_bcds[header['CHNLNUM']]
        elif header['CHNLNUM'] == 2:
            if header['MJD_OBS'] < 54403.0: return self.conversion_bcds[2]
            elif header['MJD_OBS'] >= 54403.0: return self.conversion_bcds[3]

    def subslit_bcd_spectrum(self, image_data, header, suborder_num, subslit_num):

        '''
        Extract the spectrum from a shard on a BCD.

        image_data: (arr) the loaded data from the IRS BCD
        header: the loaded header from the IRS BCD
        suborder_num: (int) desired suborder 1, 2, or 3
        subslit_num: (int) shard id to extract

        returns: wavelengths (arr, um), fluxes (arr, MJy/sr)

        '''
        
        chnlnum = header['CHNLNUM']
        suborder_num -= 1

        subslit_mask = self.subslit_masks[chnlnum][suborder_num][subslit_num]

        conv = self.conv_select(header)

        image_data = image_data*conv
        image_data = image_data*subslit_mask

        contribution_cube = self.contribution_cubes[chnlnum]*subslit_mask

        wavelengths = self.wavelength_data[chnlnum][suborder_num][0]
        included = self.wavelength_data[chnlnum][suborder_num][1]

        f_im = contribution_cube*image_data
        flux = np.nansum(np.nansum(f_im, axis=1), axis=1)

        pixel_count = self.pixel_counts[chnlnum][suborder_num][subslit_num]

        fluxes = (flux / pixel_count)[included]

        final_mask = np.where(fluxes==fluxes)

        return wavelengths[final_mask], fluxes[final_mask]
    
    def fullslit_bcd_spectrum(self, image_data, header, suborder_num):

        '''
        Extract the spectrum from a full slit on a BCD.

        image_data: (arr) the loaded data from the IRS BCD
        header: the loaded header from the IRS BCD
        suborder_num: (int) desired suborder 1, 2, or 3

        returns: wavelengths (arr, um), fluxes (arr, MJy/sr)

        '''
        
        chnlnum = header['CHNLNUM']
        suborder_num -= 1

        fullslit_mask = self.fullslit_masks[chnlnum][suborder_num]

        conv = self.conv_select(header)

        image_data = image_data*conv
        image_data = image_data*fullslit_mask

        contribution_cube = self.contribution_cubes[chnlnum]*fullslit_mask

        wavelengths = self.wavelength_data[chnlnum][suborder_num][0]
        included = self.wavelength_data[chnlnum][suborder_num][1]

        f_im = contribution_cube*image_data
        flux = np.nansum(np.nansum(f_im, axis=1), axis=1)

        pixel_count = self.fullslit_pixel_counts[chnlnum][suborder_num]

        fluxes = (flux / pixel_count)[included]

        final_mask = np.where(fluxes==fluxes)

        return wavelengths[final_mask], fluxes[final_mask]