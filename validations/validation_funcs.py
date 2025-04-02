import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u

from simladb import query, simladb, simladbX, DB_bcd, DB_shardpos, \
                    DB_judge1, DB_judge2, DB_bcdwise, scorners
from simla_variables import SimlaVar
from simla_utils import fmt_scorners, zoom_image, angular_separation

def make_stoplight(aorkey, j1cut, j2cut, chnlnum, save=None):

    wisepath = SimlaVar().wisepath

    rects = query(simladbX.select(*scorners, DB_judge1.BACKSUB_PHOT, DB_judge2.F_MEDIAN, DB_bcdwise.WISE_FILE)\
                 .where((DB_bcd.AORKEY==aorkey)&(DB_bcd.CHNLNUM==chnlnum)))
    corners, j1s, j2s, wisefile = \
        fmt_scorners(rects), rects['BACKSUB_PHOT'].to_numpy(), rects['F_MEDIAN'].to_numpy(), rects['WISE_FILE'][0]

    colors = np.ones_like(j1s)*np.nan
    colors = np.where((np.abs(j1s)>j1cut)&(np.abs(j2s)>j2cut), 'red', colors)
    colors = np.where((np.abs(j1s)<j1cut)&(np.abs(j2s)>j2cut), 'blue', colors)
    colors = np.where((np.abs(j1s)>j1cut)&(np.abs(j2s)<j2cut), 'yellow', colors)
    colors = np.where((np.abs(j1s)<j1cut)&(np.abs(j2s)<j2cut), 'green', colors)

    minra, avra, maxra = np.min(corners.T[0]), np.mean(corners.T[0]), np.max(corners.T[0])
    mindec, avdec, maxdec = np.min(corners.T[1]), np.mean(corners.T[1]), np.max(corners.T[1])

    buffer, wise_pixsize = 1.25, 1.375
    pix_sep = angular_separation([minra, mindec], [maxra, maxdec]) * (buffer/wise_pixsize) * 3600

    image_data, header, wcs = zoom_image([avra, avdec], pix_sep, wisepath+wisefile)

    def pixel_region(region):
        pr = []
        for p in region:
            sky_c = SkyCoord(p[0],p[1],unit='deg')
            pixel_p = astropy.wcs.utils.skycoord_to_pixel(sky_c, wcs)
            pr.append([pixel_p[0], pixel_p[1]])
        return pr

    plt.figure(figsize=(15, 15), dpi=80)
    ax = plt.subplot()
    for i in range(len(corners)):
        region = pixel_region(corners[i])
        color = colors[i]
        r = plt.Polygon(region, edgecolor=color, facecolor='none', ls='-', lw=1, zorder=20) 
        ax.add_patch(r)

    plt.annotate(str(aorkey), xy=(0.05, 0.9), xycoords='axes fraction', size=25, color='orange')
    plt.annotate('J1 cut='+str(j1cut), xy=(0.05, 0.85), xycoords='axes fraction', size=25, color='orange')
    plt.annotate('J2 cut='+str(j2cut), xy=(0.05, 0.8), xycoords='axes fraction', size=25, color='orange')

    plt.annotate('J1$\,$X J2$\,$X', xy=(0.05, 0.25), xycoords='axes fraction', size=25, color='red')
    plt.annotate('J1$\,$X J2$\,\checkmark$', xy=(0.05, 0.2), xycoords='axes fraction', size=25, color='yellow')
    plt.annotate('J1$\,\checkmark$ J2$\,$X', xy=(0.05, 0.15), xycoords='axes fraction', size=25, color='blue')
    plt.annotate('J1$\,\checkmark$ J2$\,\checkmark$', xy=(0.05, 0.1), xycoords='axes fraction', size=25, color='green')

    plt.imshow(np.log10(image_data), cmap='gray_r', origin='lower')
    
    if save is not None:
        plt.savefig(save, format='pdf')
    plt.show()

def full_slit_position(bcd):

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
        },
        'sh': {
            'width': 4.7 / 3600,
            'length': 11.3 / 3600,
        },
        'lh': {
            'width': 11.1 / 3600,
            'length': 22.3 / 3600,
        },
    }

    def make_rect(width, length, pos, slt_type):

        # create rectangle at origin
        rect = [[- (width/2),- (length/2)],
                [(width/2),- (length/2)],
                [(width/2),(length/2)],
                [- (width/2), (length/2)],
                [- (width/2),- (length/2)]]

        # rotate rectangle at origin and ranslate rectangle to position on sky in degrees
        slt_center = SkyCoord(pos[0], pos[1], unit='deg')
        rot_rect = []
        for p in rect:
            r = np.sqrt((p[0]**2)+(p[1]**2))
            a = np.arctan2(p[1], p[0])
            a = 90-np.degrees(a - np.radians(position_angle))
            newcoord = slt_center.directional_offset_by(a*u.deg, r*u.deg)
            rot_rect.append([newcoord.ra.degree, newcoord.dec.degree])

        class rect_obj:
            def __init__(self, rectangle, slt_type, center_pos):
                self.corners = rectangle[0:4]
                self.region = rectangle
                self.slt_type = slt_type
                self.center = [slt_center.ra.degree, slt_center.dec.degree]
                self.pa = position_angle
                self.file_name = bcd

        return rect_obj(rot_rect, slt_type, slt_center) 
    
    bcd_header = fits.open(bcd)[0].header
    fovname = bcd_header['FOVNAME']
    position_angle = bcd_header['PA_SLT']

    slt_pos = [bcd_header['RA_SLT'], bcd_header['DEC_SLT']]
    xslt_pos = None 

    rectangles = []
    if 'IRS_Short-Lo_1st' in fovname: 
        rectangles.append(make_rect(slt_info['sl1']['width'], slt_info['sl1']['length'], slt_pos, 'sl1'))
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.append(make_rect(slt_info['sl2']['width'], slt_info['sl2']['length'], xslt_pos, 'sl2'))
    elif 'IRS_Short-Lo_2nd' in fovname: 
        rectangles.append(make_rect(slt_info['sl2']['width'], slt_info['sl2']['length'], slt_pos, 'sl2'))
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.append(make_rect(slt_info['sl1']['width'], slt_info['sl1']['length'], xslt_pos, 'sl1'))
    elif 'IRS_Short-Lo_Module' in fovname:
        # distances from module center to slit center:
        SL1_dist = 0.01088423350029247 # degrees
        SL2_dist = 0.010884086370068513
        #
        big_rect_center = SkyCoord(bcd_header['RA_FOV'], bcd_header['DEC_FOV'], unit='deg')
        offset = big_rect_center.directional_offset_by(-1*(180-position_angle)*u.deg, SL1_dist*u.deg)
        slt_pos = [offset.ra.degree, offset.dec.degree]
        rectangles.append(make_rect(slt_info['sl1']['width'], slt_info['sl1']['length'], slt_pos, 'sl1'))
        xoffset = big_rect_center.directional_offset_by((position_angle)*u.deg, SL2_dist*u.deg)
        xslt_pos = [xoffset.ra.degree, xoffset.dec.degree]
        rectangles.append(make_rect(slt_info['sl2']['width'], slt_info['sl2']['length'], xslt_pos, 'sl2'))
    elif 'IRS_Long-Lo_1st' in fovname: 
        rectangles.append(make_rect(slt_info['ll1']['width'], slt_info['ll1']['length'], slt_pos, 'll1'))
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.append(make_rect(slt_info['ll2']['width'], slt_info['ll2']['length'], xslt_pos, 'll2'))
    elif 'IRS_Long-Lo_2nd' in fovname: 
        rectangles.append(make_rect(slt_info['ll2']['width'], slt_info['ll2']['length'], slt_pos, 'll2'))
        xslt_pos = [bcd_header['RA_XSLT'], bcd_header['DEC_XSLT']]
        rectangles.append(make_rect(slt_info['ll1']['width'], slt_info['ll1']['length'], xslt_pos, 'll1'))
    elif 'IRS_Long-Lo_Module' in fovname:
        # distances from module center to slit center:
        LL1_dist = 0.026609582651535315 # degrees
        LL2_dist = 0.026607916376313438
        #
        big_rect_center = SkyCoord(bcd_header['RA_FOV'], bcd_header['DEC_FOV'], unit='deg')
        offset = big_rect_center.directional_offset_by(-1*(180-position_angle)*u.deg, LL1_dist*u.deg)
        slt_pos = [offset.ra.degree, offset.dec.degree]
        rectangles.append(make_rect(slt_info['ll1']['width'], slt_info['ll1']['length'], slt_pos, 'll1'))
        xoffset = big_rect_center.directional_offset_by((position_angle)*u.deg, LL2_dist*u.deg)
        xslt_pos = [xoffset.ra.degree, xoffset.dec.degree]
        rectangles.append(make_rect(slt_info['ll2']['width'], slt_info['ll2']['length'], xslt_pos, 'll2'))
    elif 'IRS_Short-Hi' in fovname: 
        rectangles.append(make_rect(slt_info['sh']['width'], slt_info['sh']['length'], slt_pos, 'sh'))
    elif 'IRS_Long-Hi' in fovname: 
        rectangles.append(make_rect(slt_info['lh']['width'], slt_info['lh']['length'], slt_pos, 'lh'))
            
    return rectangles