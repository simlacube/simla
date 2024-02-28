#!/usr/bin/env python
# coding: utf-8

# ## has usableHeader, 
# ## it has an accurate distance between points (actually equidistant in angular distance), checks for edge cases, does an area check to make sure the aperture contains a significant amount of data

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import array
import math
from astropy.coordinates import SkyCoord, angular_separation
from astropy.wcs import WCS
import astropy.units as u
from astropy import wcs
from photutils.aperture import CircularAperture, SkyCircularAperture, ApertureStats
from photutils.aperture import aperture_photometry

from shapely.geometry.polygon import Polygon

from shapely.geometry import Point

from regions import Regions

from PyAstronomy import pyasl

def usableHeader(filename):
	'''
	I've had some trouble with headers and this function has the solution developed by Jess Sutter
	just give it a fits file name (string) and it will return a new header that PAHFIT likes
	'''
	hdulist = fits.open(filename)
	header = hdulist[0].header
	
	#recreate a usable header (thank you Jess)
	hdr = fits.Header()
	
	hdr['CRPIX1']=header['CRPIX1']
	hdr['CRPIX2']=header['CRPIX2']
	hdr['CDELT1']=header['CDELT1']
	hdr['CDELT2']=header['CDELT2']
	hdr['CRVAL1']=header['CRVAL1']
	hdr['CRVAL2']=header['CRVAL2']
	hdr['CTYPE1']=header['CTYPE1'].strip()
	hdr['CTYPE2']=header['CTYPE2'].strip()
	hdr['PC1_1']=header['PC1_1']
	hdr['PC2_1']=header['PC2_1']
	hdr['PC1_2']=header['PC1_2']
	hdr['PC2_2']=header['PC2_2']
	hdr['NAXIS1']=header['NAXIS1']
	hdr['NAXIS2']=header['NAXIS2']
	hdr['EQUINOX']=header['EQUINOX']
	
	
	world = WCS(hdr).celestial
	return world

def multiFileOverlap(filenamelist,radius):
	'''
	this function takes in a list of files for a single galaxy and an arcsec radius
	returns the portion of the hexagonal grid in wcs that appears in all the files and the radius of the aperture in wcs
	calls the hexGridPix function
	'''
	
	polygon = []
	colors = ['magenta','orange','limegreen','cornflowerblue'] ### for plotting 4 files

	#### for loop that gets the bounds and polygon of each fits file
	for k in range(len(filenamelist)): ### go through each file info
		if 'nan' in filenamelist[k]:
			pass
		else:
			hdulist = fits.open(filenamelist[k])
			header = hdulist[0].header

			### recreate a usable header (thank you Jess)

			hdr = fits.Header()
			hdr['CRPIX1']=header['CRPIX1']
			hdr['CRPIX2']=header['CRPIX2']
			hdr['CDELT1']=header['CDELT1']
			hdr['CDELT2']=header['CDELT2']
			hdr['CRVAL1']=header['CRVAL1']
			hdr['CRVAL2']=header['CRVAL2']
			hdr['CTYPE1']=header['CTYPE1'].strip()
			hdr['CTYPE2']=header['CTYPE2'].strip()
			hdr['PC1_1']=header['PC1_1']
			hdr['PC2_1']=header['PC2_1']
			hdr['PC1_2']=header['PC1_2']
			hdr['PC2_2']=header['PC2_2']
			hdr['NAXIS1']=header['NAXIS1']
			hdr['NAXIS2']=header['NAXIS2']
			hdr['EQUINOX']=header['EQUINOX']

			world = WCS(hdr).celestial

			xmaxpix = hdr['NAXIS1']
			ymaxpix = hdr['NAXIS2']
		
			### getting the general max and min bounds of the region
			worldup = world.pixel_to_world(xmaxpix,ymaxpix)
			worldleft = world.pixel_to_world(xmaxpix,0)
			worlddown = world.pixel_to_world(0,0)
			worldright = world.pixel_to_world(0,ymaxpix)
			#print(worldup.ra.value,worldleft.ra.value,worlddown.ra.value,worldright.ra.value)
		
			coords = ((worlddown.ra.value,worlddown.dec.value),(worldright.ra.value,worldright.dec.value),(worldup.ra.value,worldup.dec.value),(worldleft.ra.value,worldleft.dec.value),(worlddown.ra.value,worlddown.dec.value))
		
			polygon.append(Polygon(coords))
		
	### finding intersection of intersection with all the files; using shapely
	inter = polygon[0].intersection(polygon[1])
	
	for i in range(len(polygon)):
		inter = inter.intersection(polygon[i])
		###optional plotting of all files shapes
		#plt.plot(*polygon[i].exterior.xy, color = colors[i], alpha = 0.5)
	#plt.title('wcs overlap')
	#plt.legend()
	#plt.grid()
	#plt.show()	
	#plt.show()	
	
	### other optional plotting of overlap
	
	#plt.plot(*polygon[1].exterior.xy, color = 'red', alpha = 0.5)
	#plt.plot(*inter.exterior.xy, color = 'green', alpha = 0.5)
	#plt.show()
	
	### collecting the RA and DEC of the intersection of the files 
	declination = []
	rightac = []
	for i in range(len(inter.exterior.coords.xy[0])):
		declination.append(inter.exterior.coords.xy[1][i])
		rightac.append(inter.exterior.coords.xy[0][i])

	raddeg = radius/3600 ### convert from arcsec to degrees
	deccenters = []
	if len(declination) == 0:
		return [math.nan],[math.nan],math.nan,inter
	### this if/else statement is necessary because sometimes the dec is negative and so the condition about dec < decstop doesn't work	
	if np.mean(declination) > 0:
		dec = np.nanmin(declination) ### dec start
			
		decstop = np.nanmax(declination)

		while dec <= decstop:
			deccenters.append(dec)
			dec+=raddeg*np.sqrt(3/4) ###  pythagorean theorem of isosceles right triangle gives us sqrt(3/4); hypotenuse should be raddeg; short side is 1/2 raddeg
			
	else:
		dec = np.nanmax(declination) ### dec start
		decstop = np.nanmin(declination)

		while dec >= decstop:
			deccenters.append(dec)
			dec-=raddeg*np.sqrt(3/4)

	rastop = np.nanmax(rightac) ### ra is always positive (0 to 360)
	rav = []
	decv = [] ### for graphing purposes we need the dec per each ra so there is a list of repeating dec 
	
	for i in range(len(deccenters)): ### looping through declination because RA angular scale changes with dec by RA = ang/cos(dec)
		ra = np.nanmin(rightac) ### ra start
		while ra <= rastop*1.0005:
			if i % 2 == 1: ### odd ### we want every other dec row to be offset by raddeg/2 to make equilateral triangles 
				rav.append(ra)
				decv.append(deccenters[i])
			else: ### even
				rav.append(ra+raddeg/2/math.cos(deccenters[i]*math.pi/180)) ### the raddeg/2/cos handles the half radius offset 
				decv.append(deccenters[i])
			ra+=raddeg/math.cos(deccenters[i]*math.pi/180)
	
	
	### optional plotting of hex grid before any cuts
	#plt.scatter(rav,decv, alpha = 0.5,color = 'red')
	#plt.xlabel('RA (degrees)')
	#plt.ylabel('dec (degrees)')
	
	
	
	### checks if the aperture would extend beyond the edge of the overlap using shapely contains function
	for i in range(len(rav)):
		p = Point(rav[i],decv[i])
		if inter.contains(p):
			pass
		else:
			rav[i] = -999
	
	### need them as arrays to make the masking function work	
	rav = np.asarray(rav)
	decv = np.asarray(decv)
	
	### masking values that would extend beyond the edge
	maskeddecv = np.ma.masked_where(rav<-10, decv)
	maskedrav = np.ma.masked_where(rav<-10, rav)
	
	### optional plotting of masked hex grid with intersection 
	#plt.scatter(maskedrav,maskeddecv, alpha = 0.5, color = 'blue')
	#plt.plot(*inter.exterior.xy, color = 'green', alpha = 0.5)
	#plt.grid()
	#plt.show()
	
	return maskeddecv,maskedrav,raddeg*u.degree, inter

def multiFileApCut(filenamelist,radius): 
	'''
	this function takes the overlap of a list of files and determines which apertures contain sufficient data
	arguments: filenamelist (list of file names) and radius (radius of apertures in arcsecs)
	calls multiFileOverlap
	'''
	### get the overlap of the files in the list
	#### the worldgrids are the central coordinates of the evenly spaced apertures 
	#### inter is a polygon of the overlap of all files
	worldgriddec,worldgridra,rad,inter = multiFileOverlap(filenamelist,radius)
	
	### checking that the overlap has worked
	if np.isnan(rad):
		return math.nan,math.nan
	
	### making point objects for each aperture from the worldgrids to check in they are inside the polygon
	apple = []
	for i in range(len(worldgriddec)):
		if np.isnan(worldgriddec[i]) or not isinstance(worldgriddec[i],float): #### ignoring the nans and empty
			pass
		else: 
			### the buffer is essentially the radius of the point (rad is in degrees)
			apple.append(Point(worldgridra[i],worldgriddec[i]).buffer(rad.value))	
	
	### checks if there actually are apertures within the overlap
	if len(apple) == 0:
		return math.nan,math.nan				
	
	### the ra and dec values we actually want based on the area threshold test
	goodAp1ra = []
	goodAp1dec = []
	
	#### we can change this value based on how much we want each aperture to contain (0-1)
	threshold = 0.75
	
	### this value is what the sum of the aperture should be if it was fully in the image
	#### i think there was a rounding area with the buffer so i'm using the area of the circles i made
	#### i'm rounding because i ran into some issues with the areas
	##### the five is arbitrary
	goalArea = round(apple[0].area, 5) #rad**2*math.pi ### <- had some issues with rad**2*pi because of rounding or something 
	
	### optional plotting of the apertures labelled with numbers to help calculate the distances between apertures (further down in code)
	'''	
	for i in range(len(apple)):
		plt.plot(*apple[i].exterior.xy, alpha = 0.5, label = i)
	plt.scatter(worldgridra,worldgriddec, s = 20, color ='g')
	plt.plot(*inter.exterior.xy, color = 'green')
	plt.legend()	
	plt.show()
	'''
	### checking if the aperture (apple) contains a certain threshold of the image by calculating the overlap between the intersection of the fits files and the aperture	
	for i in range(len(apple)):
		ter = inter.intersection(apple[i])
		### optional plotting of intersection of apertures with intersection
		#plt.plot(*ter.exterior.xy, color = 'red')
		#plt.show()
		testarea = round(ter.area,5)
		if testarea >= threshold*goalArea:
			goodAp1ra.append(apple[i].centroid.x)
			goodAp1dec.append(apple[i].centroid.y)
		
	### continued optional plotting of the good apertures and the intersection
	#plt.scatter(goodAp1ra,goodAp1dec, s = 1000, alpha = 0.25,color ='g',edgecolor = 'b')
	#plt.plot(*inter.exterior.xy, color = 'green')
	#plt.show()
	
	### checking distances
	#### be careful with checking that these points are actually next to each other 
	'''
	### be careful here this version is cartesian not angular
	a = Point(apple[0].centroid.x,apple[0].centroid.y)
	b = Point(apple[1].centroid.x,apple[1].centroid.y)
	c = Point(apple[2].centroid.x,apple[2].centroid.y)
	d = Point(apple[3].centroid.x,apple[3].centroid.y)
	e = Point(apple[4].centroid.x,apple[4].centroid.y)
	
	print('zero','one',a.distance(b)*3600)
	print('zero','two',a.distance(c)*3600)
	print('one','two',b.distance(c)*3600)
	print('three','two',d.distance(c)*3600)
	print('one','four',b.distance(e)*3600)
	print('four','two',e.distance(c)*3600)
	print('zero','four',e.distance(a)*3600)
	'''
	'''
	### angular distances
	print('zero,one',pyasl.getAngDist(apple[0].centroid.x,apple[0].centroid.y,apple[1].centroid.x,apple[1].centroid.y)*3600)
	print('zero,two',pyasl.getAngDist(apple[0].centroid.x,apple[0].centroid.y,apple[2].centroid.x,apple[2].centroid.y)*3600)
	print('one,two',pyasl.getAngDist(apple[1].centroid.x,apple[1].centroid.y,apple[2].centroid.x,apple[2].centroid.y)*3600)
	print('three,two',pyasl.getAngDist(apple[3].centroid.x,apple[3].centroid.y,apple[2].centroid.x,apple[2].centroid.y)*3600)
	print('one,four',pyasl.getAngDist(apple[1].centroid.x,apple[1].centroid.y,apple[4].centroid.x,apple[4].centroid.y)*3600)
	print('four,two',pyasl.getAngDist(apple[4].centroid.x,apple[4].centroid.y,apple[2].centroid.x,apple[2].centroid.y)*3600)
	'''
	### checking that there are actually apertures that pass all the tests
	if len(goodAp1ra) == 0:
		return math.nan, math.nan

	return goodAp1ra,goodAp1dec

def makeDS9regions(ra,dec,rad):
	regions_str = '# Region file format: DS9\nimage\ncircle('+str(ra)+','+str(dec)+','+str(rad)+') # color=green\n'

	regions = Regions.parse(regions_str, format='ds9')

	print(regions)

	print(regions[0].visual)

