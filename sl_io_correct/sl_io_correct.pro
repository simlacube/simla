;
; NAME:  
;
;    	SL_IO_correct
;
; CONTACT:
;
;    	carl.starkey@utoledo.edu
;
; DESCRIPTION:
;    
;    	A routine for analyzing and correcting for the off-order light found
;		between the short-low orders utilizing a box-car average in row and
;		BCD space. This routine is intended for use with the CUBISM package 
;		by Smith, J. D. T., et al. 
;
;		Note: This program utilizes background subtracted data. If the cube 
;				located at cube_location does not have a background OR if the 
;				background records differ between the SL1 and SL2 cubes, results
;				may be unpredictable. 
;
;		Note: If save_location is set, this program creates a 3D fits image  
;				without altering the original BCDs. The 3D fits image can be loaded
;				into cubism and used just like any other cubism project, except that
;				it will be in a read-only state. Additional bad-pixel masking or 
;				other similar changes will be impossible, thus it is recommended
;				that SL_IO_correct be the last change made to a dataset.
;
;		Note: Due to the boxcar average method used in the sigma-trimmed
;				average, the ends of the bcd/row sub-cube will have been created
;				using an identical subset of pixels. This can result in a lack of 
;				information regarding edge-effects, or carrying said effects
;				much farther in BCD space than is reasonable, particularly for large
;				values of bcd/row_average_range. As a potential alternative, see
;				the TRIM_EDGES keyword.
;
; CALLING SEQUENCE:
;
;    	SL_IO_correct, cube, [ save_location ],
;    	               BCD_AVERAGE_RANGE=, ROW_AVERAGE_RANGE=
;
; INPUT PARAMETERS:
;
;    	cube - The filepath to the *.cpj file to be
;              processed, or a cube object itself.
;
; OPTIONAL INPUT PARAMETERS:
;
;       save_location - The filepath to save the corrected cube fits file. 
;
;
; OPTIONAL KEYWORDS
;
;		BCD_AVERAGE_RANGE - The number of BCDs that will be utilized in the boxcar
;									average. Default: 5
;		ROW_AVERAGE_RANGE - The number of rows that will be utilized in the boxcar
;									average. Default: 5	
;	 	VISUALIZE - Display the different average flux found in two different
;						subregions of the inter-order light column. If set, the 
;						cube will NOT be built or saved. The green lines on the graph
;						and inset image represent AOR boundaries. The lines on the 
;						image are a different shade to emphasize that the image may
;						not be perfectly aligned with the graph. Note that this
;						keyword requires the Coyote Graphics System by David Fanning
;						be part of your IDL path. 
;		TRIM_EDGES - This keyword causes the boxcar average method to shrink the
;						averaging range to 1/2 the normal range as one approaches the
;						edge of an AOR or row. Note that for exceptionally large 
;						average ranges, this can result in the number of bcds/rows 
;						contributing to the average constantly increasing until 
;						halfway through the AOR, reaching bcd_average_range, 
;						at which point it will begin decreasing as the boxcar 
;						approaches the end of the AOR.
;		QUIET 	  - Silences sl_io_correct from reporting its progress.
;
; OUTPUTS:
;
;   	A corrected cubism project file, saved to disk.
;
; EXAMPLE:
;
;
;    SL_IO_correct, '/home/user/proj/cubes/sl1.cpj', $
;                   '/home/user/proj/cubes/sl1_iocorected.fits', $
;                    bcd_average_range = 7, row_average_range=10
;    
; MODIFICATION HISTORY:
;
;    2011-03 (Carl Starkey): Written.
;	  2011-04-06 (Carl Starkey): Generalized for more general cubes.
;	  2011-04-07 (Carl Starkey): Continued tinkering to appease those not 
;											conforming to original project protocols. 
;	  2011:04-08 (Carl Starkey): Error fixes and visualization capability.
;	  2011:04-12 (Carl Starkey): User error catching.
;	  2011:04-13 (Carl Starkey): Visualization code updated with D. Fanning's 
;										 	graphics utilities for device independence
;	  2011:04-14 (Carl Starkey): Assumed the cmask and flatfield will be in the
;											same directory as sl_io_correct.pro and added
;											appropriate errors if it is not.
;	  2011-04-18 (Carl Starkey): Allow users to hand this routine a cube rather 
;											than the location of one saved on disk. More
;											error catching. Now saves output as a 3D fits
;											file.

pro sl_io_correct, cube_location, save_location, $
	BCD_AVERAGE_RANGE = BCD_AVERAGE_RANGE, ROW_AVERAGE_RANGE = ROW_AVERAGE_RANGE, $
	VISUALIZE = VISUALIZE, TRIM_EDGES = TRIM_EDGES, QUIET=QUIET

	if obj_class(cube_location) eq 'CUBEPROJ' then begin
		cube = cube_location
		if STREGEX(cube->ProjectName(), '\(IO_COR\)', /EXTRACT) eq '(IO_COR)' $
		then begin
			print, '% SL_IO_CORRECT: *************************ERROR*************************'
			print, '% SL_IO_CORRECT: ************CUBE HAS ALREADY BEEN PROCESSED************'
			PRINT, '% SL_IO_CORRECT: *******************WITH SL_IO_CORRECT******************'
			PRINT, '% SL_IO_CORRECT: ***********************QUITTING************************'
			return					
		endif
	endif else begin
		cube_file = file_search(cube_location)
		if cube_file eq '' then begin
			print, '% SL_IO_CORRECT:*************************ERROR*************************'
			print, '% SL_IO_CORRECT:**************CUBE PROJECT FILE NOT FOUND**************'
			PRINT, '% SL_IO_CORRECT:********PLEASE CHECK YOUR FILEPATH AND TRY AGAIN*******'
			PRINT, '% SL_IO_CORRECT:***********************QUITTING************************'
			return		
		endif
		; Load the targetted cube.
		cube = cubeproj_load(cube_location)
	endelse
	
	; Make certain this is a SL cube
	cube->getproperty, module=temp_order
	if temp_order eq 'SL' then print, '% SL_IO_CORRECT: Short-low cube loaded.' else begin
		print, '% SL_IO_CORRECT: *************************ERROR*************************'
		print, '% SL_IO_CORRECT: **************CUBE PROJECT NOT IDENTIFIED**************'
		PRINT, '% SL_IO_CORRECT: ******************AS A SHORT-LOW CUBE******************'
		PRINT, '% SL_IO_CORRECT: ***********************QUITTING************************'
		return	
	endelse
	; Retrieve the BCDs!
	cube -> RestoreData, /ALL
	bcds = cube -> BCD(/ALL, uncertainty = unc)

	; Determine the number of records we will be processing.
	array_dimensions = size(bcds)
	num_recs = array_dimensions[3]
	
	; Get the background too.
	cube->GetProperty, background=background
	; We only care about background subtracted signals, so do that!
	if not keyword_set(background) then begin
		print, '% SL_IO_CORRECT: *************************WARNING*************************'
		print, '% SL_IO_CORRECT: *********NO BACKGROUND RECORDS HAVE BEEN SET*************'
		print, '% SL_IO_CORRECT: *******THIS ROUTINE WAS INTENDED FOR BACKGROUND**********'
		print, '% SL_IO_CORRECT: ********************SUBTRACTED DATA**********************'
		print, '% SL_IO_CORRECT: ******************USE AT YOUR OWN RISK*******************'
		print, '% SL_IO_CORRECT: *************************WARNING*************************'	
	endif else begin
		temp_size = size(background)
		background = rebin(background, temp_size[1], temp_size[2],num_recs)
		bcds = bcds-background
	endelse
	; We now wish to create a 3 dimensional array containing the inter-order
	; light between the sl1 and sl2/3 arrays. To do so, we'll be using the
	; flatap chmask to give us an idea of what pixels on the bcd we are
	; interested in.

	; Unfortunately, it has a little extra 'inter-order-light' sections that 
	; we don't actually want to use. So, we're going to trim that off. 
	dir=file_dirname((routine_info('sl_io_correct',/SOURCE)).PATH)
	cmask_file = file_search(dir+'/b0_flatfield_cmask.fits')
	if cmask_file eq '' then begin
		print, '% SL_IO_CORRECT: *************************ERROR*************************'
		print, '% SL_IO_CORRECT: **********b0_flatfield_cmask.fits NOT FOUND************'
		PRINT, '% SL_IO_CORRECT: ***********MAKE CERTAIN IT IS LOCATED AT **************'
		PRINT, '% SL_IO_CORRECT: *********THE SAME LOCATION AS SL_IO_CORRECT.PRO********'
		PRINT, '% SL_IO_CORRECT: ***********************QUITTING************************'
		return
	endif else if not keyword_set(quiet) then $
           print, '% SL_IO_CORRECT: Reading '+cmask_file
	io_temp = readfits(cmask_file)
	io_temp[45:127,*] = 256
	io_temp[0:33,*] = 256
	io_temp[33:37,127] = 256

	; Within the cmask, everything having a value of 128 represents inter-order
	; light that was not used at all in processing the on-order light. Thus, it
	; is exactly what we want to keep and analyze. 
	; For the ease of future programming, we will set everything we want to use
	; equal to 1, and everything else set to NaN. This will allow us to simply
	; multiply this 'template' to our BCDs and only have the data we want 
	; remaining.

	wh = where(io_temp eq 128, complement = wt)
	io_temp = findgen(128,128)
	io_temp[wh] = 1
	io_temp[wt] = !VALUES.F_NAN

	; Applying this template.
	io_temp = rebin(io_temp, 128,128,num_recs)
	bcds = bcds*io_temp



	; So, the bcds variable is a 128x128xnumber_of_records in its host cube array
	; with values equal to NAN everywhere that is not pertaining to the 
	; inter-order light found between the SL1 and SL2+3 arrays

	; It is our objective to choose a rolling selection of pixels both 
	; in the row-space (2nd axis) and BCD space (3rd axis) and perform an 
	; intellegent sigma-trimmed averaging of each sub-box of pixels in order
	; to generate a row by bcd image of the inter-order light for a given 
	; datacube

	; Some data cubes may have records that are disabled for use as dedicated
	; backgrounds, or have multiple AORs. Due to the nature of our IO correction,
	; we wish to respect AOR boundaries when averaging over multiple BCDs, thus, 
	; we need to know where they begin and end in our BCD array. 

	; The following is some CUBISM mojo designed to determine at what BCD index
	; does the given cube switch from one AOR to the other. This will be used
	; in the subsequent processing loop.

	; The author of this program is rather proud of this litte bit of code.
	; Feel free to bask in its glory. 
	cube -> GetProperty, DR=drs, /POINTER
	aorids=(*drs).AORKEY
  	uniqids=aorids[uniq(aorids,sort(aorids))] ; Ok, so that part was in Cubism
	aorbreaks = replicate(0, n_elements(uniqids))
	for i = 0, n_elements(aorbreaks)-1 do aorbreaks[i] = max(where((*drs).AORKEY eq uniqids[i]))
	aorbreaks = aorbreaks[sort(aorbreaks)]
	isdisabled = where((*drs).disabled eq 1)
	
	; Well bask'd? Moving on. 


	; We now get into actually processing the for loop which will select a 
	; sub-cube of pixels using a boxcar average (described in more detail below).
	; The results of which will be stuffed into the following 128xnum_recs array
	; for future processing.
	io_intermediate = replicate(0.0, 128,num_recs)
	if not keyword_set(bcd_average_range) then bcd_average_range = 5
	if not keyword_set(row_average_range) then row_average_range = 5

	; There is a possibility that the user entered a BCD or ROW average range
	; that is greater than the length of the BCD, or the number of available
	; AORs. Due to the boxcar averaging we're doing, this could cause some
	; inproper results. Thus, if the user enters a value too large, it needs
	; to be set to the maximum width of the BCD/aor and print a warning.

	; If this code seems messy, I blame users that try to do silly things. 

	; So, we not only need to determine what is the minimum number of records in
	; all of our AORs, but we need to account for the fact that some of the aors
	; may be disabled for being a dedicated background... which typically means
	; they would have a rather small number of aors within them. 
	temp_aors = aorbreaks[where(aorbreaks ne aorbreaks[min(where(aorbreaks ge min(isdisabled)))])]
	if n_elements(temp_aors) eq 0 then begin
		print, '% SL_IO_CORRECT: *************************ERROR*************************'
		print, '% SL_IO_CORRECT: *********ALL AORS PRESENT ARE SET TO DISABLED**********'
		PRINT, '% SL_IO_CORRECT: ***********NO CORRECTION WILL BE PERFORMED*************'
		PRINT, '% SL_IO_CORRECT: ***********************QUITTING************************'
		return
	endif else begin
		if n_elements(temp_aors) eq 1 then temp = [temp_aors[0]+1] else begin
			temp = replicate(0, n_elements(temp_aors)-1)
			for i=0, n_elements(temp_aors)-2 do temp[i]=temp_aors[i+1]-temp_aors[i]
			temp = [temp_aors[0]+1,temp]
		endelse 
	endelse

	if bcd_average_range ge min(temp) then begin
		print, '% SL_IO_CORRECT: *************************WARNING*************************'
		print, '% SL_IO_CORRECT: *************BCD_AVERAGE_RANGE GREATER*******************'
		print, '% SL_IO_CORRECT: *********THAN THE NUMBER OF RECORDS IN AN AOR************'
		print, '% SL_IO_CORRECT: ************SETTING BCD_AVERAGE_RANGE EQUAL**************'
		print, '% SL_IO_CORRECT: **********TO THE SIZE OF THE SMALLEST AOR: '+strtrim(min(temp),2)+'***********'
		print, '% SL_IO_CORRECT: *************************WARNING*************************'	
		bcd_average_range = min(temp)	
	endif

	if bcd_average_range eq 0 then begin
		print, '% SL_IO_CORRECT: *************************WARNING*************************'
		print, '% SL_IO_CORRECT: ***********BCD_AVERAGE_RANGE MINIMUM IS ONE**************'
		print, '% SL_IO_CORRECT: *********THAN THE NUMBER OF RECORDS IN AN AOR************'
		print, '% SL_IO_CORRECT: *******NOW SETTING BCD_AVERAGE_RANGE EQUAL TO ONE********'
		print, '% SL_IO_CORRECT: *************************WARNING*************************'	
		bcd_average_range = 1	
	endif

	if row_average_range lt 1 then begin
		print, '% SL_IO_CORRECT: *************************WARNING*************************'
		print, '% SL_IO_CORRECT: ***********ROW_AVERAGE_RANGE MINIMUM IS ONE**************'
		print, '% SL_IO_CORRECT: *******NOW SETTING ROW_AVERAGE_RANGE EQUAL TO ONE********'
		print, '% SL_IO_CORRECT: *************************WARNING*************************'	
		row_average_range = 1	
	endif

	if row_average_range gt 128 then begin
		print, '% SL_IO_CORRECT: *************************WARNING*************************'
		print, '% SL_IO_CORRECT: ***********ROW_AVERAGE_RANGE MAXIMUM IS 128**************'
		print, '% SL_IO_CORRECT: *******NOW SETTING ROW_AVERAGE_RANGE EQUAL TO 128********'
		print, '% SL_IO_CORRECT: *************************WARNING*************************'	
		row_average_range = 128	
	endif

	for i = 0, num_recs-1 do begin
		; If the record is disabled, we do not want to deal with it.
		if where(isdisabled eq i) eq -1 then begin
			; First we generate the range over which we will be averaging the BCDs.
			; To do that, we need to establish where the AOR boundaries are in 
			; relation to our current position in the bcds cube
			if i eq 0 then begin 
					current_break = aorbreaks[0]
					last_break = 0
			endif else begin
					t = where(i le aorbreaks)
					current_break = aorbreaks[min(t)]
					if min(t) eq 0 then last_break = -1 else last_break = aorbreaks[min(t)-1]
			endelse

			; After that bit of fun, we now know where the current AOR began 
			; via last_break and where it ends via current_break. This is important
			; as we now go into selecting the BCDs which will contribute to our
			; sub-cube of pixels. The following code is a little complicated as 
			; we have elected to perform an expanding/contracting boxcar average
			; in BCD space, meaning that for a BCD average length of 5, we'd see
			; something like this (assuming last_break eq 270 and c_break eq 278):
			; 271: [271, 272, 273]
			; 272: [271, 272, 273, 274]
			; 273: [271, 272, 273, 274, 275]
			; 274:      [272, 273, 274, 275, 276]
			; 275:			  [273, 274, 275, 276, 277]
			; 276:					 [274, 275, 276, 277, 278] 
			; 277:							[275, 276, 277, 278]
			; 278:								  [276, 277, 278]
			; 279:													  [279, 280, 281]  etc etc
				
			if keyword_set(TRIM_EDGES) then begin 
				if i eq 0 then z = indgen(bcd_average_range/2+bcd_average_range mod 2)	else $
				if i le last_break + bcd_average_range/2 then begin
					; If we are near the beginning of the aor, our boxcar will be small
					z = indgen(bcd_average_range/2+bcd_average_range mod 2+(i-last_break-1)) + last_break + 1
				endif else if i le current_break - bcd_average_range/2 - 1 then begin
					; We are somewhere in the middle of the aor, our boxcar is fullsize
					z = indgen(bcd_average_range)+i-(bcd_average_range/2)
				endif else begin
					; We are approaching the end of the aor, our boxcar must shrink
					z = indgen(bcd_average_range/2+(current_break-i)+(bcd_average_range mod 2)) + i-bcd_average_range/2+(1-bcd_average_range mod 2)
				endelse
			endif else begin
				if i eq 0 then z = indgen(bcd_average_range/2+bcd_average_range mod 2)	else $
				if i le last_break + bcd_average_range/2 then begin
					; If we are near the beginning of the aor, hold at the starting position
					z = indgen(bcd_average_range)+last_break+1
				endif else if i le current_break - bcd_average_range/2 - 1 then begin
					; We are somewhere in the middle of the aor, our boxcar is fullsize
					z = indgen(bcd_average_range)+i-(bcd_average_range/2)
				endif else begin
					; We are approaching the end of the aor, begin finite boxcar!
					z = indgen(bcd_average_range)+current_break - bcd_average_range + 1
				endelse

			endelse

			; Next, we generate the range over which we will be averaging the rows.
			; This is similarly done to the aor boxcar, except we only have 2 
			; breakpoints, 0 and 128

			for j=0, 128-1 do begin
				; Determine the range of rows to average over
				if keyword_set(TRIM_EDGES) then begin
					if j le row_average_range/2 then begin
						y = indgen(row_average_range-(row_average_range/2-j))
					endif else if j lt (128 - row_average_range/2 -1) then begin
						y = lindgen(row_average_range)+j-(row_average_range/2)
					endif else begin
						y = indgen(row_average_range/2+(128-j-1)+(row_average_range mod 2)) + j-row_average_range/2+(1-row_average_range mod 2)
					endelse
				endif else begin
					if j le row_average_range/2 then begin
						y = indgen(row_average_range)
					endif else if j lt (128 - row_average_range/2 -1) then begin
						y = indgen(row_average_range)+j-(row_average_range/2)
					endif else begin
						y = indgen(row_average_range) + 128-row_average_range
					endelse
				endelse
				; We are now capable of selecting our box of pixels that we are
				; interested in binning over. 
				; Also, noting that the row dimension is the vertical axis on the 
				; BCD image, this is quite possibly one of the few times where the 
				; IDL convention of collumn,row actually makes sense. 
				pixel_cube = bcds[*,y,z]

				; Note that there are lots of NaN datapoints that we do not care
				; about, so we'll get rid of them with finite.
				wh = where(finite(pixel_cube))
				if min(wh) ne -1 then 	pixel_cube = pixel_cube[wh] else $
					pixel_cube = [0]

				; We now have a 1-D array of pixel values that we believe represent
				; position [j,i] in our io_intermediate array ([row,bcd] btw)

				; In the interests of avoiding points radically outside the mean, 
				; we will perform an iterative 1-sigma based trim. In order to avoid
				; throwing away too many of our pixels, we will simply shoot for 
				; cutting off no more than 80% of n_elements(pixel_cube) for 
				; simplicity. Note that we should have approximately 125ish pixels
				; to play with if the defaults were chosen. (5 rows, 5 bcds, 
				; 5ish collumns)
				; The variable p is present to make certain that this does not end
				; up being an infinite loop while giving it plenty of time to 
				; process through the sub-cube. This section of code is most likely
				; the most computationally expensive portion of the entire routine.

				; Catch added that if the sigma-trim does not trim off any pixels, 
				; don't do another iteration.

				pix_start = n_elements(pixel_cube)
				p = 10
				while (p gt 0 and n_elements(pixel_cube) gt 0.8*pix_start) do begin
					pix_mean = mean(pixel_cube)
					pix_sigma = sigma(pixel_cube)	
					wh = where( (pixel_cube ge (pix_mean - pix_sigma)) and (pixel_cube le (pix_mean + pix_sigma)), complement=lo)
					if n_elements(lo) eq 0 then break					
					pixel_cube = pixel_cube[wh]
					p -= 1
				endwhile	

				; The pixel_cube should now be trimmed, we may now store its mean
				; in our array.
				io_intermediate[j,i] = mean(pixel_cube)
						
				; This row for this BCD has been completed, process the next row
			endfor
			; This BCD has been completed, process the next BCD.
			if not keyword_set(quiet) then print, '% SL_IO_CORRECT: '+strtrim(i+1,2)+' of '+strtrim(num_recs,2)+' records processed.'
		endif
	endfor
			if not keyword_set(quiet) and n_elements(isDisabled) ne 0 then $
				print, '% SL_IO_CORRECT: '+strtrim(n_elements(isDisabled),2)+' disabled records skipped.'


	; At this point, we now have a 128*num_recs image containing a sigma-trimmed
	; row/bcd averaged value of the flux present in each row of each bcd of 
	; the given datacube. 

	; If you wish to visualize the results/effect of certain choices of 
	; bcd_average_range and row_average_range, this quick and dirty plotting 
	; routine may be useful to you. At least, it was to me.
	if keyword_set(visualize) then begin

		total_mean = findgen(num_recs-n_elements(isdisabled))
		for i=0, num_recs-n_elements(isdisabled)-1 do total_mean[i] = mean(io_intermediate[*,i])
		upper_mean = findgen(num_recs-n_elements(isdisabled))
		for j=0,num_recs-n_elements(isdisabled)-1 do upper_mean[j] = mean(io_intermediate[84:123,j])
		lower_mean = findgen(num_recs-n_elements(isdisabled))
		for h=0,num_recs-n_elements(isdisabled)-1 do lower_mean[h] = mean(io_intermediate[2:43,h])

		window_xstart = 0.1
		window_xend = 0.98
		window_ystart = 0.1
		window_yend = 0.93
		charsize= 1.5

		cgplot, [0,0],[0,0],/nodata, xrange=[0,num_recs], /xstyle, $
				yrange=[2*min([total_mean,upper_mean,lower_mean]), $
							2*max([total_mean,upper_mean,lower_mean])], /ystyle, $
				title = 'Inter-order light bcd range: '+strtrim(bcd_average_range,2)+ $
							' row range: '+strtrim(row_average_range,2), $
				xtitle = 'Dummy axis for BCD index', $
				ytitle = 'Average signal [e-/sec]', $
				charsize = charsize, font=-1, $
				position=[window_xstart,window_ystart,window_xend,window_yend], /normal


	

		a = mean(total_mean)
		cgplot, findgen(num_recs-n_elements(isdisabled)-1), total_mean, color='Purple', /overplot
		cgplot, [0,num_recs-n_elements(isdisabled)], [a,a+0.0000001], color='Purple', /overplot



		b = mean(upper_mean)
		cgplot, findgen(num_recs-n_elements(isdisabled)), upper_mean, color='Turquoise', /overplot
		cgplot, [0,num_recs-n_elements(isdisabled)], [b,b+0.0000001], color='Turquoise', /overplot



		c = mean(lower_mean)
		cgplot, findgen(num_recs-n_elements(isdisabled)), lower_mean, color='Red', /overplot
		cgplot, [0,num_recs-n_elements(isdisabled)], [c,c+0.0000001], color='tomato', /overplot


		for i=0, n_elements(aorbreaks)-1 do begin	
			cgplot, [aorbreaks[i],aorbreaks[i]+.0000001],[-9000000,9000000], color='green', /overplot
		endfor

		to_be_tvd = rotate(io_intermediate, 3)
		to_be_tvd = reverse(to_be_tvd,2)
		tvd_scl_min = min(to_be_tvd[where(to_be_tvd ge mean(to_be_tvd)-2*sigma(to_be_tvd))])
		tvd_scl_max = max(to_be_tvd[where(to_be_tvd le mean(to_be_tvd)+2*sigma(to_be_tvd))])
		tvd_size = size(to_be_tvd)

		tvd_xstart = window_xstart
		tvd_xend = window_xend
		tvd_ystart = 0.8
		tvd_yend = tvd_ystart+tvd_size[2]/1000.0
		tvd_red_max = 48.0*(tvd_yend-tvd_ystart)/(128.0)+tvd_ystart
		tvd_blue_min = 84.0*(tvd_yend-tvd_ystart)/(128.0)+tvd_ystart
		x_v1 = (tvd_xend-tvd_xstart)/tvd_size(1)
		x_v2 = tvd_xstart
		cgimage, to_be_tvd, position=[tvd_xstart, tvd_ystart, tvd_xend, tvd_yend], /scale, minvalue=tvd_scl_min, maxvalue=tvd_scl_max, /noerase
		cgcolorbar, position=[(tvd_xend+tvd_xstart)/2+0.1, tvd_ystart-0.05, $
			(tvd_xend+tvd_xstart)/2+0.4, tvd_ystart-0.01], charsize=charsize-.5, $
			minrange=tvd_scl_min, maxrange=tvd_scl_max, format='(F10.2)', font=-1

		cgcolorfill, [tvd_xstart-0.01,tvd_xstart-0.01,tvd_xstart-0.005,tvd_xstart-0.005],$
			[tvd_ystart,tvd_yend,tvd_yend,tvd_ystart], color='Purple', /normal
		cgcolorfill, [tvd_xstart-0.005,tvd_xstart-0.005,tvd_xstart,tvd_xstart],$
			[tvd_blue_min,tvd_yend,tvd_yend,tvd_blue_min],	color='Turquoise', /normal
		cgcolorfill, [tvd_xstart-0.005,tvd_xstart-0.005,tvd_xstart,tvd_xstart],$
			[tvd_ystart,tvd_red_max,tvd_red_max,tvd_ystart], color='red', /normal
		for i=0, n_elements(aorbreaks)-1 do begin	
			cgcolorfill, [aorbreaks[i]*x_v1+x_v2-.001,aorbreaks[i]*x_v1+x_v2-.001,$
			 aorbreaks[i]*x_v1+x_v2+.002, aorbreaks[i]*x_v1+x_v2+.002], $
			[tvd_ystart, tvd_ystart+tvd_size[2]/1000.0, $
			tvd_ystart+tvd_size[2]/1000.0, tvd_ystart], $
			color='Forest Green', /normal
		endfor
		cgText, /normal, tvd_xstart-0.04, tvd_ystart+0.01, orientation=90, 'Row ->', ALIGNMENT=0, charsize=charsize, font=-1
		cgText, /normal, tvd_xstart+(tvd_xend-tvd_xstart)/2, tvd_ystart-0.03, 'BCD ->', ALIGNMENT=0.5, charsize=charsize, font=-1

	endif else begin

	
		; Now we need to apply the flat-field to our data.
		; This will take our 128x820 image and recreate a 128x128x820 cube that 
		; should be subtracted off of the main data-cube to perform the IO 
		; correction
	
		flatfield_file = file_search(dir+'/b0_flatfield.fits')
		if cmask_file eq '' then begin
			print, '% SL_IO_CORRECT: *************************ERROR*************************'
			print, '% SL_IO_CORRECT: *************b0_flatfield.fits NOT FOUND***************'
			PRINT, '% SL_IO_CORRECT: ************MAKE CERTAIN IT IS LOCATED AT *************'
			PRINT, '% SL_IO_CORRECT: *********THE SAME LOCATION AS SL_IO_CORRECT.PRO********'
			PRINT, '% SL_IO_CORRECT: ***********************QUITTING************************'
			return
		endif else if not keyword_set(quiet) then $
											print, '% SL_IO_CORRECT: Reading '+flatfield_file
		flat_field = readfits(flatfield_file)
		; The flat_field datafile has two layers. (At least the one we had did.)
		; We only care about the bottom one.
		flat_field = flat_field[*,*,0]
		; The flat_field is a 128x128 array. For each point in that array which is 
		; finite, we need to divide that value into the io row-average in 
		; io_intermediate.

		io_final = replicate(0.0, 128, 128, num_recs)
		for k = 0, num_recs - 1 do begin
			for j = 0, 128-1 do begin
				for i=0, 128-1 do begin
					io_final[i,j,k] = io_intermediate[j,k]/flat_field[i,j]
				endfor
			endfor
		endfor

		; It occurs to me that idl interprets X - NaN as NaN. Since all of our 
		; off-IO values are NaN, this will present a problem if we attempt to 
		; just neatly subtract off our correction from the cube. 
		; So, now that we're done using finite to locate the changes we wish to 
		; make, we'll go ahead and convert all of the NaN into 0.0
		temp = where(finite(io_final), complement = wh)
		io_final[wh] = 0.0

		; Since DR seems to be a pointer to an array of pointers, we don't seem 
		; to be able to simply do (*drs).bcd - io_final, despite io_final being
		; a nice and pretty 128x128x820 array... So, we for loop.
		for i=0, num_recs-1 do begin
			*(*drs)[i].bcd = *(*drs)[i].bcd - io_final[*,*,i]
		endfor


		; Now for a little more cubism mojo to actually apply our changes
		cube -> SetDirty 
		cube -> BuildCube
		project_name = cube->ProjectName()
		cube->SetProjectName, project_name + ' (IO_COR)'
		if keyword_set(save_location) then begin 
			cube -> SaveCube, save_location
			obj_destroy, cube
			print, '% SL_IO_CORRECT: File saved to :' + save_location
		endif
	endelse


	; This was a triumph!
end
