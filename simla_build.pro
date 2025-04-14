;; SIMLA_BUILD (2022, JDS)
;;
;;  Build a single cube in one MODULE (integer) and (optionally, for
;;  low-res) ORDER, from a list of files FILES.  Use a
;;  BACKGROUND_FRAME (array) or BACKGROUND_IDS (list of bcdid's, among
;;  FILES).  Load a list of bad pixels to ignore (BP), globally,
;;  and/or for individual records (a HASH with key RECID, or 'global',
;;  elements a list of (1D) pixel values)

pro simla_build, files, module, outfile, ORDER=ord, BACKGROUND_FRAME = bgf, $
                 BACKGROUND_UNC = bgu, BADPIX=bp, AUTO_BADPIX = abp
  
  if n_elements(bp) eq 0 then bp = []
  
  modname=irs_module(module,/TO_NAME)
  name = file_basename(outfile)
  cube=obj_new('CubeProj',name)
  print, 'building ',name
  ;; Set the build order and other properties
  cube->SetProperty,FEEDBACK=0,ORDER=ord, $
                             /RECONSTRUCTED_POSITIONS,$
                             /SAVE_ACCOUNTS,SAVE_DATA=0,/FLUXCON,/SLCF, $
                             /WAVECUT,/LOAD_UNCERTAINTY,/USE_UNCERTAINTY, $
                             /USE_BACKGROUND,/PIXEL_OMEGA,/REPORT_QUIET
  
  if keyword_set(new_calib) then begin 
     print,' -> Force loading latest calset'
     cube->LoadCalib,/FORCE,_EXTRA=e ; reload new calibration
     cube->GetProperty,CAL_FILE=cal_file
     print,' -> Loaded: ',file_basename(cal_file)
  endif 

  print," -> Loading ",strtrim(n_elements(files),2)," files."   
  cube->AddData, files

  ;; Background Frame
  if n_elements(bgf) gt 0 then $
     cube->SetProperty, BACKGROUND_FRAME = bgf
  if n_elements(bgu) gt 0 then $
     cube->SetProperty, BACKGROUND_UNC = bgu
  
  foreach list, bp, id do begin 
     if id eq 'global' then begin
        cube->ToggleBadPixel, list, /SET
     endif else begin
        cube->ToggleBadPixel, bp, RECORD_SET=id, /SET
     endelse 
  endforeach 
  
  ;; Set up Wavsamp trimming (shift SL1 2% right, March, 2007)
  if modname eq 'SL' then begin 
     cube->SetProperty, APERTURE=ord eq 1?irs_aperture(.05,.96): $
                                 irs_aperture(.03,.94)
  endif else if modname eq 'LL' then begin 
     cube->SetProperty, APERTURE=irs_aperture(.03,.96)
  endif 
  
  ;; Build the cube
  print,FORMAT='(%" -> Building %s Cube...",$)', name
  t0=systime(1)
  cube->BuildCube
  print,FORMAT='(%" done in %5.2f min")',(systime(1)-t0)/60.
  
  ;; Auto-badpix 
  if keyword_set(abp) then begin 
     if modname eq 'SL' || modname eq 'LL' then begin 
        sigma_thresh=4.
        minfrac=0.4
     endif else begin ;; High-res 
        sigma_thresh=5.
        minfrac=0.5
     endelse 
     print, $
        string(FORMAT='(A,F0.2,"/",F0.2)', $
               ' -> Generating Global Auto-bad-pixels with params: ', $
               sigma_thresh,minfrac)
     
     cube->AutoBadPixels, MAXVAR=sigma_thresh, MINFRAC=minfrac,$
                                 /WITH_BACKGROUND, USE_UNC=0
     ;; Re-build the cube
     print,FORMAT='(%" -> Re-Building %s Cube...",$)', name
     t0=systime(1)
     cube->BuildCube
     print,FORMAT='(%" done in %5.2f min")',(systime(1)-t0)/60.
     cube->GetProperty,GLOBAL_BAD_PIXEL_LIST=bpl
     print,FORMAT='(%" -> Found %d Global Auto-Bad-Pixels")', $
           n_elements(bpl)
           
  endif   
  
  outfile = REPSTR(outfile, "_unc", "")
  cube->SaveCube, outfile
  cubename = REPSTR(outfile, "fits", "cpj")
  cube->Save, cubename
  
  return 
end
