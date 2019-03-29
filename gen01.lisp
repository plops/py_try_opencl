(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)

;(setf *features* (union *features* '(:plot :pd :highres)))
;(setf *features* (set-difference *features* '(:highres)))

;https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480

(defun timer (name body)
  (let ((start (format nil "time_before_~a" name))
	(end (format nil "time_after_~a" name)))
   `(do0
     (setf ,start (current_milli_time))
     ,body
     (setf ,end (current_milli_time))
     (print (dot (string ,(format nil "~a time: {}ms" name))
		 (format (- ,end ,start)))))))

(progn
  (defparameter *path* "/home/martin/stage/py_try_opencl/")
  (defparameter *code-file* "run_01_ocl")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))

  (let* (
	 (code
	  `(do0
	    "#!/usr/bin/env python3" ;; pipenv run python
	    

	    (string3 ,(format nil "opencl test code.
Usage:
  ~a [-vh]

Options:
  -h --help               Show this screen
  -v --verbose            Print debugging output
"
			      *code-file*))
	    
	    "# martin kielhorn 2019-03-26"
	    ;"from __future__ import print_function"
	    #+plot
	    (do0
	     (imports (matplotlib))
	     (imports ((plt matplotlib.pyplot)))
	     (plt.ion)
	     (setf font (dict ((string size) (string 5))))
	     (matplotlib.rc (string "font") **font)
	     ;(imports ((xrp xarray.plot)))
	     )
	    (imports (os
		      sys
		      docopt
		      traceback
		      (np numpy)
		      (cl pyopencl)
		      ;(pd pandas)
		      ;(xr xarray)
		      pathlib
		      time
		      
		      ))

	    
	    
	    (def current_milli_time ()
	      (return (int (round (* 1000 (time.time))))))
	    (do0
	     (class bcolors ()
		    (setf OKGREEN (string "\\033[92m")
			  WARNING (string "\\033[93m")
			  FAIL (string "\\033[91m")
			  ENDC (string "\\033[0m")))
	     "global g_last_timestamp"
	     (setf g_last_timestamp (current_milli_time))
	     (def milli_since_last ()
	       "global g_last_timestamp"
	       (setf current_time (current_milli_time)
		     res (- current_time g_last_timestamp)
		     g_last_timestamp current_time)
	       (return res))
	     (def fail (msg)
			  (print (+ bcolors.FAIL
				    (dot (string "{:8d} FAIL ")
					 (format (milli_since_last)))
				    msg
				    bcolors.ENDC))
			  (sys.stdout.flush))
	     (def plog (msg)
	       (print (+ bcolors.OKGREEN
			 (dot (string "{:8d} LOG ")
			      (format (milli_since_last)))
			 msg
			 bcolors.ENDC))
	       (sys.stdout.flush)))	    
	    (setf args (docopt.docopt __doc__ :version (string "0.0.1")))
	    (if (aref args (string "--verbose"))
		(print args))

	    (setf platforms (cl.get_platforms)
		  platform (aref platforms 0)
		  devices (platform.get_devices cl.device_type.GPU)
		  device (aref devices 0)
		  context (cl.Context (list device)))
	    (plog (dot (string "using first device of {}")
		       (format devices)))
	    (setf queue (cl.CommandQueue context device))
	    (do0
	     (plog (string "create refractive index distribution"))
	     (setf img_in_y 128
		   img_in_z 34
		   )
	     (setf img_in (np.full (tuple img_in_z img_in_y)  1.5s0 :dtype np.float32)))
	    (try
	     (do0
	      (plog (string "instantiate in and output arrays on the gpu"))
	      (setf gpu_shape img_in.shape
		    img_in_gpu (cl.Image context  cl.mem_flags.READ_ONLY (cl.ImageFormat cl.channel_order.R cl.channel_type.FLOAT) :shape gpu_shape)
		    img_out_gpu (cl.Image context cl.mem_flags.WRITE_ONLY (cl.ImageFormat cl.channel_order.R cl.channel_type.FLOAT) :shape gpu_shape)
		    ))
	     ("Exception as e"
	      (do0
	       (do0
		(setf (ntuple type_ value_ tb_ ) (sys.exc_info)
		      lineno (dot tb_ tb_lineno))
		#+nil
		(fail (dot (string "Error in line {} of {} {}: '{}' prop='{}' value={}")
			   (format lineno
				   (dot (type e)
					__name__)
				   (dot (string ".")
					(join (list self.__module__
						    self.__class__.__name__)))
				   e
				   prop
				   value)))
		(for (e (traceback.format_tb tb_))
		     (fail e)))
	       
	       
	       (setf fmts (cl.get_supported_image_formats context
							  cl.mem_flags.READ_ONLY
							  cl.mem_object_type.IMAGE2D))
	       (plog (dot (string "supported READ_ONLY IMAGE2D formats: {}.")
			  (format fmts))))))
	    )))
    (write-source *source* code)))
