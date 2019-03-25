(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator"))
(in-package :cl-py-generator)

;(setf *features* (union *features* '(:plot :pd :highres)))
;(setf *features* (set-difference *features* '(:highres)))

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
		      ;(pd pandas)
		      ;(xr xarray)
		      pathlib
		      time
		      
		      ))

	    (def current_milli_time ()
	      (return (int (round (* 1000 (time.time))))))
	    
	    	    
	    (setf args (docopt.docopt __doc__ :version (string "0.0.1")))
	    (if (aref args (string "--verbose"))
		(print args)))))
    (write-source *source* code)))
