To build the OpenBT package:
----------------------------

1. ./configure --with-mpi

Note: you can disable extensive output from OpenBT's MCMC by adding --with-silent to the configure command:

./configure --with-mpi --with-silent

2. make

Note: this will build the three main programs: openbtcli, openbtpred and openbtvartivity.  These programs are manipulated in R via the functions
availabile in openbt.R.  

3. sudo make install

This will copy the executable programs (openbtcli, openbtpred, openbtvartivity) to a system-wide location (e.g. such as /usr/local/bin).  

4. To use the package, ensure the openbt.R wrapper script is available from your working directory.  Start R and source openbt.R to load the R wrapper functions for using the OpenBT package.  There are some simple examples in example.R on using the package.

Note: On first sourcing openbt.R, some support packages may be automatically installed.  This requires you to have a working internet connection.

Note: This package currently requires a working MPI library to be installed in
order to build.  The package was developed using OpenMPI.  A version using
OpenMP will be forthcoming.

Note: Saved fitted models make use of compression to save disk space.  Since they are compressed, they are not in human-readable format.  A file extension of .obt is automaticalled added to saved models.  You can load .obt files at a later time, for instance, to make use of the fitted posterior.



To Remove the OpenBT package:
-----------------------------

1. sudo make uninstall

This will remove all program files and libraries from the system-wide directories.



Troubleshooting
---------------

1. [Linux/Unix] If you receive a library path error such as ``openbtcli: error while loading shared libraries: libpsbrt.so.0: cannot open shared object file: No such file or directory'' when running the program from a system-wide install, you may need to update the system library cache by running sudo ldconfig.



Contact
-------

Please send feedback to mpratola@gmail.com.

