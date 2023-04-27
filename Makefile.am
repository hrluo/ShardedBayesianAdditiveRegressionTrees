# automake(Makefile.am) -> Makefile.in
# autoconf(Makefile.in, configure.ac) -> Makefile

ACLOCAL_AMFLAGS = -I m4

AUTOMAKE_OPTIONS = foreign subdir-objects no-installinfo no-installman dist-zip

# main package installed binaries
bin_PROGRAMS = openbtcli openbtpred openbtvartivity openbtsobol openbtmopareto openbtinfl openbtcalibrate openbtemulate


# main package programs
openbtcli_SOURCES = cli.cpp openbt.R example.R
openbtcli_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

openbtpred_SOURCES = pred.cpp
openbtpred_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

openbtvartivity_SOURCES = vartivity.cpp
openbtvartivity_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

openbtsobol_SOURCES = sobol.cpp
openbtsobol_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

openbtmopareto_SOURCES = mopareto.cpp
openbtmopareto_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

openbtinfl_SOURCES = influence.cpp
openbtinfl_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

openbtcalibrate_SOURCES = calibrate.cpp
openbtcalibrate_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

openbtemulate_SOURCES = emulate.cpp
openbtemulate_LDADD = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la

# test programs -- these are only built if 'make check' is run.
check_PROGRAMS = testcrn testtree testbrt testlebrt testmbrt testsbrt testambrt testpsbrt testsinglebinomial testsinglepoisson
testcrn_SOURCES = test_crn.cpp
testcrn_LDADD = libcrn.la
testtree_SOURCES = test_tree.cpp
testtree_LDADD = libcrn.la libtree.la libbrt.la
testbrt_SOURCES = test_brt.cpp
testbrt_LDADD = libcrn.la libtree.la libbrt.la
testmbrt_SOURCES = test_mbrt.cpp
testmbrt_LDADD = libcrn.la libtree.la libbrt.la libmbrt.la
testsbrt_SOURCES = test_sbrt.cpp
testsbrt_LDADD = libcrn.la libtree.la libbrt.la libmbrt.la libsbrt.la
testambrt_SOURCES = test_ambrt.cpp
testambrt_LDADD = libcrn.la libtree.la libbrt.la libmbrt.la libsbrt.la libambrt.la
testpsbrt_SOURCES = test_psbrt.cpp
testpsbrt_LDADD = libcrn.la libtree.la libbrt.la libmbrt.la libsbrt.la libambrt.la libpsbrt.la
testsinglebinomial_SOURCES = test_singlebinomial.cpp
testsinglebinomial_LDADD = libcrn.la libtree.la libbrt.la libsinglebinomial.la
testsinglepoisson_SOURCES = test_singlepoisson.cpp
testsinglepoisson_LDADD = libcrn.la libtree.la libbrt.la libsinglepoisson.la
testlebrt_SOURCES = test_lebrt.cpp
testlebrt_LDADD = libcrn.la libtree.la libbrt.la liblebrt.la


# libraries
lib_LTLIBRARIES = libsinglebinomial.la libsinglepoisson.la liblebrt.la libpsbrt.la libambrt.la libsbrt.la libmbrt.la libbrt.la libtree.la libcrn.la
#libHelloWorld_la_LDFLAGS = -version-info 0:0:0
libcrn_la_SOURCES = crn.cpp crn.h rn.h tnorm.cpp tnorm.h
libtree_la_SOURCES = treefuns.cpp treefuns.h tree.cpp tree.h
libbrt_la_SOURCES = brt.cpp brt.h brtmoves.cpp brtfuns.cpp brtfuns.h dinfo.h
libmbrt_la_SOURCES = mbrt.cpp mbrt.h
libsbrt_la_SOURCES = sbrt.cpp sbrt.h
libambrt_la_SOURCES = ambrt.cpp ambrt.h
libpsbrt_la_SOURCES = psbrt.cpp psbrt.h
libsinglepoisson_la_SOURCES = singlepoisson.cpp singlepoisson.h
libsinglebinomial_la_SOURCES = singlebinomial.cpp singlebinomial.h
liblebrt_la_SOURCES = lebrt.cpp lebrt.h
