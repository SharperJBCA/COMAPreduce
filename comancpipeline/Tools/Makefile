QUIJOTEFCOMP=gfortran
QUIJOTEINC=
QUIJOTELIB=-L/star/lib/

LIBS=-lsla
FLAGS=

all: pysla

pysla:

	f2py -c --f90exec=$(QUIJOTEFCOMP) $(FLAGS) -m pysla pysla.f90 $(QUIJOTEINC) $(QUIJOTELIB) $(LIBS)
