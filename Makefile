.PHONY: all examples

all: examples

examples:
ifdef USE_BLAS
ifdef USE_PNG
	$(MAKE) USE_BLAS=1 USE_PNG=1 -C $@
else
	$(MAKE) USE_BLAS=1 -C $@
endif
else
ifdef USE_PNG
	$(MAKE) USE_PNG=1 -C $@
else
	$(MAKE) -C $@
endif
endif
