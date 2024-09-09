all: extra_tiny_cross_entropy extra_tiny_cifar100 extra_tiny_go

extra_tiny_cross_entropy: extra_tiny_cross_entropy.cpp
	g++ -Wall -g -rdynamic -std=c++11 -o extra_tiny_cross_entropy extra_tiny_cross_entropy.cpp

extra_tiny_cifar100: extra_tiny_cifar100.cpp png_simplified.hpp
ifdef USE_BLAS
ifdef USE_PNG
	g++ -DUSE_PNG -Wall -g -rdynamic -std=c++11 -o extra_tiny_cifar100 extra_tiny_cifar100.cpp -lpthread -lopenblas -lpng
else
	g++ -DUSE_BLAS -Wall -g -rdynamic -std=c++11 -o extra_tiny_cifar100 extra_tiny_cifar100.cpp -lpthread -lopenblas
endif
else
ifdef USE_PNG
	g++ -DUSE_PNG -Wall -g -rdynamic -std=c++11 -o extra_tiny_cifar100 extra_tiny_cifar100.cpp -lpthread -lpng
else
	g++ -Wall -g -rdynamic -std=c++11 -o extra_tiny_cifar100 extra_tiny_cifar100.cpp -lpthread
endif
endif

extra_tiny_go: extra_tiny_go.cpp
ifdef USE_BLAS
	g++ -DUSE_BLAS -Wall -g -rdynamic -std=c++11 -o extra_tiny_go extra_tiny_go.cpp -lpthread -lopenblas
else
	g++ -Wall -g -rdynamic -std=c++11 -o extra_tiny_go extra_tiny_go.cpp -lpthread
endif
