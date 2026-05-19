CC = gcc
CFLAGS = -shared -fPIC -O2 -Wall

all: libcuda.so.1 libcudart.so.11.0

libcuda.so.1: libcuda_stub.c
	$(CC) $(CFLAGS) -o $@ $<

libcudart.so.11.0: libcudart_stub.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f libcuda.so.1 libcudart.so.11.0

.PHONY: all clean