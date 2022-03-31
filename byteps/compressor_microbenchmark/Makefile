CC = g++
CFLAGS = -g -Wall -O2
LDFLAGS = -fopenmp

TARGET = test
DEPS = common.h compressor.h
OBJ = efsignSGD.o onebit.o randomk.o dgc.o

%.o: %.cc $(DEPS)
	$(CC) $(LDFLAGS) -c -o $@ $<


$(TARGET): $(OBJ) test.o
	$(CC) $(LDFLAGS) -o $@ $^ $(CFLAGS)

all: $(TARGET)

.PHONY: clean

clean:
	rm -f *.o $(TARGET)