# Builds the fused token cross-entropy kernel.
#
# -O2 with -fno-fast-math so the kernel's double arithmetic is plain IEEE-754
# and matches a straightforward float64 reference operation for operation.

CXX ?= g++
CXXFLAGS ?= -O2 -fno-fast-math -fPIC -std=c++17

BUILD := build
SOURCE := kernels/fused_ce.cpp
TARGET := $(BUILD)/libfused_ce.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE) | $(BUILD)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(SOURCE)

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
