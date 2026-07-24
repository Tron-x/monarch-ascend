CANN_HOME ?= $(shell echo $${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest})
HIXL_INC  ?= $(CANN_HOME)/include
HIXL_LIB  ?= $(CANN_HOME)/lib64

CXX       := g++
CC        := gcc
CXXFLAGS  := -std=c++17 -O2 -I$(HIXL_INC)
CFLAGS    := -std=c11 -O2
LDFLAGS   := -L$(HIXL_LIB) -lcann_hixl -lmetadef -lascendcl -lc_sec -lpthread -ldl -Wl,-rpath,$(HIXL_LIB)

BUILDDIR  := build

NATIVE_CPP := $(wildcard native/*.cpp)
NATIVE_C   := $(wildcard native/*.c)
UTIL_CPP   := $(wildcard util/*.cpp)
UTIL_C     := $(wildcard util/*.c)

NATIVE_BINS := $(patsubst native/%.cpp,$(BUILDDIR)/%,$(NATIVE_CPP)) \
               $(patsubst native/%.c,$(BUILDDIR)/%,$(NATIVE_C))
UTIL_LIBS   := $(patsubst util/%.cpp,$(BUILDDIR)/lib%.so,$(UTIL_CPP)) \
               $(patsubst util/%.c,$(BUILDDIR)/lib%.so,$(UTIL_C))

.PHONY: all clean run-connect run-transfer run-d2d

all: $(BUILDDIR) $(NATIVE_BINS) $(UTIL_LIBS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%: native/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BUILDDIR)/%: native/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ $< -lpthread -ldl

$(BUILDDIR)/lib%.so: util/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $< $(LDFLAGS)

$(BUILDDIR)/lib%.so: util/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $< -ldl

clean:
	rm -f $(BUILDDIR)/test_hixl_* $(BUILDDIR)/lib*.so

# --- Run targets ---

run-connect: $(BUILDDIR)/test_hixl_connect
	HCCL_INTRA_ROCE_ENABLE=1 $(BUILDDIR)/test_hixl_connect

run-transfer: $(BUILDDIR)/test_hixl_transfer
	HCCL_INTRA_ROCE_ENABLE=1 $(BUILDDIR)/test_hixl_transfer

run-d2d: $(BUILDDIR)/test_hixl_d2d
	HCCL_INTRA_ROCE_ENABLE=1 $(BUILDDIR)/test_hixl_d2d
