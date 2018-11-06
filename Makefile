# names of .cc files that have a main() function
TARGETS = gauss

# names of other .cc files
CXXFILES = # none for now

# Let the programmer choose 32 or 64 bits, but default to 64 bit
BITS ?= 64

# Output folder
ODIR := ./obj$(BITS)
output_folder := $(shell mkdir -p $(ODIR))

# Names of the .o, .d, and .exe files
COMMONOFILES = $(patsubst %, $(ODIR)/%.o, $(CXXFILES))
ALLOFILES    = $(patsubst %, $(ODIR)/%.o, $(CXXFILES) $(TARGETS))
EXEFILES     = $(patsubst %, $(ODIR)/%.exe, $(TARGETS))
DFILES       = $(patsubst %.o, %.d, $(ALLOFILES))

# Basic tool configuration for GCC/G++.
#
# Note: These lines will require some changes in order to work with TBB
CXX      = icpc
LD       = icpc
CXXFLAGS = -MMD -O3 -m$(BITS) -ggdb -std=c++1y -Wall -L /usr/local/tbb -ltbb
LDFLAGS  = -m$(BITS) -lm -std=c++1y -Wall -L /usr/local/tbb -ltbb

# Standard build targets and rules follow
.DEFAULT_GOAL = all
.PRECIOUS: $(ALLOFILES)
.PHONY: all clean

all: $(EXEFILES)

$(ODIR)/%.o: %.cc
	@echo "[CXX] $< --> $@"
	@$(CXX) $< -o $@ -c $(CXXFLAGS)

$(ODIR)/%.exe: $(ODIR)/%.o $(COMMONOFILES)
	@echo "[LD] $^ --> $@"
	@$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	@echo Cleaning up...
	@rm -rf $(ODIR)

-include $(DFILES)
