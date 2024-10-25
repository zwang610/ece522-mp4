## 
## Simple makefile for decaf programming projects
##


.PHONY: clean strip

# Set the default target. When you make with no arguments,
# this will be the target built.
COMPILER = sim
# PREPROCESSOR = dpp
# PREPROCESSOR = dpp
PRODUCTS = $(COMPILER) 
default: $(PRODUCTS)

# Set up the list of source and object files
SRCS = analysis.cc simulationComponents.cc simulationEvents.cc simulator.cc main.cc

# OBJS can deal with either .cc or .c files listed in SRCS
OBJS = $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS))) $(patsubst %.c,%.o,$(filter %.c, $(SRCS)))

JUNK = $(OBJS)

# Define the tools we are going to use
CC= g++
LD = g++

# Set up the necessary flags for the tools

# We want debugging and most warnings, but lex/yacc generate some
# static symbols we don't use, so turn off unused warnings to avoid clutter
# Also STL has some signed/unsigned comparisons we want to suppress
CFLAGS = -g -Wall -Wno-unused -Wno-sign-compare

# The -d flag tells lex to set up for debugging. Can turn on/off by
# setting value of global yy_flex_debug inside the scanner itself
LEXFLAGS = -d

# The -d flag tells yacc to generate header with token types
# The -v flag writes out a verbose description of the states and conflicts
# The -t flag turns on debugging capability
# The -y flag means imitate yacc's output file naming conventions
YACCFLAGS = -dvty

# Link with standard c library, math library, and lex library
LIBS = -lc

# Rules for various parts of the target

.cc.o: $*.cc
	$(CC) $(CFLAGS) -c -o $@ $*.cc

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $*.c

%.o: %.cc
	$(CC) $(CFLAGS) -c -o $@ $*.cc

# rules to build compiler (dcc)

$(COMPILER) : $(OBJS)
	$(LD) -o $@ $(OBJS) $(LIBS)

$(COMPILER).purify : $(OBJS)
	purify -log-file=purify.log -cache-dir=/tmp/$(USER) -leaks-at-exit=no $(LD) -o $@ $(OBJS) $(LIBS)


# This target is to build small for testing (no debugging info), removes
# all intermediate products, too
strip : $(PRODUCTS)
	strip $(PRODUCTS)
	rm -rf $(JUNK)


# make depend will set up the header file dependencies for the 
# assignment.  You should make depend whenever you add a new header
# file to the project or move the project between machines
#
depend:
	sed -i '/^# DO NOT DELETE$$/{n;q}' Makefile
	$(CC) -MM $(SRCS) $(PREP_SRCS) >> Makefile

clean:
	rm -f $(JUNK) $(PRODUCTS)

# DO NOT DELETE

ast.o: ast.cc ast.h
analysis.o: analysis.cc analysis.h
main.o: main.cc
