# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.23.0/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.23.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect"

# Include any dependencies generated for this target.
include examples/CMakeFiles/freenect-regview.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/freenect-regview.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/freenect-regview.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/freenect-regview.dir/flags.make

examples/CMakeFiles/freenect-regview.dir/regview.c.o: examples/CMakeFiles/freenect-regview.dir/flags.make
examples/CMakeFiles/freenect-regview.dir/regview.c.o: examples/regview.c
examples/CMakeFiles/freenect-regview.dir/regview.c.o: examples/CMakeFiles/freenect-regview.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/CMakeFiles/freenect-regview.dir/regview.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT examples/CMakeFiles/freenect-regview.dir/regview.c.o -MF CMakeFiles/freenect-regview.dir/regview.c.o.d -o CMakeFiles/freenect-regview.dir/regview.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples/regview.c"

examples/CMakeFiles/freenect-regview.dir/regview.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect-regview.dir/regview.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples/regview.c" > CMakeFiles/freenect-regview.dir/regview.c.i

examples/CMakeFiles/freenect-regview.dir/regview.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect-regview.dir/regview.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples/regview.c" -o CMakeFiles/freenect-regview.dir/regview.c.s

# Object files for target freenect-regview
freenect__regview_OBJECTS = \
"CMakeFiles/freenect-regview.dir/regview.c.o"

# External object files for target freenect-regview
freenect__regview_EXTERNAL_OBJECTS =

bin/freenect-regview: examples/CMakeFiles/freenect-regview.dir/regview.c.o
bin/freenect-regview: examples/CMakeFiles/freenect-regview.dir/build.make
bin/freenect-regview: lib/libfreenect.0.6.3.dylib
bin/freenect-regview: /opt/homebrew/lib/libusb-1.0.dylib
bin/freenect-regview: examples/CMakeFiles/freenect-regview.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../bin/freenect-regview"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect-regview.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/freenect-regview.dir/build: bin/freenect-regview
.PHONY : examples/CMakeFiles/freenect-regview.dir/build

examples/CMakeFiles/freenect-regview.dir/clean:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" && $(CMAKE_COMMAND) -P CMakeFiles/freenect-regview.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/freenect-regview.dir/clean

examples/CMakeFiles/freenect-regview.dir/depend:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples/CMakeFiles/freenect-regview.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : examples/CMakeFiles/freenect-regview.dir/depend

