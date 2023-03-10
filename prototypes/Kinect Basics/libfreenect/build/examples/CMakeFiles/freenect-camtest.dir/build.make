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
CMAKE_BINARY_DIR = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build"

# Include any dependencies generated for this target.
include examples/CMakeFiles/freenect-camtest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/freenect-camtest.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/freenect-camtest.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/freenect-camtest.dir/flags.make

examples/CMakeFiles/freenect-camtest.dir/camtest.c.o: examples/CMakeFiles/freenect-camtest.dir/flags.make
examples/CMakeFiles/freenect-camtest.dir/camtest.c.o: ../examples/camtest.c
examples/CMakeFiles/freenect-camtest.dir/camtest.c.o: examples/CMakeFiles/freenect-camtest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/CMakeFiles/freenect-camtest.dir/camtest.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/examples" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT examples/CMakeFiles/freenect-camtest.dir/camtest.c.o -MF CMakeFiles/freenect-camtest.dir/camtest.c.o.d -o CMakeFiles/freenect-camtest.dir/camtest.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples/camtest.c"

examples/CMakeFiles/freenect-camtest.dir/camtest.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect-camtest.dir/camtest.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/examples" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples/camtest.c" > CMakeFiles/freenect-camtest.dir/camtest.c.i

examples/CMakeFiles/freenect-camtest.dir/camtest.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect-camtest.dir/camtest.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/examples" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples/camtest.c" -o CMakeFiles/freenect-camtest.dir/camtest.c.s

# Object files for target freenect-camtest
freenect__camtest_OBJECTS = \
"CMakeFiles/freenect-camtest.dir/camtest.c.o"

# External object files for target freenect-camtest
freenect__camtest_EXTERNAL_OBJECTS =

bin/freenect-camtest: examples/CMakeFiles/freenect-camtest.dir/camtest.c.o
bin/freenect-camtest: examples/CMakeFiles/freenect-camtest.dir/build.make
bin/freenect-camtest: lib/libfreenect.0.6.3.dylib
bin/freenect-camtest: /opt/homebrew/lib/libusb-1.0.dylib
bin/freenect-camtest: examples/CMakeFiles/freenect-camtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../bin/freenect-camtest"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/examples" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect-camtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/freenect-camtest.dir/build: bin/freenect-camtest
.PHONY : examples/CMakeFiles/freenect-camtest.dir/build

examples/CMakeFiles/freenect-camtest.dir/clean:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/examples" && $(CMAKE_COMMAND) -P CMakeFiles/freenect-camtest.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/freenect-camtest.dir/clean

examples/CMakeFiles/freenect-camtest.dir/depend:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/examples" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/examples" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/examples/CMakeFiles/freenect-camtest.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : examples/CMakeFiles/freenect-camtest.dir/depend

