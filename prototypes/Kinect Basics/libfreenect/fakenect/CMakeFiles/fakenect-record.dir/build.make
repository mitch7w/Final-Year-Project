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
include fakenect/CMakeFiles/fakenect-record.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include fakenect/CMakeFiles/fakenect-record.dir/compiler_depend.make

# Include the progress variables for this target.
include fakenect/CMakeFiles/fakenect-record.dir/progress.make

# Include the compile flags for this target's objects.
include fakenect/CMakeFiles/fakenect-record.dir/flags.make

fakenect/CMakeFiles/fakenect-record.dir/record.c.o: fakenect/CMakeFiles/fakenect-record.dir/flags.make
fakenect/CMakeFiles/fakenect-record.dir/record.c.o: fakenect/record.c
fakenect/CMakeFiles/fakenect-record.dir/record.c.o: fakenect/CMakeFiles/fakenect-record.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object fakenect/CMakeFiles/fakenect-record.dir/record.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT fakenect/CMakeFiles/fakenect-record.dir/record.c.o -MF CMakeFiles/fakenect-record.dir/record.c.o.d -o CMakeFiles/fakenect-record.dir/record.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect/record.c"

fakenect/CMakeFiles/fakenect-record.dir/record.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/fakenect-record.dir/record.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect/record.c" > CMakeFiles/fakenect-record.dir/record.c.i

fakenect/CMakeFiles/fakenect-record.dir/record.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/fakenect-record.dir/record.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect/record.c" -o CMakeFiles/fakenect-record.dir/record.c.s

fakenect/CMakeFiles/fakenect-record.dir/parson.c.o: fakenect/CMakeFiles/fakenect-record.dir/flags.make
fakenect/CMakeFiles/fakenect-record.dir/parson.c.o: fakenect/parson.c
fakenect/CMakeFiles/fakenect-record.dir/parson.c.o: fakenect/CMakeFiles/fakenect-record.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building C object fakenect/CMakeFiles/fakenect-record.dir/parson.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT fakenect/CMakeFiles/fakenect-record.dir/parson.c.o -MF CMakeFiles/fakenect-record.dir/parson.c.o.d -o CMakeFiles/fakenect-record.dir/parson.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect/parson.c"

fakenect/CMakeFiles/fakenect-record.dir/parson.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/fakenect-record.dir/parson.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect/parson.c" > CMakeFiles/fakenect-record.dir/parson.c.i

fakenect/CMakeFiles/fakenect-record.dir/parson.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/fakenect-record.dir/parson.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect/parson.c" -o CMakeFiles/fakenect-record.dir/parson.c.s

# Object files for target fakenect-record
fakenect__record_OBJECTS = \
"CMakeFiles/fakenect-record.dir/record.c.o" \
"CMakeFiles/fakenect-record.dir/parson.c.o"

# External object files for target fakenect-record
fakenect__record_EXTERNAL_OBJECTS =

bin/fakenect-record: fakenect/CMakeFiles/fakenect-record.dir/record.c.o
bin/fakenect-record: fakenect/CMakeFiles/fakenect-record.dir/parson.c.o
bin/fakenect-record: fakenect/CMakeFiles/fakenect-record.dir/build.make
bin/fakenect-record: lib/libfreenect.0.6.3.dylib
bin/fakenect-record: /opt/homebrew/lib/libusb-1.0.dylib
bin/fakenect-record: fakenect/CMakeFiles/fakenect-record.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable ../bin/fakenect-record"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fakenect-record.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
fakenect/CMakeFiles/fakenect-record.dir/build: bin/fakenect-record
.PHONY : fakenect/CMakeFiles/fakenect-record.dir/build

fakenect/CMakeFiles/fakenect-record.dir/clean:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" && $(CMAKE_COMMAND) -P CMakeFiles/fakenect-record.dir/cmake_clean.cmake
.PHONY : fakenect/CMakeFiles/fakenect-record.dir/clean

fakenect/CMakeFiles/fakenect-record.dir/depend:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/fakenect/CMakeFiles/fakenect-record.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : fakenect/CMakeFiles/fakenect-record.dir/depend

