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
include wrappers/c_sync/CMakeFiles/freenect_sync.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include wrappers/c_sync/CMakeFiles/freenect_sync.dir/compiler_depend.make

# Include the progress variables for this target.
include wrappers/c_sync/CMakeFiles/freenect_sync.dir/progress.make

# Include the compile flags for this target's objects.
include wrappers/c_sync/CMakeFiles/freenect_sync.dir/flags.make

wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o: wrappers/c_sync/CMakeFiles/freenect_sync.dir/flags.make
wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o: ../wrappers/c_sync/libfreenect_sync.c
wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o: wrappers/c_sync/CMakeFiles/freenect_sync.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o -MF CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o.d -o CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o -c "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/c_sync/libfreenect_sync.c"

wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/freenect_sync.dir/libfreenect_sync.c.i"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/c_sync/libfreenect_sync.c" > CMakeFiles/freenect_sync.dir/libfreenect_sync.c.i

wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/freenect_sync.dir/libfreenect_sync.c.s"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/c_sync/libfreenect_sync.c" -o CMakeFiles/freenect_sync.dir/libfreenect_sync.c.s

# Object files for target freenect_sync
freenect_sync_OBJECTS = \
"CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o"

# External object files for target freenect_sync
freenect_sync_EXTERNAL_OBJECTS =

lib/libfreenect_sync.0.6.3.dylib: wrappers/c_sync/CMakeFiles/freenect_sync.dir/libfreenect_sync.c.o
lib/libfreenect_sync.0.6.3.dylib: wrappers/c_sync/CMakeFiles/freenect_sync.dir/build.make
lib/libfreenect_sync.0.6.3.dylib: lib/libfreenect.0.6.3.dylib
lib/libfreenect_sync.0.6.3.dylib: /opt/homebrew/lib/libusb-1.0.dylib
lib/libfreenect_sync.0.6.3.dylib: wrappers/c_sync/CMakeFiles/freenect_sync.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library ../../lib/libfreenect_sync.dylib"
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/freenect_sync.dir/link.txt --verbose=$(VERBOSE)
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync" && $(CMAKE_COMMAND) -E cmake_symlink_library ../../lib/libfreenect_sync.0.6.3.dylib ../../lib/libfreenect_sync.0.dylib ../../lib/libfreenect_sync.dylib

lib/libfreenect_sync.0.dylib: lib/libfreenect_sync.0.6.3.dylib
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libfreenect_sync.0.dylib

lib/libfreenect_sync.dylib: lib/libfreenect_sync.0.6.3.dylib
	@$(CMAKE_COMMAND) -E touch_nocreate lib/libfreenect_sync.dylib

# Rule to build all files generated by this target.
wrappers/c_sync/CMakeFiles/freenect_sync.dir/build: lib/libfreenect_sync.dylib
.PHONY : wrappers/c_sync/CMakeFiles/freenect_sync.dir/build

wrappers/c_sync/CMakeFiles/freenect_sync.dir/clean:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync" && $(CMAKE_COMMAND) -P CMakeFiles/freenect_sync.dir/cmake_clean.cmake
.PHONY : wrappers/c_sync/CMakeFiles/freenect_sync.dir/clean

wrappers/c_sync/CMakeFiles/freenect_sync.dir/depend:
	cd "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/c_sync" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync" "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/build/wrappers/c_sync/CMakeFiles/freenect_sync.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : wrappers/c_sync/CMakeFiles/freenect_sync.dir/depend

