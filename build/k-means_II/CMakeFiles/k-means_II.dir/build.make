# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/goldpancake/Documents/K-means_II

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/goldpancake/Documents/K-means_II/build

# Include any dependencies generated for this target.
include k-means_II/CMakeFiles/k-means_II.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include k-means_II/CMakeFiles/k-means_II.dir/compiler_depend.make

# Include the progress variables for this target.
include k-means_II/CMakeFiles/k-means_II.dir/progress.make

# Include the compile flags for this target's objects.
include k-means_II/CMakeFiles/k-means_II.dir/flags.make

k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.o: k-means_II/CMakeFiles/k-means_II.dir/flags.make
k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.o: ../k-means_II/src/kmeans.cpp
k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.o: k-means_II/CMakeFiles/k-means_II.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/goldpancake/Documents/K-means_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.o"
	cd /home/goldpancake/Documents/K-means_II/build/k-means_II && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.o -MF CMakeFiles/k-means_II.dir/src/kmeans.cpp.o.d -o CMakeFiles/k-means_II.dir/src/kmeans.cpp.o -c /home/goldpancake/Documents/K-means_II/k-means_II/src/kmeans.cpp

k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/k-means_II.dir/src/kmeans.cpp.i"
	cd /home/goldpancake/Documents/K-means_II/build/k-means_II && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/goldpancake/Documents/K-means_II/k-means_II/src/kmeans.cpp > CMakeFiles/k-means_II.dir/src/kmeans.cpp.i

k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/k-means_II.dir/src/kmeans.cpp.s"
	cd /home/goldpancake/Documents/K-means_II/build/k-means_II && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/goldpancake/Documents/K-means_II/k-means_II/src/kmeans.cpp -o CMakeFiles/k-means_II.dir/src/kmeans.cpp.s

# Object files for target k-means_II
k__means_II_OBJECTS = \
"CMakeFiles/k-means_II.dir/src/kmeans.cpp.o"

# External object files for target k-means_II
k__means_II_EXTERNAL_OBJECTS =

k-means_II/libk-means_II.so: k-means_II/CMakeFiles/k-means_II.dir/src/kmeans.cpp.o
k-means_II/libk-means_II.so: k-means_II/CMakeFiles/k-means_II.dir/build.make
k-means_II/libk-means_II.so: k-means_II/CMakeFiles/k-means_II.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/goldpancake/Documents/K-means_II/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libk-means_II.so"
	cd /home/goldpancake/Documents/K-means_II/build/k-means_II && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/k-means_II.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
k-means_II/CMakeFiles/k-means_II.dir/build: k-means_II/libk-means_II.so
.PHONY : k-means_II/CMakeFiles/k-means_II.dir/build

k-means_II/CMakeFiles/k-means_II.dir/clean:
	cd /home/goldpancake/Documents/K-means_II/build/k-means_II && $(CMAKE_COMMAND) -P CMakeFiles/k-means_II.dir/cmake_clean.cmake
.PHONY : k-means_II/CMakeFiles/k-means_II.dir/clean

k-means_II/CMakeFiles/k-means_II.dir/depend:
	cd /home/goldpancake/Documents/K-means_II/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/goldpancake/Documents/K-means_II /home/goldpancake/Documents/K-means_II/k-means_II /home/goldpancake/Documents/K-means_II/build /home/goldpancake/Documents/K-means_II/build/k-means_II /home/goldpancake/Documents/K-means_II/build/k-means_II/CMakeFiles/k-means_II.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : k-means_II/CMakeFiles/k-means_II.dir/depend

