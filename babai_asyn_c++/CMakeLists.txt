cmake_minimum_required(VERSION 3.16)
project(cils_solver VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
    
    # OpenMP support
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP found and enabled")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    message(STATUS "MSVC compiler detected - OpenMP may not be available")
else()
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # WSL or MinGW
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
elseif(UNIX AND NOT APPLE)
    # Linux settings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    # Add math library
    find_library(MATH_LIBRARY m)
endif()

# Find required packages
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Optional MATLAB support
option(ENABLE_MATLAB "Enable MATLAB support" OFF)
if(ENABLE_MATLAB)
    find_package(Matlab QUIET)
    if(Matlab_FOUND)
        message(STATUS "MATLAB found: ${Matlab_ROOT_DIR}")
    else()
        message(WARNING "MATLAB not found - MATLAB support will be disabled")
        set(ENABLE_MATLAB OFF)
    endif()
endif()

# Optional MPI support
option(ENABLE_MPI "Enable MPI support" OFF)
if(ENABLE_MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found and enabled")
    else()
        message(WARNING "MPI not found - MPI support will be disabled")
        set(ENABLE_MPI OFF)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)

if(ENABLE_MATLAB AND Matlab_FOUND)
    include_directories(${Matlab_INCLUDE_DIRS})
endif()

# Create source file lists
file(GLOB_RECURSE CILS_SOURCES 
    "src/source/*.cpp"
    "main.cpp"
)

# Create the main executable
add_executable(cils ${CILS_SOURCES})

# Set target properties
set_target_properties(cils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Link libraries
target_link_libraries(cils 
    Boost::system 
    Boost::filesystem
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cils OpenMP::OpenMP_CXX)
endif()

if(ENABLE_MATLAB AND Matlab_FOUND)
    if(WIN32)
        # Windows MATLAB libraries
        target_link_libraries(cils 
            "${Matlab_ROOT_DIR}/extern/bin/win64/microsoft/libMatlabEngine.lib"
            "${Matlab_ROOT_DIR}/extern/bin/win64/microsoft/libMatlabDataArray.lib"
        )
    else()
        # Linux/Unix MATLAB libraries
        target_link_libraries(cils 
            "${Matlab_ROOT_DIR}/extern/bin/glnxa64/libMatlabEngine.so"
            "${Matlab_ROOT_DIR}/extern/bin/glnxa64/libMatlabDataArray.so"
        )
    endif()
    message(STATUS "Linked MATLAB libraries")
endif()

if(ENABLE_MPI AND MPI_CXX_FOUND)
    target_link_libraries(cils MPI::MPI_CXX)
endif()

if(MATH_LIBRARY)
    target_link_libraries(cils ${MATH_LIBRARY})
endif()

# Installation
install(TARGETS cils
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Create a build script for WSL
if(WIN32)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/build_wsl.sh.in"
        "${CMAKE_CURRENT_BINARY_DIR}/build_wsl.sh"
        @ONLY
    )
    file(COPY "${CMAKE_CURRENT_BINARY_DIR}/build_wsl.sh" 
         DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}"
         FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endif()

# Print configuration summary
message(STATUS "=== CILS Solver Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "MATLAB: ${ENABLE_MATLAB}")
message(STATUS "MPI: ${ENABLE_MPI}")
message(STATUS "Boost: ${Boost_FOUND}")
message(STATUS "==================================")