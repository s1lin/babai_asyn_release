# CILS Solver

A C++ implementation of the CILS (Constrained Integer Least Squares) solver with support for various optimization algorithms.

## Features

- **OpenMP Support**: Parallel processing capabilities
- **Boost Integration**: Uses Boost libraries for enhanced functionality
- **Optional MATLAB Support**: Can integrate with MATLAB Engine
- **Optional MPI Support**: Distributed computing capabilities
- **Cross-Platform**: Supports Windows, Linux, and WSL

## Prerequisites

### Required Dependencies

- **CMake** (version 3.16 or higher)
- **C++17 compatible compiler** (GCC 7+, Clang 5+, or MSVC 2017+)
- **Boost** (system and filesystem components)
- **OpenMP** (usually included with modern compilers)

### Optional Dependencies

- **MATLAB** (for MATLAB Engine integration)
- **MPI** (for distributed computing)

## Installation

### Windows

#### Option 1: Using PowerShell (Recommended)

```powershell
# Basic build
.\build_windows.ps1

# Clean build
.\build_windows.ps1 -Clean

# Build with MATLAB support
.\build_windows.ps1 -EnableMatlab

# Build with MPI support
.\build_windows.ps1 -EnableMPI

# Build in WSL from Windows
.\build_windows.ps1 -WSL
```

#### Option 2: Using Command Prompt

```cmd
# Basic build
build_windows.bat

# Clean build
build_windows.bat clean
```

#### Option 3: Manual CMake

```cmd
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --parallel
```

### WSL (Windows Subsystem for Linux)

#### Option 1: Using the WSL Build Script

```bash
# Make the script executable
chmod +x build_wsl.sh

# Basic build
./build_wsl.sh

# Clean build
./build_wsl.sh clean

# Build with test arguments
./build_wsl.sh 128 1 1 1 1
```

#### Option 2: Manual Build in WSL

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake libboost-all-dev libomp-dev

# Build the project
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake libboost-all-dev libomp-dev

# Build the project
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Configuration Options

The CMake configuration supports several options:

- `CMAKE_BUILD_TYPE`: Set to `Debug`, `Release`, `RelWithDebInfo`, or `MinSizeRel`
- `ENABLE_MATLAB`: Enable MATLAB Engine support (ON/OFF)
- `ENABLE_MPI`: Enable MPI support (ON/OFF)

### Example CMake Configuration

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MATLAB=ON \
    -DENABLE_MPI=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Usage

### Running the Executable

The built executable `cils` expects command-line arguments:

```bash
./cils <size_n> <is_local> <info> <sec> <blk>
```

Where:
- `size_n`: Problem size
- `is_local`: Local optimization flag
- `info`: Information level
- `sec`: Section parameter
- `blk`: Block size parameter

### Example Usage

```bash
# Run with test parameters
./cils 128 1 1 1 1

# Run with different parameters
./cils 256 0 2 1 2
```

## Project Structure

```
babai_asyn_c++/
├── CMakeLists.txt          # Main CMake configuration
├── main.cpp                # Main entry point
├── build_wsl.sh.in         # WSL build script template
├── build_windows.bat       # Windows batch build script
├── build_windows.ps1       # Windows PowerShell build script
├── README.md               # This file
└── src/
    ├── include/            # Header files
    ├── source/             # Source files
    └── example/            # Example implementations
```

## Troubleshooting

### Common Issues

1. **CMake version too old**
   - Update CMake to version 3.16 or higher

2. **Boost not found**
   - Install Boost development libraries
   - On Ubuntu: `sudo apt install libboost-all-dev`
   - On Windows: Use vcpkg or download from boost.org

3. **OpenMP not found**
   - Install OpenMP development package
   - On Ubuntu: `sudo apt install libomp-dev`
   - On Windows: Use a compiler that supports OpenMP

4. **MATLAB Engine not found**
   - Ensure MATLAB is installed and in PATH
   - Set `ENABLE_MATLAB=OFF` if MATLAB is not needed

5. **WSL build issues**
   - Ensure WSL is properly installed and configured
   - Install required packages in WSL: `sudo apt install build-essential cmake libboost-all-dev`

### Build Scripts

The project includes several build scripts for different environments:

- `build_wsl.sh`: Linux/WSL build script
- `build_windows.bat`: Windows command prompt build script
- `build_windows.ps1`: Windows PowerShell build script with advanced features

## Development

### Adding New Features

1. Add source files to `src/source/`
2. Add headers to `src/include/`
3. Update `CMakeLists.txt` if new dependencies are needed
4. Test on multiple platforms

### Code Style

- Use C++17 features
- Follow modern CMake practices
- Include proper error handling
- Add comments for complex algorithms

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
