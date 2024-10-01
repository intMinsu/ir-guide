import jetson_inference
import jetson_utils
import subprocess

def show_args(func_name, so_file):
    # Run `nm` to get the mangled names of the function from the shared object file
    try:
        # Call nm and filter out the lines with the function name
        nm_output = subprocess.check_output(f"nm -D {so_file} | grep {func_name}", shell=True).decode('utf-8').strip()

        # Print mangled output from nm
        print(f"Mangled output for {func_name} from {so_file}:")
        print(nm_output)
        print("")

        # Now demangle the output using c++filt
        mangled_names = nm_output.split("\n")
        for mangled_name in mangled_names:
            mangled_symbol = mangled_name.split()[-1]  # The mangled name is typically at the end of the line
            demangled = subprocess.check_output(f"echo {mangled_symbol} | c++filt", shell=True).decode('utf-8').strip()

            # Print the demangled function name
            print(f"Demangled output: {demangled}")
        print("")

    except subprocess.CalledProcessError as e:
        print(f"Error running nm or c++filt: {e}")

so_file_dir = "~/anaconda3/envs/pytorch210/lib/python3.8/site-packages"

show_args('saveImage', so_file_dir + '/' + 'jetson_utils_python.so')
# Mangled output for saveImage from ~/anaconda3/envs/pytorch210/lib/python3.8/site-packages/jetson_utils_python.so:
# U _Z13saveImageRGBAPKcP6float4iifiP11CUstream_st
#                  U _Z9saveImagePKcPvii11imageFormatiP11CUstream_st
#                  U _Z9saveImagePKcPvii11imageFormatiRK6float2bP11CUstream_st

# Demangled output: saveImageRGBA(char const*, float4*, int, int, float, int, CUstream_st*)
# Demangled output: saveImage(char const*, void*, int, int, imageFormat, int, CUstream_st*)
# Demangled output: saveImage(char const*, void*, int, int, imageFormat, int, float2 const&, bool, CUstream_st*)
