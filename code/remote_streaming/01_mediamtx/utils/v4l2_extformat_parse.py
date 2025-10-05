import re
import subprocess
from fractions import Fraction  # To convert float fps to fraction
import json

def get_v4l2_formats(device='/dev/video0'):
    # Run the 'v4l2-ctl --list-formats-ext' command
    command = ['v4l2-ctl', '-d', device, '--list-formats-ext']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        return {}
    
    # Capture the output
    output = result.stdout
    
    # Initialize dictionary to store formats
    formats_dict = {}
    
    # Regular expression patterns to match
    format_pattern = re.compile(r"\s*\[\d+\]:\s+'(\w+)'")
    resolution_pattern = re.compile(r"\s+Size: Discrete\s+(\d+x\d+)")
    framerate_pattern = re.compile(r"\s+Interval: Discrete\s+\d+\.\d+s\s+\((\d+\.\d+) fps\)")
    
    current_format = None
    
    # Parse the output line by line
    for line in output.splitlines():
        # Match format
        format_match = format_pattern.match(line)
        if format_match:
            current_format = format_match.group(1)
            formats_dict[current_format] = []
        
        # Match resolution
        resolution_match = resolution_pattern.search(line)
        if resolution_match and current_format:
            resolution = resolution_match.group(1)
            formats_dict[current_format].append({'resolution': resolution, 'framerates': []})
        
        # Match framerate
        framerate_match = framerate_pattern.search(line)
        if framerate_match and current_format:
            fps_float = float(framerate_match.group(1))  # Capture the fps value as a float
            
            # Convert the float fps to a fraction (e.g., 7.5 becomes 15/2)
            fps_fraction = Fraction(fps_float).limit_denominator(1000)
            
            # Format the fraction as a string 'numerator/denominator'
            framerate_fraction = f"{fps_fraction.numerator}/{fps_fraction.denominator}"
            
            if formats_dict[current_format]:
                formats_dict[current_format][-1]['framerates'].append(framerate_fraction)
    
    return formats_dict

# Example usage
formats_dict = get_v4l2_formats('/dev/video0')

# Print the resulting dictionary
for format_name, resolutions in formats_dict.items():
    print(f"Format: {format_name}")
    for resolution_info in resolutions:
        print(f"  Resolution: {resolution_info['resolution']}, Framerates: {resolution_info['framerates']}")

# Save it to a JSON file for later use
with open('../camera_formats.json', 'w') as f:
    json.dump(formats_dict, f, indent=4)

print("Camera formats saved to 'camera_formats.json'")