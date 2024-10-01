import json

# Load the camera formats from the JSON file
def load_camera_formats(file_path='camera_formats.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to check if the combination is compatible
def check_compatible_combination(camera_formats, src_format, w, h, fps):
    # Convert resolution and fps to strings (e.g., '1280x720' and '10/1')
    resolution = f"{w}x{h}"
    fps_str = fps
    
    # Check if the format is supported
    if src_format not in camera_formats:
        raise ValueError(f"Source format '{src_format}' is not supported.")
    
    # Iterate through available resolutions for the selected format
    for res_info in camera_formats[src_format]:
        if res_info['resolution'] == resolution:
            # Check if the fps is compatible with the resolution
            if fps_str in res_info['framerates']:
                print(f"Compatible combination found: {src_format}, {resolution}, {fps_str}")
                return True
    
    # If no match was found, raise an error
    raise ValueError(f"No compatible combination found for {src_format}, {resolution}, {fps_str}")

# Function to validate camera configuration
def val_camera_configuration(src_format, 
                            w, 
                            h, 
                            fps, 
                            camera_formats_file='camera_formats.json'):
    try:
        # Load the camera formats from the JSON file
        camera_formats = load_camera_formats(camera_formats_file)

        # YUYV and YUY2 are identical
        if src_format == 'YUY2':
            src_format = 'YUYV'
        
        # Check if the combination is valid
        check_compatible_combination(camera_formats, src_format, w, h, fps)
        
        return True  # Return True if valid
    except ValueError as e:
        print(f"Error: {e}")
        return False  # Return False if invalid