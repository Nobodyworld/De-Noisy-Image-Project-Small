import json
import os
import re

# Set root_directory to the directory where this script is located
root_directory = os.path.dirname(os.path.abspath(__file__))
path_pattern = re.compile(r'[\w_]+\s*=\s*[rR]?["\'][^"\']+[\'"]')
paths = {}

def read_file_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            return f.readlines()

for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            content = read_file_lines(file_path)
            for line in content:
                match = path_pattern.match(line.strip())
                if match:
                    parts = match.group(0).split('=', 1)
                    if len(parts) == 2:
                        variable, path = parts[0].strip(), parts[1].strip().strip('"').strip("'")
                        if file not in paths:
                            paths[file] = []
                        paths[file].append({variable: path})

output_path = os.path.join(root_directory, 'extracted_paths.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(paths, f, indent=4)

print(f"Paths extracted to {output_path}")
