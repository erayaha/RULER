# Open the pip_list.txt and create a new requirements.txt file
with open('pip_list.txt', 'r') as infile, open('requirements.txt', 'w') as outfile:
    lines = infile.readlines()
    for line in lines:
        if 'Package' in line or '---' in line:
            # Skip the header and separator lines
            continue
        # Split the line into package and version
        package_info = line.split()
        if len(package_info) == 2:
            package, version = package_info
            outfile.write(f'{package}=={version}\n')
