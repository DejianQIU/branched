import os
import re


def needs_hawk_addition(filename):
    """
    Check if the filename needs 'hawk' to be added
    Returns True if 'hawk' is not present and the file follows the expected pattern
    """
    # Check if file is a CSV
    if not filename.endswith('.csv'):
        return False

    # Check if 'hawk' is already present
    if 'hawk' in filename:
        return False

    # Check if file follows the expected pattern
    pattern = r'.*_xp.*_Np\d+_BpP\d+_mcl\d+_bl\d+_nb\d+_v\d+_wgh\d+_pw\d+\.csv$'
    return bool(re.match(pattern, filename))


def add_hawk_to_filename(filename):
    """
    Add 'hawk' to the filename between BpP and mcl segments
    """
    # Split the filename at underscores
    parts = filename.split('_')

    # Find the index of the BpP part
    bpp_index = next(i for i, part in enumerate(parts) if part.startswith('BpP'))

    # Insert 'hawk' after BpP
    parts.insert(bpp_index + 1, 'hawk')

    # Join the parts back together
    return '_'.join(parts)


def main():
    # Get the parent directory where the script is located
    parent_dir = os.getcwd()

    # List of folders to process
    folders_to_process = ['dropsize', 'polymer_adsorption', 'rdensity', 'surface_coverage']

    # Process each folder
    for folder in folders_to_process:
        folder_path = os.path.join(parent_dir, folder)

        # Skip if folder doesn't exist
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder}")
            continue

        print(f"\nProcessing folder: {folder}")

        # Process each file in the folder
        for filename in os.listdir(folder_path):
            if needs_hawk_addition(filename):
                new_filename = add_hawk_to_filename(filename)
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")


if __name__ == "__main__":
    main()