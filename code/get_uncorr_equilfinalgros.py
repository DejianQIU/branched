#!/usr/local/bin/python3.10

import os
import subprocess as subp
import glob

# Define the base directory
base_dir = "../branched_polymer"

# Get all xp directories
xp_dirs = glob.glob(os.path.join(base_dir, "xp*"))

for xp_dir in xp_dirs:
    # Get all mcl directories
    mcl_dirs = glob.glob(os.path.join(xp_dir, "mcl*"))

    for mcl_dir in mcl_dirs:
        # Get all branch length and number directories
        branch_dirs = glob.glob(os.path.join(mcl_dir, "branch_length*_num_branches*"))

        for branch_dir in branch_dirs:
            equil_dir = os.path.join(branch_dir, "5_equil")

            if os.path.isdir(equil_dir):
                control_file = os.path.join(equil_dir, "control.in")

                if os.path.isfile(control_file):
                    with open(control_file, 'r') as f:
                        lines = f.readlines()

                    # Initialize variables
                    totalsteps = None
                    xtrjfreq = None

                    # Read the control.in file for totalsteps and xtrjfreq
                    for line in lines:
                        if line.startswith("totalsteps"):
                            parts = line.split()
                            if len(parts) >= 2:
                                totalsteps = int(parts[1].strip())
                            else:
                                print(f"Unexpected format for totalsteps line: {line.strip()}")
                        elif line.startswith("xtrjfreq"):
                            parts = line.split()
                            if len(parts) >= 2:
                                xtrjfreq = int(parts[1].strip())
                            else:
                                print(f"Unexpected format for xtrjfreq line: {line.strip()}")

                    if totalsteps is not None and xtrjfreq is not None:
                        print(f"\nProcessing directory: {equil_dir}")
                        print(f"totalsteps: {totalsteps}, xtrjfreq: {xtrjfreq}")
                        max_value = totalsteps // xtrjfreq
                        step_values = [max_value // 2 + i * ((max_value - max_value // 2) // 4) for i in range(5)]
                        print(f"max_value: {max_value}")
                        print(f"step_values: {step_values}")

                        # Execute the dpdwetting trjtogro command for each step value
                        for i, value in enumerate(step_values, start=1):
                            output_filename = f"equilfinal{i}.gro"
                            command = [
                                "dpdwetting", "trjtogro",
                                "-bs", str(value),
                                "-es", str(value),
                                "-o", output_filename
                            ]

                            # Change directory to equil_dir and execute the command
                            try:
                                result = subp.run(command, cwd=equil_dir, capture_output=True, text=True)

                                if result.returncode == 0:
                                    print(f"Successfully created: {output_filename}")
                                else:
                                    print(f"Error executing command for {output_filename}: {result.stderr}")
                            except Exception as e:
                                print(f"Exception occurred while processing {output_filename}: {str(e)}")
                    else:
                        print(f"Could not find totalsteps or xtrjfreq in {control_file}")
                else:
                    print(f"control.in not found in {equil_dir}")
            else:
                print(f"5_equil directory not found in {branch_dir}")

print("\nScript execution completed.")
