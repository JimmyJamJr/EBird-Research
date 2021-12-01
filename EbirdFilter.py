# This script takes the main NV data file and split it into files for
# individual birds

import os


# Remove punctuation from string
def clean_name(name):
    name = name.replace("/", "-")
    name = name.replace("\'", "")
    return name


# Separate species into individual files
def find_species(filename, bird_folder, replace, match_list=[]):
    # Delete all existing files if replace is on
    if replace:
        for file in os.scandir(os.path.join(os.getcwd(), bird_folder)):
            os.remove(file.path)

    # Dictionary to store logging info
    species_dict = {}

    line_count = 0

    with open(filename) as in_file:
        next(in_file)

        # iterate through file
        for line in in_file:

            # line_count += 1
            # if line_count > 2000:
            #     break

            # iterate through line
            line_arr = line.split("\t")

            # Check with match list
            if len(match_list) == 0 or line_arr[4] in match_list:

                # Add to species dictionary
                species = clean_name(line_arr[4])

                if species in species_dict.keys():
                    species_dict[species] += 1
                    write_mode = "a"
                else:
                    species_dict[species] = 1
                    write_mode = "w"

                # Open species file
                out_file_name = os.path.join(os.getcwd(), bird_folder, species + ".txt")
                with open(out_file_name, write_mode) as out_file:
                    out_file.write(line)
                    print("Found", species)

    print(species_dict)


find_species(filename="ebd_US-NV_relFeb-2021.txt", bird_folder="birds", replace=True, match_list=[])




