#!/bin/bash

# Run the command using DNAnexus swiss-army-knife
dx run swiss-army-knife \
  -icmd='
    input_dir="/REGENIE_output/Imputation from genotype (GEL)"
    output_dir="/REGENIE_output/GEL_imputed_sample_files_fixed"

    # Loop through each .sample file
    for file_id in $(dx find data --path "$input_dir" --name "*.sample" --brief); do
        # Get the file name
        file_name=$(dx describe "$file_id" --name)
        echo "Processing file: $file_name"

        # Process the file: modify the second line
        dx cat "$file_id" | awk '\''NR == 2 {print "0 0 0 D"} NR != 2 {print}'\'' > "${file_name}_fixed"

        # Upload the processed file to the output directory
        dx upload "${file_name}_fixed" --path "$output_dir/" --brief
    done
  ' \
  --instance-type "mem3_ssd3_x4" \
  --destination "/REGENIE_output/GEL_imputed_sample_files_fixed/" \
  --brief --yes
