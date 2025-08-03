#!/bin/bash

# Path to .bgen files on DNAnexus
bgen_dir="/REGENIE_output/Imputation from genotype (GEL)"

# Command to index all .bgen files
for i in "${bgen_dir}"/*.bgen; do
    bgenix -index -g "${i}"
done

# Run the command using DNAnexus swiss-army-knife
dx run swiss-army-knife \
  -iin="${bgen_dir}/*.bgen" \
  -icmd="for i in /REGENIE_output/Imputation\ from\ genotype\ \(GEL\)/*.bgen; do bgenix -index -g \${i}; done" \
  --instance-type "mem1_ssd1_v2_x36" \
  --destination="/REGENIE_output/Imputation from genotype (GEL)/" \
  --brief --yes
