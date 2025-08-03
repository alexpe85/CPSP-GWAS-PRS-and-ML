#!/bin/bash

# Input file paths
BFILE_PATH="/PRS/merged_temp_filtered"
SCORE_FILE="/PRS/snp_beta_file.txt"
PHENO_FILE="/PRS/simplified_test_set.txt"
OUTPUT_DIR="/PRS"

# Run PLINK command
plink \
    --bfile "$BFILE_PATH" \
    --score "$SCORE_FILE" 1 2 3 sum \
    --pheno "$PHENO_FILE" \
    --out "$OUTPUT_DIR/test_set_prs"
