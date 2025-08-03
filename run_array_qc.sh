

# This script runs the QC process using PLINK on the lifted over merged PLINK files generated
# using liftover_plinks_bed.wdl script

# This script is delivered "As-Is". Notwithstanding anything to the contrary, DNAnexus will have no warranty, 
# support, liability or other obligations with respect to Materials provided hereunder.
# MIT License(https://github.com/dnanexus/UKB_RAP/blob/main/LICENSE) applies to this script.


# How to Run:
# Run this shell script using:
# sh run_array_qc.sh
# on the command line on your machine



# Outputs:
# - List of variants to use in regenie GWAS step 1 (/Data/array_qc/imputed_array_snps_qc_pass.snplist)
# - Log file (/Data/array_qc/imputed_array_snps_qc_pass.log)
# - List of samples remained after filtering (/Data/array_qc/imputed_array_snps_qc_pass.id)

#!/bin/sh

# Set the data file directory (location of merged files)
data_file_dir="/new_GWAS"

# Path to the phenotype file
phenotype_file="/train_cpsp.phe"

# Command to run PLINK QC
run_plink_qc="plink2 --bfile ukb_c1-22_GRCh38_full_analysis_set_plus_decoy_hla_merged \
 --keep train_cpsp.phe --autosome \
 --maf 0.01 --mac 20 --geno 0.1 --hwe 1e-15 \
 --mind 0.1 --write-snplist --write-samples \
 --no-id-header --out array_snps_qc_pass"

# DNAnexus run command
dx run swiss-army-knife -iin="${data_file_dir}/ukb_c1-22_GRCh38_full_analysis_set_plus_decoy_hla_merged.bed" \
   -iin="${data_file_dir}/ukb_c1-22_GRCh38_full_analysis_set_plus_decoy_hla_merged.bim" \
   -iin="${data_file_dir}/ukb_c1-22_GRCh38_full_analysis_set_plus_decoy_hla_merged.fam" \
   -iin="${phenotype_file}" \
   -icmd="${run_plink_qc}" --tag="Array QC" --instance-type "mem1_ssd1_v2_x36" \
   --destination="${data_file_dir}/" --brief --yes