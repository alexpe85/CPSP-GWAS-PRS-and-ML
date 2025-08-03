




data_file_dir="/new_PRS"

# Path to the phenotype file
phenotype_file="/train_cpsp.phe"


dx run(plink_train_cmd)

plink_train_cmd = (
        "./plink --bfile imputed_merged " \
        f"--score gwas_score_file.txt 1 2 3 header " \
        "--pheno train_cpsp.pheno " \
        "--pheno-name chronic_pain_cc " \
        f"--extract clumped_snps_p{p}.txt " \
        f"--out prs_train_p{p}_clumped" \
    )

# DNAnexus run command
dx run swiss-army-knife -iin="${data_file_dir}/imputed_merged.bed" \
   -iin="${data_file_dir}/imputed_merged.bim" \
   -iin="${data_file_dir}/imputed_merged.fam" \
   -iin="${data_file_dir}/{phenotype_file}" \
   -icmd="${run_plink_qc}" --tag="PRS" --instance-type "mem1_ssd1_v2_x36" \
   --destination="${data_file_dir}/" --brief --yes