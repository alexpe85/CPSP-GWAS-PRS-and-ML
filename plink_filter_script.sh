dx run swiss-army-knife \
  -iin="/new_PRS/imputed_merged_no_dups.fam" \
  -iin="/new_PRS/imputed_merged_no_dups.bim" \
  -iin="/new_PRS/imputed_merged_no_dups.bed" \
  -iin="/new_PRS/gwas_for_prs_0.05.tsv" \
  -iin="/new_PRS/train_plink_standard.pheno" \
  -iin="/new_PRS/clumped_snps_p0.1.txt" \
  -icmd="plink --bfile imputed_merged_no_dups \
    --score gwas_for_prs_0.05.tsv 2 4 5 header \
    --pheno train_plink_standard.pheno \
    --pheno-name plink_pheno \
    --extract clumped_snps_p0.1.txt \
    --out train_PRS_0.05" \
  --instance-type "mem1_ssd1_v2_x36" \
  --destination="/new_PRS/" \
  --tag="train_0.05_PRS" \
  --brief --yes


