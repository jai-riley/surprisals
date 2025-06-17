library(mgcv)
library(gratia)
library(ggplot2)

get_model <- function(language, model_num, type) {
  filename <- paste("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/models/", type, "/", language, "/", language, "_GAMM_", model_num, ".rds", sep = "")
  readRDS(filename)
}

width <- 6
height <- 6
save_path <- "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/gam_check_plots/"
list_models <- list("base","ngram_word","ngram_pos","ngram_word_pos","trans_word","rnng_word","pcfg_lex","pcfg_syn","pcfg_pos","pcfg_total")
for (language in list("E","S","K","C")){
  for (model_name in list_models){
    model = get_model(language, model_name, "V2")
    appraise(model)
    ggsave(filename =paste0(model_name,".svg"), path = paste0(save_path,'/',language), width=width, height=height, device="svg")
  }
}
