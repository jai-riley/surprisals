library(mgcv)
# args <- commandArgs(trailingOnly = TRUE)

# # Set default language or get from command line
# if (length(args) > 0) {
#   language <- args[1]  # First argument is the language
# } else {
#   language <- "E"  # Default to "E" if no argument is provided
# }

get_model <- function(language, model_num, type) {
  filename <- paste0("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/models/", type, "/", language, "/", language, "_GAMM_", model_num, ".rds")
  readRDS(filename)
}

extract_part <- function(filename) {
  sub("^[ESCK]_GAMM_(.*)\\.rds$", "\\1", filename)
}

x_labels <- list(
  "base"  = "Base Model",
  "ngram_word"       = "N-Gram Word",
  "ngram_pos"        = "N-Gram POS",
  "ngram_word_pos"    = "N-Gram Word/POS",
  "pcfg_total"    = "PCFG Total",
  "pcfg_syn"      = "PCFG Syntactic",
  "pcfg_lex"  = "PCFG Lexical",
  "pcfg_pos"      = "PCFG POS",
  "rnng_word"          = "RNNG Word",
  "trans_word"   = "Transformer Word"

)
for (language in c("E", "C", "K", "S")) {
  print(getwd())
  files <- list.files(paste0("models/V2/", language), pattern = "\\.rds$", full.names = FALSE)
  # print(paste("Language", language, "files:", paste(files, collapse=", ")))
  for (filename in files) {
      print(filename)
      model_name <- extract_part(filename)

      # Skip base model if it's in the directory
      # if (model_name == base_model_name) next

      # Skip unknown models
      if (!(model_name %in% names(x_labels))) {
        warning(paste("Skipping unknown model:", model_name))
        next
      }

      # base_model <- get_model(language, base_model_name)
      model <- get_model(language, model_name,"V2")
      sink(paste0("model_summaries/",language,'/',model_name,".txt"))        # Redirect output to file
      print(summary(model))       # Print summary to the file
      sink()  


  }
}

# GAMM_1 <- get_model(language, "ngram_pos", "V2")

# # gam.check(GAMM_1)
# sink("output.txt")        # Redirect output to file
# summary(GAMM_1)           # Print summary to the file
# sink()  
# concurvity(GAMM_1, full=TRUE)
# plot(GAMM_1)
