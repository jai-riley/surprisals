library(mgcv)

get_model <- function(language, model_name) {
  filename <- paste0("models/shannon/", language, "/", language, "_GAMM_", model_name, ".rds")
  readRDS(filename)
}

# Extracts model name part like "TG_word" or "PCFG_lexical"
extract_part <- function(filename) {
  sub("^[ESCK]_GAMM_(.*)\\.rds$", "\\1", filename)
}

# Labels and colours based on model type names
x_labels <- list(
  "ngram_word"       = "N-Gram Word",
  "ngram_pos"        = "N-Gram POS",
  "ngram_word_pos"    = "N-Gram Word/POS",
  "pcfg_total"    = "PCFG Total",
  "pcfg_syn"      = "PCFG Syntactic",
  "pcfg_lex"  = "PCFG Lexical",
  "pcfg_pos"      = "PCFG POS",
  "rnng_word"          = "RNNG Word",
  "trans_word"   = "Transformer Word"
  # "BOW_word"       = "BOW N-Gram Word",  # dark red
  # "BOW_pos"        = "BOW N-Gram POS",  # steel blue
  # "BOW_word_pos"   = "BOW N-Gram Word/POS",  # olive green
  # "rnng_pos"       = "RNNG POS",  # burnt orange
  # "trans_pos"        = "Transformer POS",  # lavender blue
  # "trans_word_pos"   = "Transformer Word/POS",  # green-teal
  # "TG_word"  = "Transformer Grammar"   # bright yellow
)

colours <- list(
  "ngram_word"     = "#348ABD",  # blue
  "ngram_pos"      = "#188487",  # teal
  "ngram_word_pos" = "#FDBF11",  # mustard yellow
  "pcfg_total"     = "#E24A33",  # orange-red
  "pcfg_syn"       = "#FC7E5E",  # salmon
  "pcfg_lex"       = "#7E2F8E",  # purple
  "pcfg_pos"       = "#AF7F3D",  # brown
  "rnng_word"      = "#33BBCC",  # cyan
  "trans_word"     = "#E586B6",  # pink
  "BOW_word"       = "#A60628",  # dark red
  "BOW_pos"        = "#5D8AA8",  # steel blue
  "BOW_word_pos"   = "#66A61E",  # olive green
  "rnng_pos"       = "#D95F02",  # burnt orange
  "trans_pos"        = "#7570B3",  # lavender blue
  "trans_word_pos"   = "#1B9E77",  # green-teal
  "TG_word"  = "#FFD92F"   # bright yellow
)

l <- list()
base_model_name <- "base"

df <- data.frame()
df2 <- data.frame()

for (language in c("E", "C", "K", "S")) {
  print(getwd())
  files <- list.files(paste0("models/shannon/", language), pattern = "\\.rds$", full.names = FALSE)
  # print(paste("Language", language, "files:", paste(files, collapse=", ")))
  for (filename in files) {
    print(filename)
    model_name <- extract_part(filename)

    # Skip base model if it's in the directory
    if (model_name == base_model_name) next

    # Skip unknown models
    if (!(model_name %in% names(x_labels))) {
      warning(paste("Skipping unknown model:", model_name))
      next
    }

    base_model <- get_model(language, base_model_name)
    full_model <- get_model(language, model_name)

    delta_AIC <- AIC(base_model) - AIC(full_model)
    delta_logLik <- logLik(full_model) - logLik(base_model)

    df <- rbind(df, data.frame(language = language,
                               surprisal = x_labels[[model_name]],
                               delta_AIC = delta_AIC,
                               colour = colours[[model_name]]))

    df2 <- rbind(df2, data.frame(language = language,
                                 surprisal = x_labels[[model_name]],
                                 delta_logLik = delta_logLik,
                                 colour = colours[[model_name]]))
  }
}

write.csv(df, paste0("AIC_base_model_", base_model_name, "_partial_OG.csv"), row.names = FALSE, quote = FALSE)
write.csv(df2, paste0("logLik_base_model_", base_model_name, ".csv"), row.names = FALSE, quote = FALSE)
