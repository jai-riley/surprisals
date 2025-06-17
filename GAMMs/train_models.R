library(mgcv)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set default language or get from command line
if (length(args) > 0) {
  language <- args[1]  # First argument is the language
} else {
  language <- "E"  # Default to "E" if no argument is provided
}

# Load data
data_path <- "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/all_data_cleaned.csv"
data <- read.csv(data_path)

data$Subject <- factor(data$Subject)
data$Word <- factor(data$Word)
data$procWord <- factor(data$procWord)
data$procWordID <- factor(data$procWordID)
data$WordID <- factor(data$WordID)
data$POS <- factor(data$POS)
data$Trial <- factor(data$Trial)
data$HasPunct <- factor(data$HasPunct)
data$SentPos <- factor(data$SentPos)

# Function to load a model
get_model <- function(language, model_num, type) {
  filename <- paste0("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/models/", type, "/", language, "/", language, "_GAMM_", model_num, ".rds")
  readRDS(filename)
}

# Set save path dynamically based on language
# save_path <- paste0("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/models/", language, "/")

# Load base model
start_time <- Sys.time()

# # Define save path
save_path <- paste0("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/models/V2/", language, "/")

# GAMM_61 <- get_model(language, "base", "shannon")
# print(summary(GAMM_61))
# Fit the GAMM model
GAMM_61 <- bam(log(RT) ~ s(Subject, bs = "fs") + 
                s(Trial, bs = "fs") +
                 s(procWordID, bs="fs") +  
                 s(Vocab_Competence.Acc) + 
                 s(Comp_Competence.Acc) + 
                 SentPos +
                 s(WordPos) + 
                 s(LogFreq) +
                 s(WordLength) +
                 ti(LogFreq, WordLength), 
               method = "ML", 
               data = subset(data, Reader == language))

# Capture end time
end_time <- Sys.time()
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds
print(paste("Base model", language, ":", minutes, "minutes and",seconds,"seconds"))

# Save the model
saveRDS(GAMM_61, file = paste(save_path, language, "_GAMM_base.rds", sep = ""))
# # 
# # Print the AIC value of the model
# print(AIC(GAMM_61))


# GAMM_61 <- get_model(language, "base", "shannon")
# print(summary(GAMM_61))

# GAMM_61 <- get_model(language, "62", "original")
# print(summary(GAMM_61))

start_time <- Sys.time()
# Measure execution time
GAMM_62 <- update(GAMM_61, . ~ . + s(n_gram_word_surp) +
                    ti(n_gram_word_surp, WordLength) + 
                    ti(n_gram_word_surp, LogFreq) + 
                    ti(n_gram_word_surp, WordPos)+
                    ti(n_gram_word_surp, LogFreqPrev1) +
                    ti(n_gram_word_surp, n_gram_word_surpPrev1) +
                    n_gram_word_surp:SentPos)
end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

print(paste("ran ngram word for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_62))
# print(AIC(GAMM_61)-AIC(GAMM_62))
saveRDS(GAMM_62, file =paste(save_path, language, "_GAMM_ngram_word.rds", sep = ""))

start_time <- Sys.time()
GAMM_63 <- update(GAMM_61, . ~ . + s(n_gram_POS_surp) +
                    ti(n_gram_POS_surp, WordLength) + 
                    ti(n_gram_POS_surp, LogFreq) + 
                    ti(n_gram_POS_surp, WordPos) +
                    ti(n_gram_POS_surp, LogFreqPrev1) +
                    ti(n_gram_POS_surp, n_gram_POS_surpPrev1) +
                    n_gram_POS_surp:SentPos)
end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

print(paste("ran ngram pos for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_63))
saveRDS(GAMM_63, file =paste(save_path, language, "_GAMM_ngram_pos.rds", sep = ""))


start_time <- Sys.time()
GAMM_64 <- update(GAMM_61, . ~ . + s(n_gram_word_POS_surp) +
                    ti(n_gram_word_POS_surp, WordLength) + 
                    ti(n_gram_word_POS_surp, LogFreq) + 
                    ti(n_gram_word_POS_surp, WordPos) +
                    ti(n_gram_word_POS_surp, LogFreqPrev1) +
                    ti(n_gram_word_POS_surp, n_gram_word_POS_surpPrev1) +
                    n_gram_word_POS_surp:SentPos)
end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

print(paste("ran ngram word pos for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_64))
saveRDS(GAMM_64, file =paste(save_path, language, "_GAMM_ngram_word_pos.rds", sep = ""))


print("Start model 65")
start_time <- Sys.time()
GAMM_65 <- update(GAMM_61, . ~ . + s(PCFG_total_surp)+
                    ti(PCFG_total_surp, WordLength) + 
                    ti(PCFG_total_surp, LogFreq) + 
                    ti(PCFG_total_surp, WordPos) +
                    ti(PCFG_total_surp, LogFreqPrev1) +
                    ti(PCFG_total_surp, PCFG_total_surpPrev1) +
                    PCFG_total_surp:SentPos)
end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran PCFG total for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_65))
saveRDS(GAMM_65, file = paste0(save_path, language, "_GAMM_pcfg_total.rds"))


print("Start model 66")
start_time <- Sys.time()
GAMM_66 <- update(GAMM_61, . ~ . + s(PCFG_syn_surp)+
                    ti(PCFG_syn_surp, WordLength) + 
                    ti(PCFG_syn_surp, LogFreq) + 
                    ti(PCFG_syn_surp, WordPos) +
                    ti(PCFG_syn_surp, LogFreqPrev1) +
                    ti(PCFG_syn_surp, PCFG_syn_surpPrev1) +
                    PCFG_syn_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran PCFG syn for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_66))
saveRDS(GAMM_66, file =paste(save_path, language, "_GAMM_pcfg_syn.rds", sep = ""))


print("Start model 67")
start_time <- Sys.time()
GAMM_67 <- update(GAMM_61, . ~ . + s(PCFG_lex_surp)+
                    ti(PCFG_lex_surp, WordLength) + 
                    ti(PCFG_lex_surp, LogFreq) + 
                    ti(PCFG_lex_surp, WordPos) +
                    ti(PCFG_lex_surp, LogFreqPrev1) +
                    ti(PCFG_lex_surp, PCFG_lex_surpPrev1) +
                    PCFG_lex_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran PCFG lex for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_67))
saveRDS(GAMM_67, file =paste(save_path, language, "_GAMM_pcfg_lex.rds", sep = ""))


print("Start model 68")
start_time <- Sys.time()
GAMM_68 <- update(GAMM_61, . ~ . + s(PCFG_pos_surp)+
                    ti(PCFG_pos_surp, WordLength) + 
                    ti(PCFG_pos_surp, LogFreq) + 
                    ti(PCFG_pos_surp, WordPos) +
                    ti(PCFG_pos_surp, LogFreqPrev1) +
                    ti(PCFG_pos_surp, PCFG_pos_surpPrev1) +
                    PCFG_pos_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran PCFG pos for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_68))
saveRDS(GAMM_68, file =paste(save_path, language, "_GAMM_pcfg_pos.rds", sep = ""))

start_time <- Sys.time()
print("Start model 69")
GAMM_69 <- update(GAMM_61, . ~ . + s(n_gram_BOW_word_surp) +
                    ti(n_gram_BOW_word_surp, WordLength) + 
                    ti(n_gram_BOW_word_surp, LogFreq) + 
                    ti(n_gram_BOW_word_surp, WordPos) +
                    ti(n_gram_BOW_word_surp, LogFreqPrev1) +
                    ti(n_gram_BOW_word_surp, n_gram_BOW_word_surpPrev1) +
                    n_gram_BOW_word_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran BOW word for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_69))
saveRDS(GAMM_69, file =paste(save_path, language, "_GAMM_BOW_word.rds", sep = ""))

start_time <- Sys.time()
print("Start model 70")
GAMM_70 <- update(GAMM_61, . ~ . + s(n_gram_BOW_POS_surp)+
                    ti(n_gram_BOW_POS_surp, WordLength) + 
                    ti(n_gram_BOW_POS_surp, LogFreq) + 
                    ti(n_gram_BOW_POS_surp, WordPos) +
                    ti(n_gram_BOW_POS_surp, LogFreqPrev1) +
                    ti(n_gram_BOW_POS_surp, n_gram_BOW_POS_surpPrev1) +
                    n_gram_BOW_POS_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran BOW pos for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_70))
saveRDS(GAMM_70, file =paste(save_path, language, "_GAMM_BOW_pos.rds", sep = ""))

start_time <- Sys.time()
print("Start model 71")
GAMM_71 <- update(GAMM_61, . ~ . + s(n_gram_BOW_word_POS_surp)+
                    ti(n_gram_BOW_word_POS_surp, WordLength) + 
                    ti(n_gram_BOW_word_POS_surp, LogFreq) + 
                    ti(n_gram_BOW_word_POS_surp, WordPos) +
                    ti(n_gram_BOW_word_POS_surp, LogFreqPrev1) +
                    ti(n_gram_BOW_word_POS_surp, n_gram_BOW_word_POS_surpPrev1) +
                    n_gram_BOW_word_POS_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran BOW word pos for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_71))
saveRDS(GAMM_71, file =paste(save_path, language, "_GAMM_BOW_word_pos.rds", sep = ""))
# # 
start_time <- Sys.time()
print("Start model 72")
GAMM_72 <- update(GAMM_61, . ~ . + s(RNNG_word_surp)+
                    ti(RNNG_word_surp, WordLength) + 
                    ti(RNNG_word_surp, LogFreq) + 
                    ti(RNNG_word_surp, WordPos) +
                    ti(RNNG_word_surp, LogFreqPrev1) +
                    ti(RNNG_word_surp, RNNG_word_surpPrev1) +
                    RNNG_word_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran RNNG word for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_72))
saveRDS(GAMM_72, file =paste(save_path, language, "_GAMM_rnng_word.rds", sep = ""))

start_time <- Sys.time()
print("Start model 73")
GAMM_73 <- update(GAMM_61, . ~ . + s(RNNG_pos_surp)+
                    ti(RNNG_pos_surp, WordLength) + 
                    ti(RNNG_pos_surp, LogFreq) + 
                    ti(RNNG_pos_surp, WordPos) +
                    ti(RNNG_pos_surp, LogFreqPrev1) +
                    ti(RNNG_pos_surp, RNNG_pos_surpPrev1) +
                    RNNG_pos_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran RNNG pos for:", sprintf("%02d:%02d", minutes, seconds)))
# print(AIC(GAMM_61)-AIC(GAMM_73))
print(AIC(GAMM_73))
saveRDS(GAMM_73, file =paste(save_path, language, "_GAMM_rnng_pos.rds", sep = ""))

start_time <- Sys.time()
print("Start model 74")
GAMM_74 <- update(GAMM_61, . ~ . + s(transformer_word_surp)+
                    ti(transformer_word_surp, WordLength) + 
                    ti(transformer_word_surp, LogFreq) + 
                    ti(transformer_word_surp, WordPos) +
                    ti(transformer_word_surp, LogFreqPrev1) +
                    ti(transformer_word_surp, transformer_word_surpPrev1) +
                    transformer_word_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran transformer_word_surp for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_74))
saveRDS(GAMM_74, file =paste(save_path, language, "_GAMM_trans_word.rds", sep = ""))

start_time <- Sys.time()
print("Start model 75")
GAMM_75 <- update(GAMM_61, . ~ . + s(transformer_pos_surp)+
                    ti(transformer_pos_surp, WordLength) + 
                    ti(transformer_pos_surp, LogFreq) + 
                    ti(transformer_pos_surp, WordPos) +
                    ti(transformer_pos_surp, LogFreqPrev1) +
                    ti(transformer_pos_surp, transformer_pos_surpPrev1) +
                    transformer_pos_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran transformer_pos_surp for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_75))
saveRDS(GAMM_75, file =paste(save_path, language, "_GAMM_trans_pos.rds", sep = ""))


start_time <- Sys.time()
print("Start model 76")
GAMM_76 <- update(GAMM_61, . ~ . + s(transformer_word_pos_surp)+
                    ti(transformer_word_pos_surp, WordLength) + 
                    ti(transformer_word_pos_surp, LogFreq) + 
                    ti(transformer_word_pos_surp, WordPos) +
                    ti(transformer_word_pos_surp, LogFreqPrev1) +
                    ti(transformer_word_pos_surp, transformer_word_pos_surpPrev1) +
                    transformer_word_pos_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran transformer_word_pos_surp for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_76))
saveRDS(GAMM_76, file =paste0(save_path, language, "_GAMM_trans_word_pos.rds"))

start_time <- Sys.time()
print("Start model 77")
GAMM_77 <- update(GAMM_61, . ~ . + s(TG_word_surp)+
                    ti(TG_word_surp, WordLength) + 
                    ti(TG_word_surp, LogFreq) + 
                    ti(TG_word_surp, WordPos) +
                    ti(TG_word_surp, LogFreqPrev1) +
                    ti(TG_word_surp, TG_word_surpPrev1) +
                    TG_word_surp:SentPos)

end_time <- Sys.time()  # Capture end time

# Get execution time in total seconds
execution_time_secs <- as.numeric(difftime(end_time, start_time, units = "secs"))

# Convert to minutes and seconds
minutes <- floor(execution_time_secs / 60)  # Get whole minutes
seconds <- round(execution_time_secs %% 60)  # Get remaining seconds

# Print time in mm:ss format
print(paste("ran TG_word_surp for:", sprintf("%02d:%02d", minutes, seconds)))
print(AIC(GAMM_77))
saveRDS(GAMM_77, file =paste0(save_path, language, "_GAMM_TG_word.rds"))
