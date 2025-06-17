# install.packages("expm", dependencies=TRUE)
# install.packages("mvtnorm", dependencies=TRUE)
# install.packages("gld", dependencies=TRUE)
# install.packages("lmom", dependencies=TRUE)
# install.packages("DescTools", dependencies=TRUE)

library(dplyr)
library(DescTools)

# load data
data_path <- "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/all_data.csv"

data <- read.csv(data_path)
# head(data)
# nrow(data)


# remove inaccurate response data
data <- subset(data, ACC == 1)
# nrow(data)

# remove L1 stimuli data
data <- subset(data, Stimuli == "L2")
# nrow(data)


# remove data that is outside of +/-2.5 SD for each participant
data <- data %>% group_by(Subject) %>%
  filter(RT <= mean(RT) + (2.5 * sd(RT)),  RT >= mean(RT) - (2.5 * sd(RT)))
# nrow(data)


# remove data that is outside of 50 ms - 2000 ms
data = subset(data, RT >= 50 & RT <= 2000)
# nrow(data)

# remove Subject K07 because distribution is different
data = subset(data, Subject != "K07")
# nrow(data)

# remove Subject K30 because they are a native English speaker
data = subset(data, Subject != "K30")
# nrow(data)

# remove Subject E33 because they have errors in reading times
# data = subset(data, Subject != "E33")

# get the log of the reading times
log_RT <- log(data$RT)
data$log_RT <- log_RT

# get unique Word IDs
data <- data %>% rowwise() %>% mutate(WordID = paste(ItemNum, "Aw", WordNo, sep=""))
# data$WordID

# print(sum(is.na(data$n_gram_BOW_word_surp))) # Count NA values
# print(sum(is.nan(data$n_gram_BOW_word_surp))) # Count NaN values
# print(sum(is.na(data$n_gram_BOW_word_POS_surp))) # Count NA values
# print(sum(is.nan(data$n_gram_BOW_word_POS_surp))) # Count NaN values
# print(sum(is.na(data$n_gram_BOW_POS_surp))) # Count NA values
# print(sum(is.nan(data$n_gram_BOW_POS_surp))) # Count NaN values

# get winsorized surprisals
data$n_gram_word_surp_wins <- Winsorize(data$n_gram_word_surp, probs=c(0.05, 0.95))
data$n_gram_POS_surp_wins <- Winsorize(data$n_gram_POS_surp, probs=c(0.05, 0.95))
data$n_gram_word_POS_surp_wins <- Winsorize(data$n_gram_word_POS_surp, probs=c(0.05, 0.95))
data$n_gram_BOW_word_surp_wins <- Winsorize(data$n_gram_BOW_word_surp, probs=c(0.05, 0.95))
data$n_gram_BOW_POS_surp_wins <- Winsorize(data$n_gram_BOW_POS_surp, probs=c(0.05, 0.95))
data$n_gram_BOW_word_POS_surp_wins <- Winsorize(data$n_gram_BOW_word_POS_surp, probs=c(0.05, 0.95))
data$PCFG_total_surp_wins <- Winsorize(data$PCFG_total_surp, probs=c(0.05, 0.95))
data$PCFG_syn_surp_wins <- Winsorize(data$PCFG_syn_surp, probs=c(0.05, 0.95))
data$PCFG_lex_surp_wins <- Winsorize(data$PCFG_lex_surp, probs=c(0.05, 0.95))
data$PCFG_pos_surp_wins <- Winsorize(data$PCFG_pos_surp, probs=c(0.05, 0.95))
data$RNNG_word_surp_wins <- Winsorize(data$RNNG_word_surp, probs=c(0.05, 0.95))
data$RNNG_pos_surp_wins <- Winsorize(data$RNNG_pos_surp, probs=c(0.05, 0.95))
data$transformer_word_surp_wins <-  Winsorize(data$transformer_word_surp, probs=c(0.05, 0.95))
data$transformer_pos_surp_wins <-  Winsorize(data$transformer_pos_surp, probs=c(0.05, 0.95))
data$transformer_word_pos_surp_wins <-  Winsorize(data$transformer_word_pos_surp, probs=c(0.05, 0.95))
data$TG_word_surp_wins <- Winsorize(data$TG_word_surp, probs=c(0.05, 0.95))

# data$transformer_surp_wins = Winsorize(data$transformer_surp, probs=c(0.05, 0.95))

# head(data)
# data$n_gram_word_surp_winsPrev1 = data$n_gram_word_surp_wins
# data %>% mutate(n_gram_word_surp_winsPrev1=lag(n_gram_word_surp_wins))
# data$n_gram_POS_surp_winsPrev1 = data$n_gram_POS_surp_wins
# data %>% mutate(n_gram_POS_surp_winsPrev1=lag(n_gram_POS_surp_wins))
# data %>% mutate(n_gram_word_POS_surp_winsPrev1=lag(n_gram_word_POS_surp_wins))
# data$PCFG_total_surp_winsPrev1 = data$PCFG_total_surp_wins
# data %>% mutate(PCFG_total_surp_winsPrev1=lag(PCFG_total_surp_wins))
# data$PCFG_syn_surp_winsPrev1 = data$PCFG_syn_surp_wins
# data %>% mutate(PCFG_syn_surp_winsPrev1=lag(PCFG_syn_surp_wins))
# data$PCFG_lex_surp_winsPrev1 = data$PCFG_lex_surp_wins
# data %>% mutate(PCFG_lex_surp_winsPrev1=lag(PCFG_lex_surp_wins))
# data$PCFG_pos_surp_winsPrev1 = data$PCFG_pos_surp_wins
# data %>% mutate(PCFG_pos_surp_winsPrev1=lag(PCFG_pos_surp_wins))
# data$RNNG_surp_winsPrev1 = data$RNNG_surp_wins
# data %>% mutate(RNNG_surp_winsPrev1=lag(RNNG_surp_wins))
# data$transformer_surp_winsPrev1 = data$transformer_surp_wins
# data %>% mutate(transformer_surp_winsPrev1=lag(transformer_surp_wins))

# convert to factors
data$Subject <- factor(data$Subject)
data$Word <- factor(data$Word)
data$procWord <- factor(data$procWord)
data$procWord <- factor(data$procWordID)
data$WordID <- factor(data$WordID)
data$POS <- factor(data$POS)
data$Trial <- factor(data$Trial)
data$HasPunct <- factor(data$HasPunct)
data$SentPos <- factor(data$SentPos)

print("done")
write.csv(data, "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/all_data_cleaned.csv", row.names=FALSE)
