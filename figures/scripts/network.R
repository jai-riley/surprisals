library(igraph)
library(ggraph)
library(tidyverse)
library(patchwork)  

# Load your CSV file
pairwise <- read_csv("/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/analysis/mann_whitney_results.csv")

# Create unique node list and assign numbers
nodes <- sort(unique(c(pairwise$Group1, pairwise$Group2)))
node_ids <- tibble(name = nodes, number = seq_along(nodes))
print(nodes)
print(node_ids)
# Prepare edges and merge with node info
edges <- pairwise %>%
  mutate(significant = p_holm < 0.001) %>%
  select(Group1, Group2, significant)

# Build the graph
g <- graph_from_data_frame(edges, vertices = node_ids, directed = FALSE)

# Plot
plot <- ggraph(g, layout = "circle") +
  geom_edge_link(aes(color = significant), width = 1, alpha = 0.9) +
  geom_node_point(size = 6, color = "#00a7f5") +
  geom_node_text(aes(label = number), size = 4, color = "black") +
  scale_edge_color_manual(values = c("TRUE" = "#00a7f5", "FALSE" = "#b0004e"),
                          name = "Significant (p < 0.001)") +
  coord_fixed() +
  theme_void() +
  theme(
    legend.position = c(0.6, -0.05),
    legend.direction = "horizontal",
    plot.margin = margin(2, 10, 10, 10)
  )
custom_labels <- tibble(
  number = 1:length(nodes),
  model_label = c(
    "NGram POS",
    "NGram Word",
    "NGram Word/POS",
    "PCFG Lexical",
    "PCFG POS",
    "PCFG Syntactic",
    "PCFG Total",
    "RNNG Word",
    "Transformer Word"
  )
)

# Combine with node numbers
node_ids <- tibble(name = nodes, number = seq_along(nodes)) %>%
  left_join(custom_labels, by = "number")

# Create legend text using the custom labels
legend_text <- paste0(node_ids$number, " - ", node_ids$model_label, collapse = "\n")

legend_table <- ggplot() +
  geom_text(aes(x = 0, y = 0, label = legend_text), hjust = 0, vjust = 1, lineheight = 0.8, size = 4) +
  xlim(0, 1) +
  ylim(-length(node_ids$name) * 1.2, length(node_ids$name) * 0.6) + 
  theme_void() +
  theme(
    plot.margin = margin(0, 105, 0, -40)
  )

# Combine with widths for spacing
final_plot <- plot + legend_table + plot_layout(widths = c(2.5, 1.4))

# Save the plot
ggsave(filename = "mann-whitney.svg",
       plot = final_plot,
       path = "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/plots/",
       width = 6.5, height = 4, device = "svg")
# ggsave(filename = "mann-whitney.png",
#        plot = final_plot,
#        path = "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/plots/",
#        width = 6.5, height = 4, dpi = 300, device = "png")
# Save the legend mapping numbers to names
write_csv(node_ids, "/Users/jairiley/Desktop/BOW_Ngrams/GAMMs/plots/node_id_legend.csv")
