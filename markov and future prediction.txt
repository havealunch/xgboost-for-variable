# Install packages
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("tidyr")
install.packages("gridExtra")
install.packages("reshape2")
install.packages("markovchain")
install.packages("diagram")
install.packages("igraph")
install.packages("expm")

# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(reshape2)
library(markovchain)
library(diagram)
library(igraph)
library(expm)

#===============================================================================
# 1. LOAD DATA
#===============================================================================

climate_data <- read.csv(file.choose())

print(colnames(climate_data))
str(climate_data)

#===============================================================================
# 2. CALCULATE STATISTICS BY CONTINENT AND PERIOD
#===============================================================================

continent_stats <- climate_data %>%
  group_by(continent, period) %>%
  summarise(
    mean_coeff = mean(inequality_coefficient, na.rm = TRUE),
    sd_coeff = sd(inequality_coefficient, na.rm = TRUE),
    min_coeff = min(inequality_coefficient, na.rm = TRUE),
    max_coeff = max(inequality_coefficient, na.rm = TRUE),
    n_countries = n(),
    .groups = 'drop'
  )

print(continent_stats)

global_stats <- climate_data %>%
  group_by(period) %>%
  summarise(
    global_mean = mean(inequality_coefficient, na.rm = TRUE),
    global_sd = sd(inequality_coefficient, na.rm = TRUE),
    .groups = 'drop'
  )

print(global_stats)

#===============================================================================
# 3. CLASSIFY COUNTRIES INTO RISK LEVELS
#===============================================================================

classify_risk <- function(data, period_value) {
  period_data <- data %>% filter(period == period_value)
  
  q25 <- quantile(period_data$inequality_coefficient, 0.25, na.rm = TRUE)
  q75 <- quantile(period_data$inequality_coefficient, 0.75, na.rm = TRUE)
  
  period_data %>%
    mutate(
      risk_level = case_when(
        inequality_coefficient < q25 ~ "L",
        inequality_coefficient >= q25 & inequality_coefficient <= q75 ~ "M",
        inequality_coefficient > q75 ~ "H"
      )
    )
}

risk_2040 <- classify_risk(climate_data, 2040)
risk_2060 <- classify_risk(climate_data, 2060)
risk_2080 <- classify_risk(climate_data, 2080)

risk_all <- risk_2040 %>%
  select(country, risk_level_2040 = risk_level) %>%
  left_join(
    risk_2060 %>% select(country, risk_level_2060 = risk_level),
    by = "country"
  ) %>%
  left_join(
    risk_2080 %>% select(country, risk_level_2080 = risk_level),
    by = "country"
  )

risk_all <- risk_all %>%
  mutate(
    type = paste0(risk_level_2040, risk_level_2060, risk_level_2080)
  )

type_counts <- risk_all %>%
  group_by(type) %>%
  summarise(
    count = n(),
    percentage = count / nrow(risk_all) * 100,
    .groups = 'drop'
  ) %>%
  arrange(desc(count))

print("Top 10 country types:")
print(head(type_counts, 10))

#===============================================================================
# 4. ADD COEFFICIENT VALUES AND CONTINENT INFO
#===============================================================================

risk_all <- risk_all %>%
  left_join(
    climate_data %>% filter(period == 2040) %>% select(country, coeff_2040 = inequality_coefficient),
    by = "country"
  ) %>%
  left_join(
    climate_data %>% filter(period == 2060) %>% select(country, coeff_2060 = inequality_coefficient),
    by = "country"
  ) %>%
  left_join(
    climate_data %>% filter(period == 2080) %>% select(country, coeff_2080 = inequality_coefficient),
    by = "country"
  ) %>%
  left_join(
    climate_data %>% select(country, continent) %>% distinct(),
    by = "country"
  )

risk_all <- risk_all %>%
  mutate(
    trend_type = case_when(
      coeff_2040 < coeff_2060 & coeff_2060 < coeff_2080 ~ "Increasing",
      coeff_2040 > coeff_2060 & coeff_2060 > coeff_2080 ~ "Decreasing",
      coeff_2040 < coeff_2060 & coeff_2060 > coeff_2080 ~ "Inverted_U",
      coeff_2040 > coeff_2060 & coeff_2060 < coeff_2080 ~ "U_Shape",
      TRUE ~ "Other"
    )
  )

#===============================================================================
# 5. TOP 9 TYPES AND REPRESENTATIVE COUNTRIES
#===============================================================================

top9_types <- type_counts$type[1:9]

type_trend_data <- risk_all %>%
  filter(type %in% top9_types) %>%
  select(country, continent, type, coeff_2040, coeff_2060, coeff_2080) %>%
  pivot_longer(
    cols = starts_with("coeff_"),
    names_to = "period",
    values_to = "coefficient"
  ) %>%
  mutate(
    period = case_when(
      period == "coeff_2040" ~ "2040",
      period == "coeff_2060" ~ "2060",
      period == "coeff_2080" ~ "2080"
    ),
    period = factor(period, levels = c("2040", "2060", "2080"))
  )

rep_countries <- risk_all %>%
  filter(type %in% top9_types) %>%
  group_by(type) %>%
  slice(1) %>%
  select(country, continent, type)

rep_trend_data <- type_trend_data %>%
  inner_join(rep_countries, by = c("country", "type", "continent"))

#===============================================================================
# 6. MARKOV CHAIN TRANSITION MATRICES
#===============================================================================

create_transition_matrix <- function(from_levels, to_levels) {
  transitions <- table(from_levels, to_levels)
  transition_matrix <- prop.table(transitions, margin = 1)
  return(transition_matrix)
}

markov_data <- risk_all %>%
  select(country, risk_level_2040, risk_level_2060, risk_level_2080)

trans_2040_2060 <- create_transition_matrix(
  markov_data$risk_level_2040,
  markov_data$risk_level_2060
)

trans_2060_2080 <- create_transition_matrix(
  markov_data$risk_level_2060,
  markov_data$risk_level_2080
)

trans_2040_2080 <- create_transition_matrix(
  markov_data$risk_level_2040,
  markov_data$risk_level_2080
)

trans_two_step <- trans_2040_2060 %*% trans_2060_2080
rownames(trans_two_step) <- c("L", "M", "H")
colnames(trans_two_step) <- c("L", "M", "H")

print("Transition Matrix 2040 -> 2060:")
print(trans_2040_2060)

print("Transition Matrix 2060 -> 2080:")
print(trans_2060_2080)

print("Transition Matrix 2040 -> 2080:")
print(trans_2040_2080)

print("Two-step Transition Matrix 2040 -> 2080:")
print(trans_two_step)

#===============================================================================
# 7. VISUALIZATION
#===============================================================================

plot_transition_matrix <- function(matrix_data, title) {
  matrix_df <- as.data.frame(matrix_data)
  matrix_df$From <- rownames(matrix_data)
  
  matrix_long <- matrix_df %>%
    pivot_longer(
      cols = -From,
      names_to = "To",
      values_to = "Probability"
    )
  
  ggplot(matrix_long, aes(x = To, y = From, fill = Probability)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = sprintf("%.2f", Probability)), size = 5) +
    theme_minimal() +
    labs(title = title, x = "To State", y = "From State")
}

# Figure 5j
fig_5j <- ggplot(climate_data, aes(x = factor(period), y = inequality_coefficient, fill = continent)) +
  geom_boxplot() +
  theme_minimal() +
  labs(
    title = "Climate Inequality Coefficient by Continent",
    x = "Period",
    y = "Inequality Coefficient"
  ) +
  theme(legend.position = "bottom")

print(fig_5j)

# Figure 6a
fig_6a <- ggplot(rep_trend_data, aes(x = period, y = coefficient, group = country, color = continent)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  facet_wrap(~ type, scales = "free_y", nrow = 3) +
  theme_minimal() +
  labs(
    title = "Climate Inequality Trends by Country Type",
    x = "Period",
    y = "Inequality Coefficient"
  ) +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    legend.position = "bottom"
  )

print(fig_6a)

# Figures 6b-6e
fig_6b <- plot_transition_matrix(trans_2040_2060, "Transition Matrix: 2040 -> 2060")
print(fig_6b)

fig_6c <- plot_transition_matrix(trans_2060_2080, "Transition Matrix: 2060 -> 2080")
print(fig_6c)

fig_6d <- plot_transition_matrix(trans_2040_2080, "Transition Matrix: 2040 -> 2080")
print(fig_6d)

fig_6e <- plot_transition_matrix(trans_two_step, "Two-step Transition: 2040 -> 2080")
print(fig_6e)

#===============================================================================
# 8. AGENT-BASED MODEL SIMULATION
#===============================================================================

gini <- function(x) {
  n <- length(x)
  x_sorted <- sort(x)
  index <- seq_along(x_sorted)
  gini_index <- (2 * sum(index * x_sorted)) / (n * sum(x_sorted)) - (n + 1) / n
  return(gini_index)
}

abm_simulation <- function(n_agents = 1000, n_years = 30, n_simulations = 100) {
  results <- data.frame()
  
  for(sim in 1:n_simulations) {
    agents <- data.frame(
      id = 1:n_agents,
      wealth = rlnorm(n_agents, meanlog = 0, sdlog = 1),
      vulnerable = sample(c(TRUE, FALSE), n_agents, prob = c(0.2, 0.8), replace = TRUE)
    )
    
    inequality_trajectory <- data.frame(
      year = 1:n_years,
      no_policy = numeric(n_years),
      infrastructure = numeric(n_years),
      targeted = numeric(n_years)
    )
    
    for(year in 1:n_years) {
      shock <- rnorm(n_agents, mean = 0.1, sd = 0.05)
      
      wealth_no_policy <- agents$wealth * (1 - shock * 0.5)
      inequality_trajectory$no_policy[year] <- gini(wealth_no_policy)
      
      wealth_infra <- agents$wealth * (1 - shock * 0.4)
      inequality_trajectory$infrastructure[year] <- gini(wealth_infra)
      
      wealth_targeted <- agents$wealth * (1 - shock * 0.5)
      poorest_20 <- which(agents$wealth < quantile(agents$wealth, 0.2))
      wealth_targeted[poorest_20] <- wealth_targeted[poorest_20] * (1 + 0.1)
      inequality_trajectory$targeted[year] <- gini(wealth_targeted)
      
      agents$wealth <- wealth_no_policy
    }
    
    inequality_trajectory$simulation <- sim
    results <- rbind(results, inequality_trajectory)
  }
  
  return(results)
}

abm_results <- abm_simulation(n_agents = 1000, n_years = 30, n_simulations = 100)

abm_summary <- abm_results %>%
  group_by(year) %>%
  summarise(
    no_policy_mean = mean(no_policy, na.rm = TRUE),
    no_policy_sd = sd(no_policy, na.rm = TRUE),
    infrastructure_mean = mean(infrastructure, na.rm = TRUE),
    infrastructure_sd = sd(infrastructure, na.rm = TRUE),
    targeted_mean = mean(targeted, na.rm = TRUE),
    targeted_sd = sd(targeted, na.rm = TRUE),
    .groups = 'drop'
  )

# Figures 6f-6i
fig_6f <- ggplot(abm_summary, aes(x = year)) +
  geom_line(aes(y = no_policy_mean, color = "No Policy"), size = 1.2) +
  geom_ribbon(aes(ymin = no_policy_mean - no_policy_sd, 
                  ymax = no_policy_mean + no_policy_sd), 
              alpha = 0.2) +
  geom_line(aes(y = infrastructure_mean, color = "Infrastructure"), size = 1.2) +
  geom_ribbon(aes(ymin = infrastructure_mean - infrastructure_sd, 
                  ymax = infrastructure_mean + infrastructure_sd), 
              alpha = 0.2) +
  geom_line(aes(y = targeted_mean, color = "Targeted"), size = 1.2) +
  geom_ribbon(aes(ymin = targeted_mean - targeted_sd, 
                  ymax = targeted_mean + targeted_sd), 
              alpha = 0.2) +
  theme_minimal() +
  labs(
    title = "Policy Impact on Inequality",
    x = "Year",
    y = "Inequality (Gini)",
    color = "Policy Scenario"
  ) +
  theme(legend.position = "bottom")

print(fig_6f)

#===============================================================================
# 9. EXPORT RESULTS
#===============================================================================

write.csv(risk_all, "country_risk_classification.csv", row.names = FALSE)
write.csv(type_counts, "country_type_counts.csv", row.names = FALSE)
write.csv(continent_stats, "continent_statistics.csv", row.names = FALSE)
write.csv(global_stats, "global_statistics.csv", row.names = FALSE)
write.csv(abm_summary, "abm_simulation_results.csv", row.names = FALSE)

saveRDS(list(
  trans_2040_2060 = trans_2040_2060,
  trans_2060_2080 = trans_2060_2080,
  trans_2040_2080 = trans_2040_2080,
  trans_two_step = trans_two_step
), "transition_matrices.rds")

ggsave("fig_5j_continent_trends.png", fig_5j, width = 10, height = 6)
ggsave("fig_6a_type_trends.png", fig_6a, width = 12, height = 8)
ggsave("fig_6b_transition_2040_2060.png", fig_6b, width = 8, height = 6)
ggsave("fig_6c_transition_2060_2080.png", fig_6c, width = 8, height = 6)
ggsave("fig_6d_transition_2040_2080.png", fig_6d, width = 8, height = 6)
ggsave("fig_6e_transition_two_step.png", fig_6e, width = 8, height = 6)
ggsave("fig_6f_abm_trajectory.png", fig_6f, width = 10, height = 6)

#===============================================================================
# 10. SUMMARY
#===============================================================================

cat("\n========================================\n")
cat("CLIMATE INEQUALITY ANALYSIS SUMMARY\n")
cat("========================================\n")

cat("\nGLOBAL STATISTICS:\n")
print(global_stats)

cat("\nTOP 10 COUNTRY TYPES:\n")
print(head(type_counts, 10))

cat("\nTRANSITION PROBABILITIES 2040 -> 2060:\n")
print(trans_2040_2060)

cat("\nTRANSITION PROBABILITIES 2060 -> 2080:\n")
print(trans_2060_2080)

cat("\nFILES GENERATED:\n")
cat("1. country_risk_classification.csv\n")
cat("2. country_type_counts.csv\n")
cat("3. continent_statistics.csv\n")
cat("4. global_statistics.csv\n")
cat("5. abm_simulation_results.csv\n")
cat("6. transition_matrices.rds\n")
cat("7. fig_5j_continent_trends.png\n")
cat("8. fig_6a_type_trends.png\n")
cat("9. fig_6b_transition_2040_2060.png\n")
cat("10. fig_6c_transition_2060_2080.png\n")
cat("11. fig_6d_transition_2040_2080.png\n")
cat("12. fig_6e_transition_two_step.png\n")
cat("13. fig_6f_abm_trajectory.png\n")