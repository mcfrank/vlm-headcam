#!/usr/bin/env Rscript
# English-speaking children's LEVANTE vocabulary ability -> results/levante_en_{scores,item_d}.csv
#
# The vocab task is ADAPTIVE (median 54 of 144 items attempted here), so a raw proportion
# correct over administered trials is not a full-scale accuracy. We therefore export the
# fitted IRT parameters and impute expected performance on all items (see
# make_levante_ages.py). Pulls the levante-data-pilots release via rlevante/Redivis, and
# uses the ENGLISH (en-US) language-specific Rasch calibration -- the models being compared
# are trained on English, and site/language differences here are large.
suppressMessages({library(rlevante); library(dplyr)})
R <- file.path(dirname(dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value=TRUE))))), "results")

st <- fetch_scoring_table()
spec <- st |> filter(task_id == "vocab", subset == "en-US") |> slice(1) |> as.list()
m <- get_registry_file(model_spec_filename(spec), fetch_registry_dir())
stopifnot(slot(m, "itemtype") == "Rasch")
mv <- slot(m, "model_vals")
g <- as.numeric(mv |> filter(name == "g") |> pull(value))          # guessing fixed at chance
stopifnot(length(unique(round(g, 6))) == 1, isTRUE(all.equal(unique(round(g, 6)), 0.25)))
mv |> filter(name == "d") |> transmute(item = sub("-1$", "", item), d = value) |>
  write.csv(file.path(R, "levante_en_item_d.csv"), row.names = FALSE)

s <- get_scores("levante-data-pilots") |>
  filter(task_id == "vocab", language == "en-US", score_type == "ability", is.na(exclusion)) |>
  select(user_id, run_id, dataset, language, score, score_se, age, num_attempted, adaptive)
write.csv(s, file.path(R, "levante_en_scores.csv"), row.names = FALSE)
cat(sprintf("en-US: %d runs, %d children, ages %.1f-%.1f; median %d of %d items attempted\n",
            nrow(s), n_distinct(s$user_id), min(s$age), max(s$age),
            median(s$num_attempted, na.rm = TRUE), nrow(mv |> filter(name == "d"))))
