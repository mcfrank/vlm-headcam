#!/usr/bin/env Rscript
# Scale-free age-of-acquisition via Rasch item difficulties, fit with mirt on all
# English (American) Wordbank data:
#   production    — WG + WS pooled (common items link the forms)
#   comprehension — WG only (WS does not measure it)
# Writes results/wordbank_rasch.csv: one row per (normalized) word with b_produce /
# b_comprehend and admin counts. Higher b = harder = acquired later.
suppressMessages({library(wordbankr); library(dplyr); library(tidyr); library(mirt)})

norm <- function(x) trimws(tolower(sub(" \\(.*\\)$", "", x)))

get_form <- function(form) {
  items <- get_item_data(language = "English (American)", form = form) |>
    filter(item_kind == "word") |> mutate(word = norm(item_definition))
  dat <- get_instrument_data(language = "English (American)", form = form,
                             items = items$item_id, administration_info = TRUE)
  dat |> left_join(items |> select(item_id, word), by = "item_id") |>
    select(data_id, word, value)
}

cat("fetching WG...\n"); wg <- get_form("WG")
cat("fetching WS...\n"); ws <- get_form("WS")

fit_rasch <- function(long, label) {
  wide <- long |>
    distinct(data_id, word, .keep_all = TRUE) |>
    pivot_wider(id_cols = data_id, names_from = word, values_from = resp) |>
    select(-data_id)
  keep <- colMeans(!is.na(wide)) > 0
  wide <- wide[, keep]
  p <- colMeans(wide, na.rm = TRUE)
  ok <- p > 0.01 & p < 0.99                    # drop (near-)degenerate items
  cat(sprintf("%s: %d admins x %d items (%d dropped as degenerate)\n",
              label, nrow(wide), sum(ok), sum(!ok)))
  mod <- mirt(data.frame(wide[, ok], check.names = FALSE), 1, itemtype = "Rasch",
              verbose = FALSE, technical = list(NCYCLES = 2000))
  b <- coef(mod, IRTpars = TRUE, simplify = TRUE)$items[, "b"]
  tibble(word = colnames(wide)[ok], b = as.numeric(b),
         n_admin = colSums(!is.na(wide[, ok])), p_know = p[ok])
}

prod <- bind_rows(wg, ws) |> mutate(resp = as.integer(value == "produces"))
comp <- wg |> mutate(resp = as.integer(value %in% c("understands", "produces")))

bp <- fit_rasch(prod, "production (WG+WS)") |> rename(b_produce = b, n_prod = n_admin, p_produce = p_know)
bc <- fit_rasch(comp, "comprehension (WG)") |> rename(b_comprehend = b, n_comp = n_admin, p_comprehend = p_know)

out <- full_join(bp, bc, by = "word") |> arrange(b_produce)
write.csv(out, "results/wordbank_rasch.csv", row.names = FALSE)
cat("wrote results/wordbank_rasch.csv:", nrow(out), "words\n")
print(head(out, 5)); print(tail(out, 5))
