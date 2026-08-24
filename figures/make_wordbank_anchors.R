#!/usr/bin/env Rscript
# Wordbank-based child anchors for fig2B: for each Konkle test-60 word with a CDI (American
# English) item, the proportion of children who know it at each age; predicted 4AFC assumes a
# child picks the target if they know the word, else guesses (p + (1-p)/4).
# Comprehension ("understands") from WG (8-18 mo); production from WS (16-30 mo) — production
# is a lower bound on comprehension. Writes results/wordbank_anchors.csv (+ per-item file).
suppressMessages({library(wordbankr); library(dplyr); library(tidyr); library(purrr)})

konkle <- c("airplane","apple","bagel","ball","balloon","basket","bed","bell","bike","bill",
            "bird","boot","bottle","bowl","bucket","butterfly","button","cake","camera","cat",
            "chair","cheese","clock","cookie","crib","dog","doll","fan","guitar","hat","jacket",
            "juice","key","knife","leaves","lock","meat","necklace","pants","pen","phone",
            "pipe","pizza","ring","rock","rug","sandwich","shoe","socks","spoon","stool",
            "tape","tent","train","tree","trumpet","turtle","tv","umbrella","watch")
alts <- list(bike=c("bike","bicycle"), tv=c("tv","television"), phone=c("telephone","phone"),
             socks=c("sock","socks"), boot=c("boots","boot"), leaves=c("leaf","leaves"),
             cat=c("cat","kitty","kitty cat"), dog=c("dog","doggy","puppy"),
             doll=c("doll","dolly"), airplane=c("airplane","plane"), key=c("key","keys"),
             watch=c("watch","wristwatch"), rock=c("rock","stone"), tape=c("tape (n)","tape"))

norm <- function(x) {
  x <- tolower(x); x <- sub(" \\(.*\\)$", "", x); trimws(x)
}

run <- function(form, ages, measure_fun, measure_name) {
  items <- get_item_data(language = "English (American)", form = form) |>
    filter(item_kind == "word")
  items$defn <- norm(items$item_definition)
  match_word <- function(w) {
    cands <- if (w %in% names(alts)) alts[[w]] else w
    hit <- items |> filter(defn %in% cands)
    if (nrow(hit)) hit$item_id[1] else NA_character_
  }
  ids <- sapply(konkle, match_word)
  matched <- konkle[!is.na(ids)]
  cat(sprintf("%s: matched %d / %d words; unmatched: %s\n", form, length(matched), length(konkle),
              paste(konkle[is.na(ids)], collapse = " ")))
  dat <- get_instrument_data(language = "English (American)", form = form,
                             items = unname(ids[!is.na(ids)]), administration_info = TRUE)
  dat |>
    filter(age %in% ages) |>
    mutate(word = matched[match(item_id, ids[!is.na(ids)])],
           knows = measure_fun(value)) |>
    group_by(age, word) |>
    summarise(p = mean(knows, na.rm = TRUE), n_admin = n(), .groups = "drop") |>
    mutate(form = form, measure = measure_name)
}

wg <- run("WG", seq(8, 18, 2), \(v) v %in% c("understands", "produces"), "understands")
ws <- run("WS", seq(16, 30, 2), \(v) v == "produces", "produces")
items <- bind_rows(wg, ws)
write.csv(items, "results/wordbank_anchors_items.csv", row.names = FALSE)
anchors <- items |>
  group_by(form, measure, age) |>
  summarise(n_items = n(), mean_p_know = mean(p), pred_4afc = 100 * mean(p + (1 - p) / 4),
            .groups = "drop")
write.csv(anchors, "results/wordbank_anchors.csv", row.names = FALSE)
print(as.data.frame(anchors))
