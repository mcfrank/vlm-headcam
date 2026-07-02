#!/usr/bin/env Rscript
# Enrollment visualization: cumulative hours of recorded video per child vs. the child's
# age at recording. Rebuilt from metadata/videos.csv (Airtable export) + demogs.csv birthdates.
# Usage: Rscript scripts/enrollment_viz.R   ->  writes scripts/enrollment_viz.png
suppressMessages(library(tidyverse)); suppressMessages(library(lubridate))

videos <- read_csv("metadata/videos.csv", show_col_types = FALSE)
names(videos)[1] <- "unique_video_id"                 # strip UTF-8 BOM on first column
demogs <- read_csv("metadata/demogs.csv", show_col_types = FALSE)

parse_flex <- function(x) suppressWarnings(parse_date_time(x, orders = c("mdy", "ymd", "dmy")))

d <- videos %>%
  filter(dataset == "BV-main") %>%                     # the longitudinal infant corpus
  left_join(demogs %>% select(subject_id, dob = date_birth_rounded), by = "subject_id") %>%
  mutate(
    rec_date = parse_flex(date),
    dob      = parse_flex(dob),
    dur_h    = suppressWarnings(as.numeric(duration_sec)) / 3600,
    age_mo   = as.numeric(rec_date - dob) / 30.437
  ) %>%
  filter(!is.na(age_mo), !is.na(dur_h), dur_h > 0, age_mo > 0, age_mo < 60)

# per child: order by recording date, accumulate hours
d <- d %>%
  arrange(subject_id, rec_date) %>%
  group_by(subject_id) %>%
  mutate(cum_h = cumsum(dur_h)) %>%
  ungroup()

cat(sprintf("plotting %d videos, %d children, %.0f total hours\n",
            nrow(d), n_distinct(d$subject_id), sum(d$dur_h)))

p <- ggplot(d, aes(x = age_mo, y = cum_h, group = subject_id, color = subject_id)) +
  geom_line(linewidth = 0.5, alpha = 0.9) +
  geom_point(size = 0.5, alpha = 0.7) +
  scale_x_continuous(breaks = seq(0, 60, 10)) +
  labs(x = "Age (in months) during recording", y = "Cumulative hours of videos") +
  guides(color = "none") +
  theme_gray(base_size = 14)

ggsave("scripts/enrollment_viz.png", p, width = 8, height = 5.5, dpi = 200, bg = "white")
cat("wrote scripts/enrollment_viz.png\n")
