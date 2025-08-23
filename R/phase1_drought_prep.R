# Phase 1: Data prep & labels (past-only) — saves processed CSVs for Python
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(zoo)
  library(SPEI)
})

# ---- Paths (relative) ----
raw_path  <- file.path("data","raw","All_Stations_Combined.csv")
out_dir   <- file.path("data","processed")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load ----
stopifnot(file.exists(raw_path))
combined <- fread(raw_path)

# Handle date: supports either Year-Month or Date column
if ("Year-Month" %in% names(combined)) {
  combined <- combined %>%
    mutate(Date = as.Date(paste0(`Year-Month`, "-01"))) %>%
    select(-`Year-Month`, tidyselect::everything())
} else if (!"Date" %in% names(combined)) {
  stop("Provide either 'Year-Month' or 'Date' column.")
} else {
  combined$Date <- as.Date(combined$Date)
}

# Optional: drop unnamed junk cols if present
drop_cols <- intersect(c("V14","V15","V16","V17"), names(combined))
combined  <- select(combined, -all_of(drop_cols))

# ---- Forward-fill NDVI/EVI (safe, past-only) ----
ff <- function(x) na.locf(x, na.rm = FALSE)
combined <- combined %>% arrange(Station, Date) %>%
  group_by(Station) %>%
  mutate(EVI  = ff(EVI),
         NDVI = ff(NDVI),
         Rain = ff(Rain)) %>%
  ungroup()

# ---- SPI(3,12) per station (tryCatch for robustness) ----
calc_spi <- function(rain, scale) {
  out <- tryCatch({
    as.numeric(SPEI::spi(ts(rain, frequency = 12), scale = scale)$fitted)
  }, error = function(e) rep(NA_real_, length(rain)))
  out
}

combined <- combined %>%
  group_by(Station) %>%
  arrange(Date, .by_group = TRUE) %>%
  mutate(
    SPI_3  = calc_spi(Rain, 3),
    SPI_12 = calc_spi(Rain, 12)
  ) %>% ungroup()

# ---- Drought label: expanding 20th percentile with LAG (past-only) ----
make_label <- function(sm) {
  # expanding quantile up to i, then lag by 1 month
  thr <- sapply(seq_along(sm), function(i) quantile(sm[1:i], probs = 0.2, na.rm = TRUE))
  as.integer(sm < dplyr::lag(thr, 1))
}

combined <- combined %>%
  group_by(Station) %>%
  arrange(Date, .by_group = TRUE) %>%
  mutate(Drought_Label = make_label(sm_root_zone)) %>%
  ungroup()

# ---- Save per-station ----
feature_names <- c("SPI_3","SPI_12","mean_Temp","Rain","Runoff","NDVI")
writeLines(feature_names, "feature_names.txt")

split(combined, combined$Station) |>
  lapply(function(df) {
    safe <- gsub("[^A-Za-z0-9]", "_", unique(df$Station))
    fwrite(df, file.path(out_dir, paste0("processed_station_", safe, ".csv")))
  })

message("Done. Files in ", normalizePath(out_dir), " + feature_names.txt at repo root.")

