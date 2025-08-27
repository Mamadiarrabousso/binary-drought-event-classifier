
# Phase 1: Data prep for drought classifier.


  library(data.table)
  library(dplyr)
  library(zoo)
  library(lubridate)
  library(SPEI)


# Paths
raw_csv <- "data/raw/All_Stations_Combined.csv"
out_dir <- "data/processed"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Load
if (!file.exists(raw_csv)) stop("Put All_Stations_Combined.csv in data/raw/")
df <- fread(raw_csv)

# Date + keep columns
df <- df %>%
  mutate(Date = as.Date(paste0(`Year-Month`, "-01"))) %>%
  select(-`Year-Month`)

# Forward-fill NDVI/EVI using past only (per station)
df <- df %>%
  arrange(Station, Date) %>%
  group_by(Station) %>%
  mutate(
    EVI  = na.locf(EVI,  na.rm = FALSE),
    NDVI = na.locf(NDVI, na.rm = FALSE)
  ) %>%
  ungroup()

# SPI helper (robust to failures)
calc_spi <- function(rain_vec, scale) {
  tsobj <- ts(rain_vec, frequency = 12)
  out <- tryCatch({
    as.numeric(SPEI::spi(tsobj, scale = scale)$fitted)
  }, error = function(e) rep(NA_real_, length(rain_vec)))
  out
}

# Add SPI_3 and SPI_12 per station
df <- df %>%
  group_by(Station) %>%
  arrange(Date, .by_group = TRUE) %>%
  mutate(
    SPI_3  = calc_spi(Rain, 3),
    SPI_12 = calc_spi(Rain, 12)
  ) %>%
  ungroup()

# Past-only drought label: expanding 20th percentile of sm_root_zone, lagged 1 month
label_fun <- function(x) {
  thr <- sapply(seq_along(x), function(i) quantile(x[1:i], 0.2, na.rm = TRUE))
  as.integer(x < dplyr::lag(thr))
}
df <- df %>%
  group_by(Station) %>%
  arrange(Date, .by_group = TRUE) %>%
  mutate(Drought_Label = label_fun(sm_root_zone)) %>%
  ungroup()

# Feature list used in Python
feature_names <- c("SPI_3","SPI_12","mean_Temp","Rain","Runoff","NDVI")
writeLines(feature_names, file.path(out_dir, "feature_names.txt"))

# Save 2 source + 1 target station (adjust names if needed)
keep <- c("BAIRNSDALE_AIRPORT_Combined","MORWELL_LATROBE_VALLEY","GELANTIPY")
safe <- function(s) gsub("[^A-Za-z0-9]", "_", s)
for (st in keep) {
  out <- file.path(out_dir, paste0("processed_station_", safe(st), ".csv"))
  fwrite(df %>% filter(Station == st), out)
  message("saved: ", out)
}


