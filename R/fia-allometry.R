require(tidyverse)
require(ggpmisc)
require(stringr)
require(raster)

# TODO: update to current repository data/ directory paths
fia <- read_csv("~/Documents/Salo/BigTrees/fia_calif_plot_sj.csv")
fia_cols <- data.frame(names(fia))

# create shapefile from plot coordinates
fia_shp <- fia[, c("geometry", "PLOT")]
fia_shp$PLOT <- as.integer(fia_shp$PLOT)
fia_shp$lat <- as.numeric(str_split(fia_shp$geometry, "\\|", simplify = T)[, 2])
fia_shp$long <- as.numeric(str_split(fia_shp$geometry, "\\|", simplify = T)[, 1])
fia_shp <- fia_shp[-1]
fia_shp <- fia_shp[!duplicated(fia_shp$PLOT), ]

coordinates(fia_shp) <- ~ long + lat
proj4string(fia_shp) <- CRS("+proj=longlat +datum=WGS84")
fia_shp <- spTransform(fia_shp, CRS("+init=EPSG:32610"))
raster::shapefile(fia_shp, "~/Documents/Salo/BigTrees/fia_calif_plot_sj.shp", overwrite = T)

# TPA_UNA = expansion factor
# TPA_UNADJ
# It is dependent on the tree size (See page 9 of the FIA Database Description and Users Guide, version 3.0)
# D:/Dropbox/sierra_large_trees/FIADB_user guide_v3-0_P2_06_01_07.pdf
# trees >5" DBH and <24" DBH have an expansion factor of 6.018046, which is approx 1 / (4 * (1/24))
# In actuality the formula is (1 / (((pi * (24^2)) / 43560) * 4)) = 6.018046
#                                           24 ft radius, 4 subplots
# TREES <5" DBH: (1 / (((pi * (6.8^2)) / 43560) * 4)) = 74.96528 --> Microplot (6.8ft radius, 145.2672 ft2)
# TREES >5" & <24" DBH: (1 / (((pi * (24^2)) / 43560) * 4)) = 6.018046 --> Subplot (24ft radius, 1809.557 ft2)
# TREES >24" DBH: (1 / (((pi * (58.9^2)) / 43560) * 4)) = 0.9991885 --> Macroplot (58.9ft radius, 10898.84 ft2)


plot_fia <- fia %>%
  select(ClassName, PLOT, SUBP, HT, DIA, basal_area_ft2, bapa) %>%
  group_by(ClassName, PLOT) %>%
  summarize(ht_max = max(HT), dia_mean = mean(DIA), ba_sum = sum(basal_area_ft2))
subplot_fia <- fia %>%
  select(ClassName, PLOT, SUBP, HT, DIA, basal_area_ft2, bapa) %>%
  group_by(ClassName, PLOT, SUBP) %>%
  summarize(ht_max = max(HT), dia_mean = mean(DIA), ba_sum = sum(basal_area_ft2))
ind_fia <- fia %>% select(ClassName, PLOT, SUBP, HT, DIA, basal_area_ft2, bapa)

lm.formula <- y ~ x
exp.formula <- y ~ x * exp(log(z))

# Plot
ggplot(data = plot_fia, aes(x = ht_max, y = ba_sum)) +
  geom_point(alpha = 0.3, color = "blue", stroke = 0, shape = 16, size = 2) +
  geom_smooth(method = "glm", se = FALSE, color = "#00000080", linetype = 1, formula = exp.formula) +
  stat_poly_eq(
    formula = lm.formula,
    aes(label = paste(..rr.label..)),
    parse = TRUE
  ) +
  scale_y_continuous(limit = c(-200, NA), expand = c(0, 0)) +
  facet_wrap(~ClassName, nrow = 4) +
  labs(title = "Agg: Plot") +
  theme_bw() +
  theme(strip.background = element_blank())

# Subplot
ggplot() +
  geom_point(data = subplot_fia, aes(x = ht_max, y = ba_sum), alpha = 0.3, color = "blue", stroke = 0, shape = 16, size = 2) +
  facet_wrap(~ClassName, nrow = 4) +
  labs(title = "Agg: Subplot")

# Ind
ggplot() +
  geom_point(data = ind_fia, aes(x = HT, y = basal_area_ft2), alpha = 0.3, color = "blue", stroke = 0, shape = 16, size = 2) +
  facet_wrap(~ClassName, nrow = 4, scales = "free") +
  labs(title = "Agg: Individuals")
