# %%
"""
Data visualization and analysis for SIIM pneumothorax dataset.
Computes case counts, pixel ratios, and writes a summary to Data/data_analysis_summary.txt.
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import data_processing
import mask_functions as ms

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
HEIGHT, WIDTH = 1024, 1024
PIXELS_PER_IMAGE = HEIGHT * WIDTH

# -----------------------------------------------------------------------------
# Load and group data
# -----------------------------------------------------------------------------
# %%
plt.style.use("default")

df = data_processing.load_data()
print("Dataframe loaded!")
df.head()

# %%
def _group_by_image(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("ImageId")["EncodedPixels"]
        .apply(list)
        .reset_index()
        .rename(columns={"EncodedPixels": "EncodedPixels_list"})
    )

df_all_grouped = _group_by_image(df)
df_all_grouped.head()

# %%
df_pos = df[df["EncodedPixels"] != "-1"]
print("Dataframe of pneumo-positive rows loaded.")
df_pos.head()

# %%
df_grouped_pos = _group_by_image(df_pos)
print("Grouped positive pneumo done!")
df_grouped_pos.head()

# -----------------------------------------------------------------------------
# Case-level counts and ratios
# -----------------------------------------------------------------------------
# %%
count_all_cases = len(df_all_grouped)
count_positive_cases = len(df_grouped_pos)
count_negative_cases = count_all_cases - count_positive_cases

ratio_pos_of_all = (count_positive_cases / count_all_cases) * 100
ratio_pos_to_neg = (count_positive_cases / count_negative_cases) * 100 if count_negative_cases else 0
ratio_neg_of_all = (count_negative_cases / count_all_cases) * 100

print(f"Number of all cases: {count_all_cases}")
print(f"Number of positive cases: {count_positive_cases}")
print(f"Number of negative cases: {count_negative_cases}")
print(f"Ratio of positive cases of all cases: {ratio_pos_of_all:.2f}%")
print(f"Ratio of positive to negative cases: {ratio_pos_to_neg:.2f}%")
print(f"Ratio of negative to all cases: {ratio_neg_of_all:.2f}%")

# -----------------------------------------------------------------------------
# Pixel-level stats (positive images only)
# -----------------------------------------------------------------------------
# %%
total_pos_pixels = 0
total_pixels_in_pos = 0
fractions = []

for _, row in df_grouped_pos.iterrows():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for rle in row["EncodedPixels_list"]:
        part = ms.rle2mask(rle, WIDTH, HEIGHT).T
        mask = np.maximum(mask, part)
    pos = (mask > 0).sum()
    total_pos_pixels += pos
    total_pixels_in_pos += PIXELS_PER_IMAGE
    fractions.append(pos / PIXELS_PER_IMAGE)

total_pixels_all = count_all_cases * PIXELS_PER_IMAGE
ratio_pos_in_pos_images = (total_pos_pixels / total_pixels_in_pos) * 100 if total_pixels_in_pos else 0
ratio_pos_in_all_images = (total_pos_pixels / total_pixels_all) * 100 if total_pixels_all else 0

print("Total positive pixels (unique positives):", total_pos_pixels)
print("Total pixels in positive images:", total_pixels_in_pos)
print(f"Ratio of positive pixels to all pixels in positive cases: {ratio_pos_in_pos_images:.2f}%")
print(f"Ratio of positive pixels to all pixels in ALL images: {ratio_pos_in_all_images:.4f}%")

# -----------------------------------------------------------------------------
# Save summary to file (run after cells above)
# -----------------------------------------------------------------------------
# %%
os.makedirs(config.data_path, exist_ok=True)
summary_path = os.path.join(config.data_path, "data_analysis_summary.txt")

oversample_factor = math.ceil(count_negative_cases / count_positive_cases) if count_positive_cases else 0
total_neg_pixels = total_pixels_all - total_pos_pixels
pixel_ratio_neg_pos = (total_neg_pixels / total_pos_pixels) if total_pos_pixels else 0

summary_lines = [
    f"Number of all cases: {count_all_cases}",
    f"Number of positive cases: {count_positive_cases}",
    f"Number of negative cases: {count_negative_cases}",
    f"Ratio of positive cases of all cases: {ratio_pos_of_all:.2f}%",
    f"Ratio of positive to negative cases: {ratio_pos_to_neg:.2f}%",
    f"Ratio of negative to all cases: {ratio_neg_of_all:.2f}%",
    "",
    f"Total positive pixels (unique positives): {total_pos_pixels}",
    f"Total pixels in positive images: {total_pixels_in_pos}",
    f"Ratio of positive pixels to all pixels in positive cases: {ratio_pos_in_pos_images:.2f}%",
    f"Total pixels in ALL images: {total_pixels_all}",
    f"Ratio of positive pixels to all pixels in ALL images: {ratio_pos_in_all_images:.4f}%",
    "",
    "We have 2 problems: class-wise imbalance (image) and pixel-wise imbalance.",
    "Deciding oversampling rate and class rates:",
    "We will oversample and use class weights.",
    f"For oversampling (image level) we use the case ratios, not pixel ratios: {oversample_factor}",
    f"For error weights (pixel level) we use the pixel ratios (total_neg_pix / pos_pix): {pixel_ratio_neg_pos:.4f}",
]
content = "\n".join(summary_lines)
tmp_path = summary_path + ".tmp"
with open(tmp_path, "w") as f:
    f.write(content)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp_path, summary_path)
print(f"Saved summary to {summary_path}")

# %%
