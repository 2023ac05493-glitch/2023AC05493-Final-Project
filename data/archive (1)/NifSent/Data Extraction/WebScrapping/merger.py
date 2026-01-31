# Merge the files in ../data/archive (1)/NifSent/Data Extraction/WebScrapping/investing.com/Data and ../data/archive (1)/NifSent/Data Extraction/WebScrapping/Reddit/r-indianews/Data

import os
import pandas as pd

merged_df= pd.DataFrame()
for root, dirs, files in os.walk("./data/archive (1)/NifSent/Data Extraction/WebScrapping/investing.com/Data"):
    for filename in files:
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(root, filename))
            # Union all into one df
            if 'merged_df' not in locals():
                merged_df = df
            else:
                merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)
merged_df.to_csv("./data/archive (1)/NifSent/Data Extraction/WebScrapping/merged_scrapping.csv", index=False)


