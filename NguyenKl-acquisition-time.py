
import re
import os
import glob
import pandas as pd
from natsort import natsorted, ns
# Path: NguyenKl-acquisition-time.py

# read the paths from the folder
file_paths = natsorted(glob.glob('/Users/mona/Library/CloudStorage/GoogleDrive-xinqili16@g.ucla.edu/My Drive/Registration/NguyenKl_Fmbv_Mona/*bSSFP*'))

# create an empty dataframe to store the results
df = pd.DataFrame(columns=['Subject', 'AcquisitionTime'])

# loop through the file paths
for path in file_paths:
    # extract the subject name from the path
    subject = f"{os.path.basename(path)[:-5]}_T1_Mona"
    # extract the acquisition time from the path
    find = re.findall(r'\d+mg', subject)
    if len(find) == 0:
        acquisition_time = 0
    else:
        acquisition_time = int(find[0][:-2])
    # add the subject name and acquisition time to the dataframe
    df = df.append({'Subject': subject, 'AcquisitionTime': acquisition_time}, ignore_index=True)

# print the resulting dataframe
print(df)

df.to_csv('/Users/mona/Library/CloudStorage/GoogleDrive-xinqili16@g.ucla.edu/My Drive/Registration/NguyenKl_Fmbv_Mona/acquisitionTime.csv', index=False)
