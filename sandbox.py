import pandas as pd

# Generate selected_features only csv
df = pd.read_csv('D:\\Big_Data\\ADNI\\ADNIMERGE.csv')
columns = list(df.columns)
selected = ['RID', 'DX_bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY', 'ADAS11', 'ADAS13', 'CDRSB', 'FAQ', 'LDELTOTAL', 'MMSE', 'RAVLT_forgetting', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting', 'DX', 'M']
to_drop = [x for x in columns if x not in selected]

for c in to_drop:
    df.drop(c, axis=1, inplace=True)

df.to_csv('D:\\Big_Data\\ADNI\\selected_features.csv')

# normalize file
# df = pd.read_csv('D:\\Big_Data\\ADNI\\clinical_only.csv')

# df['AGE'] = (df['AGE'] - min(df['AGE'])) / (max(df['AGE'] - min(df['AGE'])))
# df['PTEDUCAT'] = (df['PTEDUCAT'] - min(df['PTEDUCAT'])) / (max(df['PTEDUCAT'] - min(df['PTEDUCAT'])))
# df['ADAS11'] = df['ADAS11'] / 70
# df['ADAS13'] = df['ADAS13'] / 85
# df['CDRSB'] = df['CDRSB'] / 18
# df['FAQ'] = df['FAQ'] / 30
# df['LDELTOTAL'] = df['LDELTOTAL'] / 25
# df['MMSE'] = df['MMSE'] / 30
# df['RAVLT_forgetting'] = df['RAVLT_forgetting'] / 15
# df['RAVLT_immediate'] = df['RAVLT_immediate'] / 75
# df['RAVLT_learning'] = df['RAVLT_learning'] / 14
# df['RAVLT_perc_forgetting'] = df['RAVLT_perc_forgetting'] / 100


# df.to_csv('D:\\Big_Data\\ADNI\\normalized.csv')
