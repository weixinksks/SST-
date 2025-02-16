from sklearn.impute import SimpleImputer
import numpy as np

# 示例数据
pattern_data = np.array([[np.nan, 2, 3,2, 3],
                         [np.nan, 7, 6,np.nan, 3],
                         [np.nan, 8, 9,2, 3]])


median_data = np.nanmedian(pattern_data)
imputer = SimpleImputer(strategy='constant', fill_value=median_data)
filled_data = imputer.fit_transform(pattern_data)


print(median_data)
print(filled_data)














