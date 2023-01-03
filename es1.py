import numpy as np
import pandas as pd
from auxi import countElems, substitute_with_mean, plot_densities, one_hot_encode, plot_pcs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
RS = 317518
FILE_NAME = "cla4lsp22_bikez_curated.csv"

np.random.seed(RS)

df_tot = pd.read_csv(FILE_NAME)
r = np.random.randint(0,3)

#data preparation - random extractions
no_missing = (~df_tot["Year"].isna()).all()
year_mask = (df_tot["Year"]%3) == r
workdf = df_tot[year_mask]
#print(((df_tot[year_mask]["Year"]%3) == r).all())

labels = ["Brand", "Model", "Year", "Category", "Rating"]
features = set(workdf.columns).difference(set(labels))
candidates_for_removal = ["Front brakes","Rear brakes", "Front tire", "Rear tire", "Front suspension","Rear suspension"]
np.random.shuffle(candidates_for_removal)
features = list(features.difference(set(candidates_for_removal[:2])))

workdf = workdf[features + labels]


#dropping features with too many NaNs and then dropping the NaN rows left
features_to_drop = workdf[features].columns[workdf[features].isna().sum(axis=0) > workdf.shape[0]/4]
print(len(features_to_drop),features_to_drop)
workdf.drop(features_to_drop, axis=1, inplace=True)
features = list(set(features).difference(set(features_to_drop)))
workdf.dropna(subset=features,inplace=True)
#print(features)

# #data preparation - dealing with missing values
# #the features that have missing values and their types
#print(workdf.isna().sum(axis=0))
#features_na = workdf.columns[workdf.isna().sum(axis=0) > 0]

# print(len(features_na))
#print(workdf[features_na].dtypes)
# #only Model seems of to be categorical, while rating might be. The others defenetly are not
# #displaying their values range
# print(f"Range of Model = {set(workdf['Model'].dropna())}\nRange of Rating = {set(workdf['Rating'].dropna())}")

# #Rating seems to be confined between 0 and 5 (or 1 and 5), but it is distributed like a regular float, hence it will be trated
# #as continuous. Model is categorical, as expected, and a reasonable aproach is to substitute the NaNs with an "other" category

# #Before adding the term "other" to describe di NaN models, first I need to make sure that "other" is not already a category
# #and that there are not present any other placeholder categories
# print(f"Is the term 'other' already present in Model? {'other' in set(workdf['Model'].dropna())}")
# #first easy check is to see if the model is unique. That can be achieved bu summing the number of unique models with the numeber of
# #NaNs and comparing the result with the number of rows
# print(len(set(workdf['Model'].dropna())) + workdf['Model'].isna().sum(), workdf.shape[0])
# #seeing that they are not equal I made a little function to count the number of occurrences and to display them in order
# print(countElems(workdf['Model'].dropna())[:10])
# #if there were placeholder positions, I would expect them in the first spots, but there don't seem to be any

#Substituting NaN with "other" in Model
workdf['Model'].fillna("other", inplace=True)
# #Now other should be a present category
# print(countElems(workdf['Model'].dropna())[:10])


# other_na_feat = list(set(features_na).difference(set(["Model"])))
# # plot_densities(workdf, other_na_feat, (3,4), "Before")

# for feature_na in other_na_feat:
#     workdf[feature_na] = workdf[feature_na].interpolate(method='linear').ffill().bfill()

categorical_col = workdf[features].select_dtypes(include=[object]).columns
new_df = one_hot_encode(workdf[features],categorical_col)
new_features_names = new_df.columns
print(new_df.shape)
print( new_df.columns)
Xworkdf_std = StandardScaler().fit_transform(new_df)
#TODO impove
#print(f"Xworkdf_std means = {Xworkdf_std.mean(axis=0)}\nXworkdf_std stds= {Xworkdf_std.std(axis=0)}")
Xworkdf_mm = MinMaxScaler((0,1)).fit_transform(new_df)
#TODO impove
#print(f"Xworkdf_mm means = {Xworkdf_mm.mean(axis=0)}\nXworkdf_mm stds= {Xworkdf_mm.std(axis=0)}")
# print(rating)
# plot_densities(workdf, other_na_feat, (3,4), "After")
# plt.show()

pca_std = PCA()
pca_mm = PCA()

std_pca = pca_std.fit(Xworkdf_std)
mm_pca = pca_mm.fit(Xworkdf_mm)
print(Xworkdf_std.shape, std_pca.components_.shape, mm_pca.components_.shape)
x = np.arange(len(pca_std.explained_variance_ratio_))
cum_sum_std = np.cumsum(pca_std.explained_variance_ratio_)
cum_sum_mm = np.cumsum(pca_mm.explained_variance_ratio_)
threshold_std =  np.where(cum_sum_std > 0.35)[0][0]
threshold_mm =  np.where(cum_sum_mm > 0.35)[0][0]

#fig, ax = plt.subplots(1,2)
# print(threshold_std, threshold_mm)
# ax[0].plot(x,cum_sum_std)
# ax[0].axvline(x = threshold_std, color = 'r', label = '>35%')
# ax[0].set_title("Standard scaler")
# ax[1].plot(x, cum_sum_mm)
# ax[1].axvline(x = threshold_std, color = 'r', label = '>35%')
# ax[1].set_title("MinMax scaler")

# plt.tight_layout()
# plt.show()

#eps np.sqrt(1/pca.n_features_)
num_pcs_std = min(threshold_std,5)
num_pcs_mm = min(threshold_mm,5)
std_pca =  PCA(num_pcs_std)
pca_mm =  PCA(num_pcs_mm)

scores_std = std_pca.fit_transform(Xworkdf_std)
sorted_indexes = scores_std.argsort(axis=0)

#first PC: low CC - high CC
#second PC: new cross motos - old long range motos
#third PC: sportsy old bikes - new midrange confort
#forth PC: quads vs classic shape bikes
#fift PC: Unknown

NUM_SAMPLES = 20
for i in range(num_pcs_std):
    print(f"\nPC nr. {i+1}--------------------------------------------------------------------------------------\n")
    mins = workdf[['Model',"Year"]].iloc[sorted_indexes[:NUM_SAMPLES,i]]
    mins["score"] = scores_std[:,i][sorted_indexes[:NUM_SAMPLES,i]]
    print("MINS:\n",mins,"\n")
    maxs = workdf[['Model',"Year"]].iloc[sorted_indexes[-NUM_SAMPLES:,i]]
    maxs["score"] = scores_std[:,i][sorted_indexes[-NUM_SAMPLES:,i]]
    print("MAXES:\n",maxs)

scores_mm = pca_mm.fit_transform(Xworkdf_std)
sorted_indexes = scores_mm.argsort(axis=0)

#first PC: low CC - high CC
#second PC: new cross motos - old long range motos

NUM_SAMPLES = 20
for i in range(num_pcs_mm):
    print(f"\nPC nr. {i+1}--------------------------------------------------------------------------------------\n")
    mins = workdf[['Model',"Year"]].iloc[sorted_indexes[:NUM_SAMPLES,i]]
    mins["score"] = scores_mm[:,i][sorted_indexes[:NUM_SAMPLES,i]]
    print("MINS:\n",mins,"\n")
    maxs = workdf[['Model',"Year"]].iloc[sorted_indexes[-NUM_SAMPLES:,i]]
    maxs["score"] = scores_mm[:,i][sorted_indexes[-NUM_SAMPLES:,i]]
    print("MAXES:\n",maxs)

w_pcs_std = std_pca.components_.T
w_pcs_mm = pca_mm.components_.T
# plot_pcs(w_pcs_std[:,:2],new_features_names,(2,1),"", np.sqrt(1/pca_std.n_features_))
# plot_pcs(w_pcs_std[:,2:4],new_features_names,(2,1),"", np.sqrt(1/pca_std.n_features_))
# plot_pcs(w_pcs_std[:,4:],new_features_names,(2,1),"", np.sqrt(1/pca_std.n_features_))
# #plot_pcs(w_pcs_mm,new_features_names,(2,1),"", np.sqrt(1/pca_mm.n_features_))
#plt.show()

# scores = np.array([[i, silhouette_score(scores_mm,KMeans(i,random_state=RS).fit_predict(scores_mm))] for i in range(2,10)])
# best_k = scores[:,0][scores[:,1].argmax()]
# sns.barplot(x=scores[:,0], y=scores[:,1])
# plt.show()
# best_k = 3
# best_k_means_mm = KMeans(int(best_k),random_state=RS)
# best_prediction_mm =best_k_means_mm.fit_predict(scores_mm)
# centroids = best_k_means_mm.cluster_centers_
# #sns.scatterplot(x=scores_mm[:,0],y=scores_mm[:,1], hue=best_prediction_mm)
# for i in range(centroids.shape[0]):
#     sns.barplot(y=centroids[i,:], x=np.arange(centroids.shape[1]))
#     plt.show()


# scores = np.array([[i, silhouette_score(scores_std,KMeans(i,random_state=RS).fit_predict(scores_std))] for i in range(2,10)])
# best_k = scores[:,0][scores[:,1].argmax()]
# # sns.barplot(x=scores[:,0], y=scores[:,1])
# # plt.show()
# best_k = scores[:,0][scores[:,1].argmax()]
#print(best_k)
best_k = 3
best_k_means_std = KMeans(int(best_k),random_state=RS)
best_prediction_std =best_k_means_std.fit_predict(scores_std)
centroids = best_k_means_std.cluster_centers_
#sns.scatterplot(x=scores_std[:,0],y=scores_std[:,1], hue=best_prediction_mm)
for i in range(centroids.shape[0]):
    sns.barplot(y=centroids[i,:], x=np.arange(centroids.shape[1]))
    plt.show()