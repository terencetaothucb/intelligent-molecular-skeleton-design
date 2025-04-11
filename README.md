# Intelligent-molecular-skeleton-design
Molecular mediators have demonstrated broad applicability in the electrolyte chemistry of lithium-sulfur (Li-S) batteries, transforming sulfur conversion from traditional multiphase reactions to highly reactive pathways. Despite tremendous efforts to elucidate the mechanistic roles of molecular mediators, the influence of molecular skeleton regulation on their mediating effects remains barely understood. Here we propose 2-chloropyrimidine as a potential "pre-mediator" and a model material for molecular skeleton design, which can be in-situ activated into a molecular mediator during sulfur reaction progression via aromatic nucleophilic substitution, homogeneously inducing a rapid redox loop over the entire electrode. Integrating quantum chemistry and machine learning protocols, we develop an intelligent molecular skeleton programming strategy that quantitatively illuminates the structure-property relationship between electronic, geometric and site features of side-chain groups and mediating performance, offering control over the activation rate and mediating activity of pre-mediators. 

# 1. Setting
## 1.1 Enviroments
* Python (Jupyter notebook) 
## 1.2 Python requirements
* python=3.12
* numpy=1.26.4
* tensorflow=2.15.0matplotlib=3.8.4
* keras=2.15.0scipy=1.14.1
* matplotlib=3.9.0scikit-learn=1.4
* scipy=1.13.1pandas=2.2.2

# 2. Datasets
* Raw and processed datasets have been deposited in TBSI-Sunwoda-Battery-Dataset, which can be accessed at [TBSI-Sunwoda-Battery-Dataset](https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset).Raw and processed datasets have been deposited in TBSI-molecular-skeleton-Dataset, which can be accessed at [[TBSI-Sunwoda-Battery-Dataset]](https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset).

# 3. Experiment
## 3.1 Overview
The main workflow includes the construction of two types of homogeneous ensemble models (including six tree models or six linear models) and the calculation of the multi-model strongly physically meaningful weighted average feature importance.
## 3.2 Construction of chemical space
 We employed artificial random sampling to generate a library of 35 non-repetitive CPyr-based molecules (dataset1) from a 196-sample-space considering 7 functional groups and 3 grafting sites (considering the symmetries of site-4 and site-6). This ensures that the database construction process is not disturbed by human factors.

We calculated the 50-dimensional physical properties of 7 functional groups to construct functional group characteristics space (feature set1). To mitigate the risk of multicollinearity-induced dilution effects in subsequent tree-based models, we manually removed features exhibiting physical redundancy and selected 9 electronic features and 6 geometric features to describe a functional group to prevent contribution attenuation

## 3.3 Homogeneous ensemble of tree models
### 3.3.1 Hyperparameter grid search
To enhance the model’s generalization ability and robustness while maintaining interpretability of tree model, we selected six widely used tree-based models as sub-models: Random Forest (RF), Gradient Boosting Regression Tree (GBRT), CatBoost Regression (CBR), AdaBoost Regression (ABR), XGBoost Regression (XGBR), and LightGBM (LGBM). For each sub-model, the optimal hyperparameters were identified via grid search optimization guided by the coefficient of determination $$R^2$$ evaluation metric. 

The $$R^2$$ and Root-Mean-Square Error are employed to reflect the prediction accuracy, which are defined as: 

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}$$

Here, $$y_i$$ represents the value calculated by density functional theory (DFT), $$\hat{y_i}$$ denotes the result predicted by the machine learning model, and $$\bar{y}$$ is the average value of DFT data.

We have specified the hyperparameter spaces of six sub-models. In line with usage requirements, these can be self-defined.
```python
cross_Valid = KFold(n_splits=5, shuffle=True, random_state=76)

# Define the hyperparameter grid for each model.
parameter_XGBR = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'max_depth': [2,3,5],
'subsample': [0.4,0.6,0.8,1]
}

parameter_RF = {
'n_estimators':[100,200,300,400,500],
'max_depth':[2,4,None],
'max_features':['auto','log2','sqrt'],
'min_samples_leaf':[1,2,3,4]
}

parameter_CBR = {
'iterations': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'depth': [3,4,5,6],
'l2_leaf_reg': [1,3,5]
}

parameter_LGBM = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'max_depth': [3,4,5,6],
'subsample': [0.6,0.8,1]
}

parameter_ABR = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'loss': ['linear', 'square'] 
}

parameter_GBRT = {
'n_estimators': [100,200,300,400,500],
'learning_rate': [0.01, 0.05, 0.1,0.15],
'max_depth': [3,4,5,6],
'max_features': ['auto', 'sqrt']
}

# Define model dictionary
estimators = {
'XGBR': XGB.XGBRegressor(random_state=0),
'RF': RandomForestRegressor(random_state=0),
'CBR': CatBoostRegressor(verbose=False, random_state=0),
'LGBM': LGBMRegressor(random_state=0, verbosity=-1),
'ABR': AdaBoostRegressor(random_state=0),
'GBRT': GradientBoostingRegressor(random_state=0)
}

# Map the model names to their corresponding hyperparameters
params_mapping = {
'XGBR': parameter_XGBR,
'RF': parameter_RF,
'CBR': parameter_CBR,
'LGBM': parameter_LGBM,
'ABR': parameter_ABR,
'GBRT': parameter_GBRT
}

grid_searches = {}
for name, estimator in estimators.items():
    params = params_mapping[name] 
    grid_searches[name] = GridSearchCV(estimator, params, scoring='r2', cv=cross_Valid, n_jobs=-1)

# Train the model and find the best parameters.
for name, grid in grid_searches.items():
    grid.fit(x_train, y_train) 
    print(f"{name} best parameters: {grid.best_params_}")
```

### 3.3.2Homogeneous integration and calculation of weighted average feature importance
After finding the optimal hyperparameters for each sub-model, we first construct a model dictionary to initialize the model and inject the found optimal hyperparameters into the model. Then, we perform model integration through Voting Regression and gradually eliminate the worst-performing model (we have reserved an interface at the definition of the model dictionary and can conveniently eliminate models by commenting out specific sub-models). This process continues until the performance of the fusion model is higher than that of all sub-models. For the analysis of model feature importance, to address variations in feature importance magnitudes across different tree models, we normalized the feature importance values of each sub-model. The R2 value of each sub-model in a single training was used as a weight to compute a weighted average of the feature importance, which was assigned as the fusion model’s feature importance for that iteration.
```python
# Define a function to initialize the model dictionary.
def initialize_best_estimators(grid_searches):
    return {
        'XGBR': XGB.XGBRegressor(**grid_searches['XGBR'].best_params_, random_state=42),
        'RF': RandomForestRegressor(**grid_searches['RF'].best_params_, random_state=42),
        'CBR': CatBoostRegressor(**grid_searches['CBR'].best_params_, verbose=False, random_state=42),
        'LGBM': LGBMRegressor(**grid_searches['LGBM'].best_params_, random_state=42, verbosity=-1),
        'ABR': AdaBoostRegressor(**grid_searches['ABR'].best_params_, random_state=42),
        'GBRT': GradientBoostingRegressor(**grid_searches['GBRT'].best_params_, random_state=42)
    }

# Define functions for calculating model performance and feature importance.
def evaluate_models(seed, X, Y, grid_searches, submodel_r2_sums, submodel_rmse_sums, num_seeds):
    cross_validator = KFold(n_splits=10, shuffle=True, random_state=seed)
    best_estimators = initialize_best_estimators(grid_searches)
    # Create VotingRegressor and only keep the uncommented models.
    submodels = [(name, estimator) for name, estimator in best_estimators.items() if estimator is not None]
    voting_regressor = VotingRegressor(submodels)

    # Calculate the mean values of R2 and RMSE for each sub-model.
    submodel_r2_means = {}
    submodel_rmse_means = {}
    rmse_scorer = make_scorer(mean_squared_error, squared=False)
    
    # Used to store the feature importance of each model.
    submodel_feature_importances = {}
    feature_importances_weighted_sum = np.zeros(X.shape[1])
    total_r2 = 0

    for name, estimator in submodels:
        r2_scores = cross_val_score(estimator, X, Y, cv=cross_validator, scoring='r2', n_jobs=-1)
        rmse_scores = cross_val_score(estimator, X, Y, cv=cross_validator, scoring=rmse_scorer, n_jobs=-1)
        submodel_r2_means[name] = np.mean(r2_scores)
        submodel_rmse_means[name] = np.mean(rmse_scores)
        submodel_r2_sums[name] += submodel_r2_means[name]
        submodel_rmse_sums[name] += submodel_rmse_means[name]

        # Calculate feature importance
        estimator.fit(X, Y)
        importances = estimator.feature_importances_
        
        # normalization processing
        importances_normalized = importances / np.sum(importances)
        
        submodel_feature_importances[name] = dict(zip(X.columns, importances_normalized))
        feature_importances_weighted_sum += importances_normalized * submodel_r2_means[name]
        total_r2 += submodel_r2_means[name]

    # Calculate the mean R2 and weighted RMSE of the fusion model.
    voting_regressor_r2_mean = np.mean(cross_val_score(voting_regressor, X, Y, cv=cross_validator, scoring='r2', n_jobs=-1))
    voting_regressor_rmse_mean = np.mean(cross_val_score(voting_regressor, X, Y, cv=cross_validator, scoring=rmse_scorer, n_jobs=-1))

    # Calculate weighted feature importance
    weighted_feature_importances = feature_importances_weighted_sum / total_r2 if total_r2 != 0 else feature_importances_weighted_sum

    weighted_feature_importances_dict = dict(zip(X.columns, weighted_feature_importances))

    return {
        'submodel_r2_means': submodel_r2_means,
        'submodel_rmse_means': submodel_rmse_means,
        'voting_regressor_r2_mean': voting_regressor_r2_mean,
        'voting_regressor_rmse_mean': voting_regressor_rmse_mean,
        'submodel_feature_importances': submodel_feature_importances,
        'weighted_feature_importances': weighted_feature_importances_dict
}
```

### 3.3.3 Traversal of random seeds from 0 to 99.
To mitigate single-training bias, we adjusted the dataset partition and iterated over random seeds from 0 to 99. The feature importance values from 100 training iterations were arithmetically averaged to obtain the final feature importance, which are defined as:

$$\ FI_i = \frac{1}{100} \sum_{t=0}^{99} \left( \frac{\sum_{m=1}^M R^{2^{(t,m)}} f_{i_i}^{(t,m)}}{\sum_{m=1}^M R^{2^{(t,m)}}} \right) \$$

Among them, $$\ FI_i \$$ represents the feature importance extracted by the fusion model for feature i, $$R^{2^{(t,m)}}$$ represents the $$R^2$$ value of the m-th sub-model in the t-th iteration, and $$f_{i_i}^{(t,m)}$$ represents the importance of feature i assigned by the m-th sub-model for feature i in the t-th iteration.
```python
# Define and save the results of all seeds.
all_results = []
seeds = range(100)

# Initialize the accumulated sum of R2 and RMSE of the submodel.
submodel_r2_sums = {name: 0.0 for name in initialize_best_estimators(grid_searches).keys()}
submodel_rmse_sums = {name: 0.0 for name in initialize_best_estimators(grid_searches).keys()}

# Initialize the weighted sum of feature importance accumulations of the fusion model.
weighted_feature_importances_sums = np.zeros(len(X.columns))

# Traverse the seeds and save the results.
for seed in seeds:
    result = evaluate_models(seed, X, Y, grid_searches, submodel_r2_sums, submodel_rmse_sums, len(seeds))
    all_results.append(result)
    weighted_feature_importances_sums += np.array(list(result['weighted_feature_importances'].values()))
    print(f"")
    print(f"Seed {seed} - Voting Regressor R2 Mean: {result['voting_regressor_r2_mean']}, RMSE Mean: {result['voting_regressor_rmse_mean']}")
    print("Submodel R2 and RMSE Means:")
    for name in result['submodel_r2_means'].keys():
        print(f"{name}: R2 Mean = {result['submodel_r2_means'][name]}, RMSE Mean = {result['submodel_rmse_means'][name]}")

# Calculate the average value of the weighted feature importance of the fusion model for all seeds.
avg_weighted_feature_importances = weighted_feature_importances_sums / len(seeds)

# Bind the feature name and the corresponding weighted feature importance and sort in descending order of importance.
sorted_feature_importances = sorted(zip(X.columns, avg_weighted_feature_importances), key=lambda x: x[1], reverse=True)
```

### 3.3.4 Results output
Finally, output the R2 and RMSE of each sub-model and the fusion model. Sort the feature importance of the weighted average output of the fusion model under the last 100 random seeds.
```python
# Output the R2, RMSE of the fusion model and each sub-model and the sorted feature importance.
avg_voting_regressor_r2_mean = np.mean([result['voting_regressor_r2_mean'] for result in all_results])
avg_voting_regressor_rmse_mean = np.mean([result['voting_regressor_rmse_mean'] for result in all_results])

print(f"")
print(f"Average Voting Regressor R2 Mean: {avg_voting_regressor_r2_mean}")
print(f"Average Voting Regressor Weighted RMSE Mean: {avg_voting_regressor_rmse_mean}")
print("Average Submodel R2 and RMSE Means:")
for name in submodel_r2_sums.keys():
    print(f"{name}: R2 Mean = {submodel_r2_sums[name] / len(seeds)}, RMSE Mean = {submodel_rmse_sums[name] / len(seeds)}")
    
# Draw a bar chart of feature importance.
features, importances = zip(*sorted_feature_importances)
plt.figure(figsize=(10, 8))
sns.barplot(x=list(importances), y=list(features))
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances Sorted by Importance')
plt.show()

print("Sorted Feature Importances:")
for feature, importance in sorted_feature_importances:
print(f"{feature}: {importance}")
```

## 3.4 Homogeneous integration of linear models 
For the construction of descriptors，we identified the top-ranked features across different sites as strong correlation factors and used them to construct functional group indexes. We utilized six common linear models as sub-models: Linear Regression (LR), Ridge Regression (RR), Least Angle Regression (LAR), Elastic Net Regression (ENR), Partial Least Squares Regression (PLSR), and Support Vector Regression (SVR). Except for setting the kernel to linear in SVR, we retain the default values for all other hyperparameters (But we still reserve an interface to implement custom hyperparameters). The homogeneous integration process of linear models is similar to the above process. For details, see `linear_voting_tunning.py`，`linear_voting_ensemble.py`.

# 4. Access
Access the raw data and processed features [here]((https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset)) under the [MIT licence](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation/blob/main/LICENSE). Correspondence to [Terence (Shengyu) Tao](terencetaotbsi@gmail.com) and CC Prof. [Xuan Zhang](xuanzhang@sz.tsinghua.edu.cn) and [Guangmin Zhou](guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.
# 5. Acknowledgements
[Yifei Zhu](zhuyifeiedu@126.com) and [Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) at Tsinghua Berkeley Shenzhen Institute conceived and formulated the algorithms, deposited model/experimental code, and authored this guideline document drawing on supplementary materials.
