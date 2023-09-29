using MultivariateStats
using Plots

# Sample data (replace this with your dataset)
data = [1.0 2.0 3.0;
        4.0 5.0 6.0;
        7.0 8.0 9.0;
        10.0 11.0 12.0]

# Perform PCA
pca_model = fit(PCA, data; maxoutdim = 2)  # Reduce to 2 principal components

# Transform the data into the reduced-dimensional space
reduced_data = transform(pca_model, data)

# Get the principal components
pcs = principalvars(pca_model)

# Variance explained by each principal component
explained_variance = principalvars(pca_model) / sum(principalvars(pca_model))

# Scatter plot of original data
scatter(data[:, 1], data[:, 2], label="Original Data", legend=:topleft, xlabel="Feature 1", ylabel="Feature 2", ratio=1)

# Scatter plot of reduced data
scatter(reduced_data[:, 1], reduced_data[:, 2], label="Reduced Data", legend=:topleft, xlabel="Principal Component 1", ylabel="Principal Component 2", ratio=1)

# Variance explained by each principal component
bar(explained_variance, label="Explained Variance", legend=:topright, xlabel="Principal Component", ylabel="Explained Variance")
