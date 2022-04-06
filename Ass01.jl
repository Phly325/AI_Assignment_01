md"## Import libraries"
⋅ using Pkg
⋅ Pkg.activate("Project.toml")
⋅ using CSV
⋅ using DataFrames
⋅ using PlutoUI
⋅ using RDatasets
⋅ using StatsBase
⋅ using Statistics
⋅ using Plots
⋅ md"## Load the dataset"
res_df = CSV.read("data/real_estate.csv", DataFrame)

begin
rename!(res_df, Symbol.(replace.(string.(names(res_df)), Ref(r"\[m\]"=>"_"))))
rename!(res_df, Symbol.(replace.(string.(names(res_df)), Ref(r"\[s\]"=>"_"))))
rename!(res_df, Symbol.(replace.(string.(names(res_df)), Ref(r"\s"=>"_"))))
end

⋅ md"# Quick exploration of the dataset"

md"# size and column names"
(414, 8)

size(res_df)
with_terminal() do
    for col_name in names(res_df)
    println(col_name)
 end
end

first(res_df, 15)

    with_terminal() do
        describe(res_df[!, :X1_transaction_date])
        end

        with_terminal() do
            describe(res_df[!, :X2_house_age])
            end

            with_terminal() do
                describe(res_df[!, :X3_distance_to_the_nearest_MRT_station])
                end
                
                
                with_terminal() do
                    describe(res_df[!, :X4_number_of_convenience_stores])
                    end

                    with_terminal() do
                        
                        
                        describe(res_df[!, :X5_latitude])
                    end


    with_terminal() do
    describe(res_df[!, :X6_longitude])
    end

with_terminal() do
    describe(res_df[!, :Y_house_price_of_unit_area])
    end

    cor(Matrix(res_df))

md"# Prepare the model"
md"### Extract training and testing datasets"

⋅ X = res_df[:, 3:7]

⋅ X_mat = Matrix(X)

⋅ Y = res_df[:,8]

⋅ Y_mat = Vector(Y)

⋅ training_size = 0.7

⋅ all_data_size = size(X_mat)[1]

⋅ training_index = trunc(Int, training_size * all_data_size)

⋅ X_mat_train = X_mat[1:training_index, :]

⋅ X_mat_test = X_mat[training_index+1:end, :]

⋅ Y_mat_train = Y_mat[1:training_index]

⋅ Y_mat_test = Y_mat[training_index+1:end]

function get_loss(feature_mat, outcome_vec, weights)
m = size(feature_mat)[1]
hypothesis = feature_mat * weights
loss = hypothesis - outcome_vec
cost = (1/(2m)) * (loss' * loss)
return cost
end

function get_scaling_params(init_feature_mat)
feature_mean = mean(init_feature_mat, dims=1)
f_dev = std(init_feature_mat, dims=1)
_dev = std(init_feature_mat, dims=1)
return (feature_mean, f_dev)
end

function scale_features(feature_mat, sc_params)
normalised_feature_mat = (feature_mat .- sc_params[1]) ./ sc_params[2]
end

scaling_params = get_scaling_params(X_mat_train)

⋅ scaling_params[1]

⋅ scaling_params[2]

⋅ scaled_training_features = scale_features(X_mat_train, scaling_params)

⋅ scaled_testing_features = scale_features(X_mat_test, scaling_params)

function train_model(features, outcome, alpha, n_iter)
total_entry_count = length(outcome)
aug_features = hcat(ones(total_entry_count, 1), features)
feature_count = size(aug_features)[2]
weights = zeros(feature_count)
loss_vals = zeros(n_iter)
for i in range(1, stop=n_iter)
pred = aug_features * weights
loss_vals[i] = get_loss(aug_features, outcome, weights)
weights = weights - ((alpha/total_entry_count) * aug_features') * (pred -
outcome)
end
return (weights, loss_vals)
end

weights_tr_errors = train_model(scaled_training_features, Y_mat_train, 0.03,4000)

Plot(weights_tr_errors[2],
label="Cost",
ylabel="Cost",
xlabel="Number of Iteration",
title="Cost Per Iteration")

total_entry_count = size(features)[1]
aug_features = hcat(ones(total_entry_count, 1), features)
preds = aug_features * weights
return preds
end

function root_mean_square_error(actual_outcome, predicted_outcome)
errors = predicted_outcome - actual_outcome
squared_errors = errors .^ 2
mean_squared_errors = mean(squared_errors)
rmse = sqrt(mean_squared_error
return rmse
end

Training_predictions = get_predictions(scaled_training_features,
weights_tr_errors[1])

Testing_predictions = get_predictions(scaled_testing_features,weights_tr_errors[1])

root_mean_square_error(Y_mat_train, training_predictions)

root_mean_square_error(Y_mat_test, testing_predictions)