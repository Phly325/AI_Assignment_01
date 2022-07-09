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
# ╠═5e3ea410-cc7b-11ec-3f5e-c7ead3aecbc7

begin
rename!(res_df, Symbol.(replace.(string.(names(res_df)), Ref(r"\[m\]"=>"_"))))
rename!(res_df, Symbol.(replace.(string.(names(res_df)), Ref(r"\[s\]"=>"_"))))
rename!(res_df, Symbol.(replace.(string.(names(res_df)), Ref(r"\s"=>"_"))))
end
# ╠═f5dc4894-f985-46ef-af60-965b14508c05

 md"# Quick exploration of the dataset"
# ╠═1ed26fd3-2d3f-4bc7-a429-424efac8853e

md"# size and column names"
(414, 8)
# ╠═704878af-84c3-4c1c-8a9a-a8e5e80e0015

size(res_df)
with_terminal() do
    for col_name in names(res_df)
    println(col_name)
 end
end
# ╠═8440ce5f-6b37-4336-b65f-9b6d0be8608e

first(res_df, 15)
# ╠═d6f87bd6-8e84-4e29-9827-89a98e6effcc

    with_terminal() do
        describe(res_df[!, :X1_transaction_date])
        end
# ╠═e15d4018-2293-4850-bb9c-9f4416e78640

        with_terminal() do
            describe(res_df[!, :X2_house_age])
            end
# ╠═636b8d5c-5554-4b56-9114-35313d75d06c

            with_terminal() do
                describe(res_df[!, :X3_distance_to_the_nearest_MRT_station])
                end
                
  # ╠═55d75c51-e91c-425a-80c8-25742210802d
              
                with_terminal() do
                    describe(res_df[!, :X4_number_of_convenience_stores])
                    end
# ╠═9cb7a164-fa7a-4123-a7bd-b61d5a0dc8df

                    with_terminal() do
                        
                        
                        describe(res_df[!, :X5_latitude])
                    end

# ╠═efe1819a-4033-4e3f-be15-1c2a90fb3d14

    with_terminal() do
    describe(res_df[!, :X6_longitude])
    end
# ╠═4dc14dff-5dee-438f-92dd-9e852a839078

with_terminal() do
    describe(res_df[!, :Y_house_price_of_unit_area])
    end
# ╠═db018f79-e6da-4755-96e7-3c79df26a8f2

    cor(Matrix(res_df))

md"# Prepare the model"
md"### Extract training and testing datasets"
# ╠═4dc14dff-5dee-438f-92dd-9e852a839078

⋅ X = res_df[:, 3:7]
# ╠═db018f79-e6da-4755-96e7-3c79df26a8f2

⋅ X_mat = Matrix(X)
# ╠═35e4251b-306a-4282-99e4-7c84a4d009f1

⋅ Y = res_df[:,8]
# ╠═38869a65-0ff2-4485-8ec2-66e7041b51b9

⋅ Y_mat = Vector(Y)
# ╠═ce24dc65-fbb5-4925-a10a-760d3d956acc

⋅ training_size = 0.7
# ╠═adf4cc6a-40bd-4b2c-b872-6a9d0a9a2f7c

⋅ all_data_size = size(X_mat)[1]
# ╠═418f2790-10e0-499e-9a16-f5584bdf48a0

⋅ training_index = trunc(Int, training_size * all_data_size)
# ╠═9c9647a3-9c6c-4c50-8ef0-e310ea7b45cd

⋅ X_mat_train = X_mat[1:training_index, :]
# ╠═35e4251b-306a-4282-99e4-7c84a4d009f1

⋅ X_mat_test = X_mat[training_index+1:end, :]
# ╠═38869a65-0ff2-4485-8ec2-66e7041b51b9

⋅ Y_mat_train = Y_mat[1:training_index]
# ╠═ce24dc65-fbb5-4925-a10a-760d3d956acc

⋅ Y_mat_test = Y_mat[training_index+1:end]
# ╠═adf4cc6a-40bd-4b2c-b872-6a9d0a9a2f7c

function get_loss(feature_mat, outcome_vec, weights)
m = size(feature_mat)[1]
hypothesis = feature_mat * weights
loss = hypothesis - outcome_vec
cost = (1/(2m)) * (loss' * loss)
return cost
end
# ╠═418f2790-10e0-499e-9a16-f5584bdf48a0

function get_scaling_params(init_feature_mat)
feature_mean = mean(init_feature_mat, dims=1)
f_dev = std(init_feature_mat, dims=1)
_dev = std(init_feature_mat, dims=1)
return (feature_mean, f_dev)
end
# ╠═9c9647a3-9c6c-4c50-8ef0-e310ea7b45cd

function scale_features(feature_mat, sc_params)
normalised_feature_mat = (feature_mat .- sc_params[1]) ./ sc_params[2]
end
# ╠═d450e164-df8e-436f-8e8f-beaa9bea9f99

scaling_params = get_scaling_params(X_mat_train)
# ╠═01f5e0f5-9f34-4e44-b6d6-7ec587a10e66

⋅ scaling_params[1]
# ╠═440c87b4-e28c-44c0-a6b2-983471dcbae9

⋅ scaling_params[2]
# ╠═1aa296be-2a65-4562-9af1-dd72ac81bbc9

⋅ scaled_training_features = scale_features(X_mat_train, scaling_params)
# ╠═3eb1392f-e420-4efb-8d94-299e3829d261

⋅ scaled_testing_features = scale_features(X_mat_test, scaling_params)
# ╠═6927badb-e1a7-41a7-ba82-3057728b9da1

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
# ╠═e31fa121-c613-4b46-b00d-9350333dbefa

weights_tr_errors = train_model(scaled_training_features, Y_mat_train, 0.03,4000)
# ╠═66abe2de-516d-404d-b2f4-0ad4ae8fa43b

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
# ╠═09b1f042-d7e5-471a-8a4d-d24351c28aa1

function root_mean_square_error(actual_outcome, predicted_outcome)
errors = predicted_outcome - actual_outcome
squared_errors = errors .^ 2
mean_squared_errors = mean(squared_errors)
rmse = sqrt(mean_squared_error
return rmse
end

# ╠═09b1f042-d7e5-471a-8a4d-d24351c28aa1
Training_predictions = get_predictions(scaled_training_features,
weights_tr_errors[1])

# ╠═41311d41-8a57-43b3-9140-836f096eab9f
Testing_predictions = get_predictions(scaled_testing_features,weights_tr_errors[1])

# ╠═36258070-8803-4d3e-826a-c88141a0f063
root_mean_square_error(Y_mat_train, training_predictions)

root_mean_square_error(Y_mat_test, testing_predictions)


# ╔═╡ Cell order:
# ╠═5e3ea410-cc7b-11ec-3f5e-c7ead3aecbc7
# ╠═f5dc4894-f985-46ef-af60-965b14508c05
# ╠═1ed26fd3-2d3f-4bc7-a429-424efac8853e
# ╠═704878af-84c3-4c1c-8a9a-a8e5e80e0015
# ╠═8440ce5f-6b37-4336-b65f-9b6d0be8608e
# ╠═c524c57c-0eed-4b0c-b8a7-7a1530e31b3b
# ╠═d6f87bd6-8e84-4e29-9827-89a98e6effcc
# ╠═e15d4018-2293-4850-bb9c-9f4416e78640
# ╠═636b8d5c-5554-4b56-9114-35313d75d06c
# ╠═55d75c51-e91c-425a-80c8-25742210802d
# ╠═9cb7a164-fa7a-4123-a7bd-b61d5a0dc8df
# ╠═efe1819a-4033-4e3f-be15-1c2a90fb3d14
# ╠═4dc14dff-5dee-438f-92dd-9e852a839078
# ╠═db018f79-e6da-4755-96e7-3c79df26a8f2
# ╠═35e4251b-306a-4282-99e4-7c84a4d009f1
# ╠═38869a65-0ff2-4485-8ec2-66e7041b51b9
# ╠═ce24dc65-fbb5-4925-a10a-760d3d956acc
# ╠═adf4cc6a-40bd-4b2c-b872-6a9d0a9a2f7c
# ╠═418f2790-10e0-499e-9a16-f5584bdf48a0
# ╠═9c9647a3-9c6c-4c50-8ef0-e310ea7b45cd
# ╠═d450e164-df8e-436f-8e8f-beaa9bea9f99
# ╠═01f5e0f5-9f34-4e44-b6d6-7ec587a10e66
# ╠═440c87b4-e28c-44c0-a6b2-983471dcbae9
# ╠═1aa296be-2a65-4562-9af1-dd72ac81bbc9
# ╠═3eb1392f-e420-4efb-8d94-299e3829d261
# ╠═6927badb-e1a7-41a7-ba82-3057728b9da1
# ╠═e31fa121-c613-4b46-b00d-9350333dbefa
# ╠═66abe2de-516d-404d-b2f4-0ad4ae8fa43b
# ╠═09b1f042-d7e5-471a-8a4d-d24351c28aa1
# ╠═41311d41-8a57-43b3-9140-836f096eab9f
# ╠═36258070-8803-4d3e-826a-c88141a0f063
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002