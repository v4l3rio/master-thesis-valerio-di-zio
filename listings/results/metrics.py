import fairlib as fl

dataset = load_example_dataset()
fairlib_dataframe = fl.DataFrame(dataset)

# Setting the target feature and sensitive attributes
fairlib_dataframe.targets = 'income'
fairlib_dataframe.sensitive = ['sex', 'race']

# Calculating fairness metrics
fairlib_dataframe.statistical_parity_difference()
fairlib_dataframe.disparate_impact()
fairlib_dataframe.equality_of_opportunity()