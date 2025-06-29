import fairlib as fl

dataset = load_example_dataset()
fairlib_dataframe = fl.DataFrame(dataset)

# Setting the target feature and sensitive attributes
fairlib_dataframe.targets = 'income'
fairlib_dataframe.sensitive = ['sex', 'race']

reweighing = fl.Reweighing()
reweighed_df = reweighing.fit_transform(fairlib_dataframe)
# The reweighed DataFrame now contains adjusted instance weights to mitigate bias in classification tasks.