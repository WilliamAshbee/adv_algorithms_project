import mutated_icospheres as mut

dataset = mut.MutatedIcospheresDataset(length=20)

mini_batch = 20

print(dataset[0][1].shape)
