from utils.recognition_dataset import FSD50K_dataset, Singleclass_dataset

dataset = FSD50K_dataset('dataset/FSD50K', split='dev')
print(len(dataset))

dataset = Singleclass_dataset(dataset)
print(len(dataset))