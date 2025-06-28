from datasets import load_dataset

# loading the dataset
training_split = load_dataset(
  'Maysee/tiny-imagenet', 
  split="train"
)

test_split = load_dataset(
  'Maysee/tiny-imagenet', 
  split="valid"
)

for index in range(len(training_split)):
  img = training_split[index]

  rbgImg = img['image'].convert('RGB')

  # resizedImg = rbgImg.resize((227, 227))
  # pixels = np.array(resizedImg).astype(np.uint8)
  # permuted = np.transpose(pixels, (2, 0, 1))
  # np.save(f"images/{img['label']}.{index}.npy", permuted)

  rbgImg.save(f"images/{img['label']}.{index}.png")
