from src.predicter import Predicter
import pandas as pd
import os 
test_image_dir = os.listdir("data/data/")

# val_dataframe = pd.read_csv("only_val.csv")
# image_numbers = val_dataframe["ID"].to_list()
# image_numbers = [str(x) + ".png" for x in image_numbers]


image_numbers = test_image_dir[:30]
predictions = []
ground_truths = []


dataframe = pd.read_csv("./pseudolabel_done.csv")
# Create predictor object
# os.chdir('./src_Tran')
predictor = Predicter()

for i in range(len(image_numbers)) :
    # print(image_number)
    predictor.image_path = "data/data/" + image_numbers[i]
    # Predict the image in sample directory
    # os.chdir('..')
    # print(predictor.image_path )
    prediction = predictor.predict()
    predictions.append(prediction["label"])
    # print(prediction)

    


    gt_cp = dataframe[dataframe["ID"] == int(image_numbers[i].replace(".png", ""))]["label_cp"]
    gt_lr = dataframe[dataframe["ID"] == int(image_numbers[i].replace(".png", ""))]["label_lr"] 
    gt_odm = dataframe[dataframe["ID"] == int(image_numbers[i].replace(".png", ""))]["label_odm"]
    
    if gt_odm.to_list()[0] != "null-centered":
        gt = [gt_cp.to_list()[0], gt_lr.to_list()[0], gt_odm.to_list()[0]]
    else:
        gt = [gt_cp.to_list()[0], gt_lr.to_list()[0]]

    ground_truths.append(gt)


print(image_numbers[:10])
print(predictions[:10])
print(ground_truths[:10])

print(len(image_numbers))
print(len(predictions))
print(len(ground_truths))

# Prediction visualization

import matplotlib.pyplot as plt

fig = plt.figure("Prediction on test image",figsize=(20,20))
fig.tight_layout()
plt.axis('off')

for i in range(len(image_numbers[:30])): 
    ax = fig.add_subplot(6, 5,i+1) 
    ax.set_xticks([])
    ax.set_yticks([])
    sample_image = plt.imread("data/data/" + image_numbers[i])
    
    plt.title("Image ID: " + image_numbers[i] +"\nGround truth: " + ','.join(ground_truths[i])  + "\nPrediction: " + ','.join(predictions[i]))
    # ax.set_title()
    plt.imshow(sample_image)

# fig.save("test_pericentral.png")   
plt.show()
# set the spacing between subplots
plt.subplots_adjust(left=None,
                    bottom=None, 
                    right=None, 
                    top=None, 
                    wspace=0.5, 
                    hspace=0.5)
plt.savefig('test_predicter.png')