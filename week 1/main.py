import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import dataset
import model_creator


# check for CUDA availability
if torch.cuda.is_available():
    print('CUDA is available, setting device to CUDA')
# set device to  CUDA for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Variables
DATASETDIR = '../MIT_split/'
MODEL_FNAME = 'model.h5'
BATCH_SIZE = 32
EPOCHS = 100
INPUT_SIZE = 64


def lr_schedule(epoch):
    lrate = 0.02
    if epoch > 40:
        lrate = 0.01
    elif epoch > 80:
        lrate = 0.006
    elif epoch > 120:
        lrate = 0.0003        
    return lrate

# get dataloaders
train_loader, test_loader = dataset.get_dataloaders(DATASETDIR, INPUT_SIZE, BATCH_SIZE)

# Create the model
model = model_creator.get_model()
summary(model, (3, INPUT_SIZE, INPUT_SIZE), device='cpu')

#Send model to GPU
model.to(device)

#Define Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



#To do (PyTorch)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    min_delta=0.001, 
    mode='max'
)

learning_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(
    train_generator,
    shuffle=True,
    steps_per_epoch=train_generator.n // train_generator.batch_size, 
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size, 
    callbacks=[mcp_save, early_stopping, learning_scheduler]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('results/accuracy.jpg')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('results/loss.jpg')

# Load the best model
model.load_weights(filepath = '.mdl_wts.hdf5')
score =model.evaluate(validation_generator, verbose=2)

print('CV loss:', score[0])
print('CV accuracy:', score[1])
