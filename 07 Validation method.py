###Preparing for ANN input data
### one-hot encoding in Python
def vectorize_sequences(sequences, dimension = 8494):
    results = np.zeros(len(sequences), dimension)
    for i, sequence in enumerate(sequence):
        results[i,sequence]= 1.
    return results 

### Transform the x_train and y_train 
x_train = vectorize_sequences(x_train)
x_test = vectoraze_sequences(x_test)

###also trasform the output data dimension
from keras.utils import to_categorical
one_hot_train_labels = to_categorical(y_train,3)
one_hot_test_labels = to_categorical(y_test,3)

###keras is mainly used for ANN and CNN 
from keras import models
from keras import layers

###Add the parameter parameters 
###In this study, since neuron numbers are 13 in two layers,
###So we set them as 13 in the Dense(),which can add parameters.
model = models.Sequential()
model.add(layers.Dense(13, activation = 'relu', input_shape = (11,)))
model.add(layers.Dense(13, activation = 'relu', ))
model.add(layers.Dense(3, activation = 'softmax'))
###Since there are three categories of the output,the parameter is set wo 3
###softmax is the fuction that can have several outputs unlike sigmoid function.

###In the classification problem, the metrics is accuracy,
###The loss is "categorical_crossentropy"
###In the binary case, the loss then will be "binary_crossentropy"
model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics = ['accuracy'])


###general validation :
x_val = x_train[:425]
partial_x_train = x_train[425:]

y_val = one_hot_train_labels [:425]
partial_y_train = one_hot_train_labels[425:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 30,
                    batch_size = 100,
                    validation_data = (x_val, y_val))
###Making figures

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


###k-cross validation

###Add the parameter parameters 
###In this study, since neuron numbers are 13 in two layers,
###So we set them as 13 in the Dense(),which can add parameters.
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(13, activation = 'relu', input_shape = (11,)))
    model.add(layers.Dense(13, activation = 'relu', ))
    model.add(layers.Dense(3, activation = 'softmax'))
###Since there are three categories of the output,the parameter is set wo 3
###softmax is the fuction that can have several outputs unlike sigmoid function.

###In the classification problem, the metrics is accuracy,
###The loss is "categorical_crossentropy"
###In the binary case, the loss then will be "binary_crossentropy"
    model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
    return model

k = 20
num_val_samples = len(x_train)  //k
num_epochs = 100
acc_scores = []
loss_scores = []
all_acc_histories = []
aaall_loss_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = x_train[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = one_hot_train_labels[i * num_val_samples : (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [x_train[: i * num_val_samples],
         x_train[(i + 1) * num_val_samples:]],
        axis = 0)
    partial_train_targets = np.concatenate(
        [one_hot_train_labels[: i * num_val_samples],
         one_hot_train_labels[(i + 1) * num_val_samples:]],
        axis = 0)    
    model = build_model()
    
    history = model.fit(partial_train_data , partial_train_targets,
             epochs = num_epochs, batch_size = 1 ,verbose = 0)
    val_loss, val_acc = model.evaluate(val_data, val_targets, verbose = 0)
    acc_scores.append(val_acc)
    loss_scores.append(val_loss)
    acc_history = history.history['acc']
    loss_history = history.history['loss']
    all_acc_histories.append(acc_history)
#   all_loss_histories.append(loss_history)
   
average_acc_history = [
    np.mean([x[i] for x i all_acc_histories] for i in range(num_epochs))
#average_loss_history = [
    np.mean([y[i] for y in all_loss_histories] for i in range(num_epochs))
    
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_acc_history) + 1), average_acc_history)
#plt.plot(range(1, len(average_loss_history) + 1), averge_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')

