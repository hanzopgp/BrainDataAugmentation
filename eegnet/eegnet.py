from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy,CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall, CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

LEARNING_RATE = 1e-3
MIN_DELTA = 1e-5
PATIENCE = 20

# Trick to use plot function from sklearn
class estimator:
    _estimator_type = ""
    classes_ = []
    def __init__(self, model, classes):
        self.model = model
        self._estimator_type = "classifier"
        self.classes_ = classes
    def predict(self, X):
        return self.model.predict(X).argmax(axis=1)

def plot_history_metrics(histories):
    total_plots = len(histories[0].history)
    cols = total_plots // 2
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1
    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for history in histories:
        for i, (key, value) in enumerate(history.history.items()):
            plt.subplot(rows, cols, pos[i])
            plt.plot(range(len(value)), value)
            plt.title(str(key))
    plt.show()

def get_callbacks():
    return [
        # ModelCheckpoint(
        #     "best_eegnet.h5", save_best_only=True, monitor="val_loss"
        # ),
        # ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=0.2,
        #     patience=2,
        #     min_lr=0.000001,
        # ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=MIN_DELTA,
            patience=PATIENCE,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        ),
    ]

# https://arxiv.org/pdf/1611.08024.pdf
def EEGNet(data_type, n_classes, n_channels=64, n_samples=128, kernel_length=64, n_filters1=8, 
           n_filters2=16, depth_multiplier=2, norm_rate=0.25, dropout_rate=0.5, 
           dropoutType="Dropout"):

    if dropoutType == "SpatialDropout2D":
        dropoutType=SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType=Dropout

    inputs = Input(shape=(n_channels, n_samples, 1))

    block1 = Conv2D(n_filters1, (1, kernel_length), padding="same", input_shape=(n_channels, n_samples, 1), use_bias=False)(inputs)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((n_channels, 1), use_bias=False, depth_multiplier=depth_multiplier, depthwise_constraint=max_norm(1.0))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropout_rate)(block1)

    block2 = SeparableConv2D(n_filters2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropout_rate)(block2)
    block2 = Flatten(name="flatten")(block2)

    classifier = Dense(n_classes, name="dense", kernel_constraint=max_norm(norm_rate))(block2)
    classifier = Activation("softmax", name="softmax")(classifier)

    model = Model(inputs=inputs, outputs=classifier) 

    optimizer = Adam(amsgrad=True, learning_rate=LEARNING_RATE)
    
    if data_type == "BCICIV":
        loss = CategoricalCrossentropy()
        acc = CategoricalAccuracy()
    elif data_type == "VEPESS":
        loss = BinaryCrossentropy()
        acc = BinaryAccuracy()
        
    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                acc,
    #             AUC(),
    #             Precision(),
    #             Recall(),
            ],
        )
    model.summary()
    return model

def train(model, train_dataset, valid_dataset, train_dataset_aug, valid_dataset_aug, y_train, y_train_aug, epochs=10, mode="CLASSIC"):
	classes = np.unique(y_train.numpy().argmax(axis=1))
	class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train.numpy().argmax(axis=1))
	class_weights = dict(zip(classes, class_weights))
	if model == "PRETRAIN":
		classes = np.unique(y_train_aug.numpy().argmax(axis=1))
		class_weights_aug = class_weight.compute_class_weight(class_weight="balanced", classes=classes, y=y_train_aug.numpy().argmax(axis=1))
		class_weights_aug = dict(zip(classes, class_weights_aug))
		print(class_weights_aug)
	if mode == "CLASSIC" or mode == "DOUBLE":
			history = model.fit(
					train_dataset,
					epochs=epochs,
					callbacks=get_callbacks(),
					validation_data=valid_dataset,
					class_weight=class_weights,
			)
			return history, _
	elif mode == "PRETRAIN":
			pretrain_history = model.fit(
					train_dataset_aug,
					epochs=epochs,
					callbacks=get_callbacks(),
					validation_data=valid_dataset_aug,
					class_weight=class_weights_aug,
			)
			final_history = model.fit(
					train_dataset,
					epochs=epochs,
					callbacks=get_callbacks(),
					validation_data=valid_dataset,
					class_weight=class_weights,
			)
			return pretrain_history, final_history

def train_n_models(n_models, n_classes, n_channels, n_samples, mode="CLASSIC"):
	histories = []
	pretrain_histories = []
	final_histories = []
	for _ in range(n_models):
			model = EEGNet(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples)
			if mode == "CLASSIC" or mode == "DOUBLE":
					history, _ = train(model)
					histories.append(history)
					return histories, _
			elif mode == "PRETRAIN":
					pretrain_history, final_history = train(model)
					pretrain_histories.append(pretrain_history)
					final_histories.append(final_history)
					return pretrain_histories, final_histories

def plot_confusion(model, class_names, X_train, y_train, X_valid, y_valid, X_test, y_test):
	classifier = estimator(model, class_names)
	color = "white"
	matrix = plot_confusion_matrix(classifier, X_train, y_train.numpy().argmax(axis=1), cmap="YlGnBu")
	matrix.ax_.set_title("Train confusion matrix", color=color)
	plt.xlabel("Predicted Label", color=color)
	plt.ylabel("True Label", color=color)
	plt.gcf().axes[0].tick_params(colors=color)
	plt.gcf().axes[1].tick_params(colors=color)
	plt.show()
	matrix = plot_confusion_matrix(classifier, X_valid, y_valid.numpy().argmax(axis=1), cmap="YlGnBu")
	matrix.ax_.set_title("Valid confusion matrix", color=color)
	plt.xlabel("Predicted Label", color=color)
	plt.ylabel("True Label", color=color)
	plt.gcf().axes[0].tick_params(colors=color)
	plt.gcf().axes[1].tick_params(colors=color)
	plt.show()
	matrix = plot_confusion_matrix(classifier, X_test, y_test.numpy().argmax(axis=1), cmap="YlGnBu")
	matrix.ax_.set_title("Test confusion matrix", color=color)
	plt.xlabel("Predicted Label", color=color)
	plt.ylabel("True Label", color=color)
	plt.gcf().axes[0].tick_params(colors=color)
	plt.gcf().axes[1].tick_params(colors=color)
	plt.show()

def classif_report(model, X_test, y_test):
	y_pred = model.predict(X_test).argmax(axis=1)
	y_true = y_test.numpy().argmax(axis=1)
	print(classification_report(y_true, y_pred))