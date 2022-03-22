import os
import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score 
from matplotlib import colorbar
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class NeuralNet:
  
  activations = ['logistic', 'tanh', 'relu']
  learning_rates = [0.001, 0.01, 0.1]
  max_iterations = [50, 100, 150, 200, 250, 300]
  num_hidden_layers = [[7], [2, 3], [7, 7, 7]]

  def __init__(self, url):
    self.data_url = url
    self.fetch_data()

  def fetch_data(self):
    self.data = pd.read_excel(self.data_url)

  def get_class_code(self, x):
    classes = {
        'SEKER' : 0,
        'BARBUNYA' : 1,
        'BOMBAY' : 2,
        'CALI' : 3,
        'DERMASON' : 4,
        'HOROZ' : 5,
        'SIRA' : 6
    }
    return classes[x]

  def preprocess_data(self):
    # encoding class
    self.data['Class'] = self.data['Class'].map(self.get_class_code)

    # seperating features and classes
    self.X = self.data[['AspectRation', 'Eccentricity','Compactness','ShapeFactor1','ShapeFactor3']]
    self.Y = self.data['Class']

    self.scale_data()
    self.split_data()

  def scale_data(self):
    scaler = StandardScaler()
    self.X = scaler.fit_transform(self.X)
    self.X = pd.DataFrame(self.X)

  def split_data(self):
    X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size = 0.2, random_state = 5)

    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test

    print('Train data : \n\t X - {} \n\t Y - {}'.format(X_train.shape, Y_train.shape))
    print('Test data : \n\t X - {} \n\t Y - {}'.format(X_test.shape, Y_test.shape))

  
  def train(self):

    model_history = {
      'activation' : [],
      'alpha' : [],
      'iterations' : [],
      'num_hidden_layers' : [],
      'score' : [],
      'rmse' : [],
      'r2' : [],
      'loss' : []
    }

    progress = tqdm(total=len(self.activations)*len(self.learning_rates)*len(self.max_iterations)*len(self.num_hidden_layers), desc='Training model....', position=0, leave=True)

    for activation in self.activations:
      for alpha in self.learning_rates:
        for hidden_layer in self.num_hidden_layers:
          for iterations in self.max_iterations:

            model = MLPClassifier(hidden_layer_sizes=(7,7,7), activation=activation, alpha=alpha, max_iter=iterations)
            model.fit(self.X_train, self.Y_train)
            predictions = model.predict(self.X_test)

            model_history['activation'].append(activation)
            model_history['alpha'].append(alpha)
            model_history['iterations'].append(model.n_iter_)
            model_history['num_hidden_layers'].append(len(hidden_layer))
            model_history['score'].append(model.score(self.X_test, self.Y_test))
            model_history['r2'].append(r2_score(self.Y_test, predictions))
            model_history['rmse'].append(mean_squared_error(self.Y_test, predictions))
            model_history['loss'].append(model.loss_curve_)

            progress.update(1)

    self.model_history = pd.DataFrame.from_dict(model_history)
    self.write_report_to_file()

  def write_report_to_file(self):
    self.model_history.to_csv('parameters_list.csv', columns=['activation', 'alpha', 'iterations', 'num_hidden_layers', 'score', 'r2', 'rmse'], index=False)

  def plot_results(self):
    # Ploting Accurancy vs Iterations
    fig, axs = plt.subplots(3, 3, figsize=(55,45))
    for i, alpha in enumerate(self.learning_rates):
      for j, layers in enumerate(self.num_hidden_layers):
        for activation in self.activations:
          y_axis = self.model_history[(self.model_history.activation == activation) & (self.model_history.alpha == alpha) & (self.model_history.num_hidden_layers == len(layers))]
          axs[i,j].plot(self.max_iterations, y_axis['score'], label='With Activation function {}'.format(activation))
        axs[i,j].set_title('Learning rate {}, No of layers {}'.format(alpha, len(layers)))
        axs[i,j].legend(loc='best')
        axs[i,j].set_xlabel('Iterations')
        axs[i,j].set_ylabel('Accuracy')

    
    # Ploting Loss vs Iterations
    fig, axs = plt.subplots(3, 3, figsize=(55,45))
    for i, alpha in enumerate(self.learning_rates):
      for j, layers in enumerate(self.num_hidden_layers):
        for activation in self.activations:
          y_axis = self.model_history[(self.model_history.activation == activation) & (self.model_history.alpha == alpha) & (self.model_history.num_hidden_layers == len(layers))]
          y_axis = y_axis[y_axis.score == y_axis.score.max()]
          axs[i,j].plot(list(y_axis['loss'])[0], label='With Activation function {}'.format(activation))
        axs[i,j].set_title('Learning rate {}, No of layers {}'.format(alpha, len(layers)))
        axs[i,j].legend(loc='best')
        axs[i,j].set_xlabel('Iterations')
        axs[i,j].set_ylabel('Loss')


    # Ploting R2 vs Iterations
    fig, axs = plt.subplots(3, 3, figsize=(55,45))
    for i, alpha in enumerate(self.learning_rates):
      for j, layers in enumerate(self.num_hidden_layers):
        for activation in self.activations:
          y_axis = self.model_history[(self.model_history.activation == activation) & (self.model_history.alpha == alpha) & (self.model_history.num_hidden_layers == len(layers))]
          axs[i,j].plot(self.max_iterations, y_axis['r2'], label='With Activation function {}'.format(activation))
        axs[i,j].set_title('Learning rate {}, No of layers {}'.format(alpha, len(layers)))
        axs[i,j].legend(loc='best')
        axs[i,j].set_xlabel('Iterations')
        axs[i,j].set_ylabel('R2')

    # Ploting RMSE vs Iterations
    fig, axs = plt.subplots(3, 3, figsize=(55,45))
    for i, alpha in enumerate(self.learning_rates):
      for j, layers in enumerate(self.num_hidden_layers):
        for activation in self.activations:
          y_axis = self.model_history[(self.model_history.activation == activation) & (self.model_history.alpha == alpha) & (self.model_history.num_hidden_layers == len(layers))]
          axs[i,j].plot(self.max_iterations, y_axis['rmse'], label='With Activation function {}'.format(activation))
        axs[i,j].set_title('Learning rate {}, No of layers {}'.format(alpha, len(layers)))
        axs[i,j].legend(loc='best')
        axs[i,j].set_xlabel('Iterations')
        axs[i,j].set_ylabel('rmse')

    plt.show()


if __name__ == '__main__':
  model = NeuralNet('https://github.com/adi199/datasets/raw/main/Dry_Bean_Dataset.xlsx')
  model.preprocess_data()
  model.train()
  model.plot_results()