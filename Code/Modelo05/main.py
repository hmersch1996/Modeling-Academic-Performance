"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function

from evolver import Evolver

from tqdm import tqdm

import logging
import numpy as np
import sys

import tensorflow as tf

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='log.txt'
)

#Esta funcion entrena a cada genome
def train_genomes(genomes, dataset):
    """Train each genome.

    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***train_networks(networks, dataset)***")
    #pbar: Progress Bar. uestra el progreso del entrenamiento del genoma
    pbar = tqdm(total=len(genomes))
    #
    for genome in genomes:
        genome.train(dataset)
        pbar.update(1)
    
    pbar.close()
#Esta funcion obtiene la exactitud promedio del la red/genoma
def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.

    Args:
        networks (list): List of networks/genomes

    Returns:
        float: The average accuracy of a population of networks/genomes.

    """
    total_accuracy = 0
    #Dentro de este for se suman las exactitudes de todos los genomas
    for genome in genomes:
        total_accuracy += genome.accuracy
    #Retorna la sumatoria de las exactitudes dividido la cantidad de genomas
    return total_accuracy / len(genomes)

#Esta funcion genera un network con el algoritmo genetico
def generate(generations, population, all_possible_genes, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")
    #Aqui empleamos la clase Evolver, es necesario inicializarla con el dato all_possible_genes
    evolver = Evolver(all_possible_genes)
    #Esta funcion de la clase evolver crea una poblacion, el dato que se le entrega es la cantidad de poblaciones a crear
    genomes = evolver.create_population(population)
    
    precision=[]
    # Evolve the generation.
    for i in range( generations ):
        #Aqui se indica en que generacion se encuentra
        logging.info("***Now in generation %d of %d***" % (i + 1, generations))
        #Imprime los genomas de esta generacion
        print_genomes(genomes)
        
        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(genomes)
        precision.append(average_accuracy)
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80) #-----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:5])
    
    
    return np.asarray(precision)
    
    #save_path = saver.save(sess, '/output/model.ckpt')
    #print("Model saved in file: %s" % save_path)



#Esta funcion imprime una lista con los genomas
def print_genomes(genomes):
    """Print a list of genomes.

    Args:
        genomes (list): The population of networks/genomes

    """
    logging.info('-'*80)

    for genome in genomes:
        genome.print_genome()

def main():
    """Evolve a genome."""
    population = 50 # Number of networks/genomes in each generation. Se recomienda usar 50/30
    #we only need to train the new ones....
    dataset = 'dataset'
    #dataset = 'Test.csv'
    #dataset=load( open('dataset.pkl', 'rb'))

    print("***Dataset:", dataset)

    
    generations = 50 # Number of times to evolve the population. #Se recomienda 100/50
    all_possible_genes = {
        'nb_neurons': [16, 32, 64, 128],
        'nb_layers':  [1, 2, 3, 4,5]
        
        #Para la activacion, solo trabajare con activadores que retornen valores entre 0 y 1
        #'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
        #'activation': ['sigmoid'],
        
        #Para esta aplicacion, voy a usar exclusivamente 'adam', si se busca trabajar con 
        #otras optimizaciones, se puede descomentar esta linea
        #'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        #'optimizer':  ['adam','adamax']
    }
  

    # replace nb_neurons with 1 unique value for each layer
    # 6th value reserved for dense layer
    nb_neurons = all_possible_genes['nb_neurons']
    for i in range(1,7):
      all_possible_genes['nb_neurons_' + str(i)] = nb_neurons
    # remove old value from dict
    all_possible_genes.pop('nb_neurons')
            
    print("***Evolving for %d generations with population size = %d***" % (generations, population))

    lista=generate(generations, population, all_possible_genes, dataset)
    
    return lista

if __name__ == '__main__':
    main()
