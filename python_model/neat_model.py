from deap import creator, base, tools, algorithms
import neat, numpy as np, random as r
import array, multiprocessing

class CPPN_genome(neat.DefaultGenome):
    def __init__(self,key):
        super().__init__(key)
    def configure_new(self, genome_config):
        super().configure_new(genome_config)
    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
    def mutate(self,config):
        super().mutate(config)

class CPPN:
    def __init__(self, config, n_inputs, n_outputs):
        self.genome = CPPN_genome(1)
        self.config = neat.Config(neat.DefaultGenome,
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  config)

        self.config.genome_config.input_keys = [-(i + 1) for i in range(n_inputs)]
        self.config.genome_config.output_keys = [i for i in range(n_outputs)]
        self.config.genome_config.num_outputs = n_outputs
        self.config.genome_config.num_inputs = n_inputs

        self.genome.configure_new(self.config.genome_config)
        self.phenotype = self.getPhenotype()
            
        self.values = np.asarray([r.randint(3, 16), r.randint(20, 60),
                                  r.randint(8,  20), r.randint(8, 30),
                                  r.randint(20, 60), r.randint(2, 10)])

    def activate(self, values):
        self.values += np.int32(np.ceil(self.phenotype.activate(values)))

    def mutate(self):
        self.genome.mutate(self.config.genome_config)
        self.phenotype = self.getPhenotype()
        self.activate(self.values)

    def getPhenotype(self):
        return (neat.nn.FeedForwardNetwork.create(self.genome, self.config))

class Morphology:
    def __init__(self, config, pop_size):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", CPPN, typecode="d", fitness=creator.FitnessMax, strategy=None)
        creator.create("Strategy", array.array, typecode="d")
        self.pop_size = pop_size

        # Create core morphology.
        self.toolbox = base.Toolbox(); 
        self.toolbox.register("individual", creator.Individual, config=config, n_inputs=6, n_outputs=6)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select",tools.selTournament, tournsize = 4)
        self.toolbox.register("mutate", creator.Individual.mutate)

        # Create morphology population
        self.pop = self.toolbox.population(self.pop_size)
        self.hof = tools.HallOfFame(1)

    def add_controller(self, controller, config, logging):
        for i in range(self.pop_size):
            setattr(self.pop[i], "controller", controller(config, logging))
    
    def add_env(self, env):
        for i in range(self.pop_size):
            setattr(self.pop[i], "env", env())
        
    def run(self, n_generations, show_best, n_controller_gens, n_evals, render):
        self.toolbox.register("evaluate", eval_controller,
            show_best=show_best,
            n_controller_gens=n_controller_gens,
            n_evals=n_evals,
            render=render)

        # Run the core morphology algorithm
        for i in range(n_generations):
            print("Generation {} in progress.".format(i))
            offspring = self.toolbox.select(self.pop, len(self.pop))
            
            # deep copy of selected population
            offspring = list(map(self.toolbox.clone, offspring))
            for o in offspring:
                self.toolbox.mutate(o)
                o.fitness = 0

            fitnesses = self.toolbox.map(self.toolbox.evaluate, offspring)
            fitness_values = []
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness = fit
                fitness_values.append(fit)

            self.pop = offspring

def eval_controller(ind, show_best, n_controller_gens, n_evals, render):
    n_joints = ind.env.generate_structure(ind.values)
    winner = ind.controller.run_eval(ind.env, n_joints, show_best,
                                          n_controller_gens, n_evals, render)

    return winner,