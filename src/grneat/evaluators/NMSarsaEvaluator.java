package evaluators;

import java.util.Random;
import java.util.Vector;

import javax.print.DocFlavor.STRING;

import rlpark.plugin.rltoys.agents.functions.FunctionProjected2D;
import rlpark.plugin.rltoys.agents.functions.ValueFunction2D;
import rlpark.plugin.rltoys.agents.rl.LearnerAgentFA;
import rlpark.plugin.rltoys.algorithms.control.ControlLearner;
import rlpark.plugin.rltoys.algorithms.control.acting.EpsilonGreedy;
import rlpark.plugin.rltoys.algorithms.control.actorcritic.onpolicy.Actor;
import rlpark.plugin.rltoys.algorithms.control.actorcritic.onpolicy.AverageRewardActorCritic;
import rlpark.plugin.rltoys.algorithms.control.qlearning.QLearning;
import rlpark.plugin.rltoys.algorithms.control.qlearning.QLearningControl;
import rlpark.plugin.rltoys.algorithms.control.sarsa.GRNSarsa;
import rlpark.plugin.rltoys.algorithms.control.sarsa.Sarsa;
import rlpark.plugin.rltoys.algorithms.control.sarsa.SarsaControl;
import rlpark.plugin.rltoys.algorithms.functions.policydistributions.PolicyDistribution;
import rlpark.plugin.rltoys.algorithms.functions.policydistributions.helpers.ScaledPolicyDistribution;
import rlpark.plugin.rltoys.algorithms.functions.policydistributions.structures.NormalDistributionScaled;
import rlpark.plugin.rltoys.algorithms.functions.stateactions.TabularAction;
import rlpark.plugin.rltoys.algorithms.functions.states.Projector;
import rlpark.plugin.rltoys.algorithms.predictions.td.OnPolicyTD;
import rlpark.plugin.rltoys.algorithms.predictions.td.TDLambda;
import rlpark.plugin.rltoys.algorithms.representations.discretizer.partitions.AbstractPartitionFactory;
import rlpark.plugin.rltoys.algorithms.representations.tilescoding.TileCodersNoHashing;
import rlpark.plugin.rltoys.algorithms.traces.RTraces;
import rlpark.plugin.rltoys.envio.actions.Action;
import rlpark.plugin.rltoys.envio.policy.Policy;
import rlpark.plugin.rltoys.envio.rl.TRStep;
import rlpark.plugin.rltoys.experiments.runners.Runner;
import rlpark.plugin.rltoys.experiments.runners.AbstractRunner.RunnerEvent;
import rlpark.plugin.rltoys.math.ranges.Range;
import rlpark.plugin.rltoys.math.vector.BinaryVector;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.PVector;
import rlpark.plugin.rltoys.math.vector.implementations.Vectors;
import rlpark.plugin.rltoys.problems.helicopter.Helicopter;
import rlpark.plugin.rltoys.problems.mazes.Maze;
import rlpark.plugin.rltoys.problems.mazes.MazeValueFunction;
import rlpark.plugin.rltoys.problems.mazes.Mazes;
import rlpark.plugin.rltoys.problems.mountaincar.MountainCar;
import rlpark.plugin.rltoys.problems.pendulum.SwingPendulum;
import zephyr.plugin.core.api.Zephyr;
import zephyr.plugin.core.api.signals.Listener;
import zephyr.plugin.core.api.synchronization.Clock;
import evolver.GRNGenome;
import grn.GRNModel;

public class NMSarsaEvaluator extends GRNGenomeEvaluator {
	public double fitness;
	public int nEpisode;
	
	public NMSarsaEvaluator() {
		numGRNInputs=1;
		numGRNOutputs=3;
		name="NMSarsa";
	}

	@Override
	public double evaluate(GRNGenome aGenome) {
		numEvaluations++;
		Vector<String> problems = new Vector<String>();
		problems.add("Maze");
		double fitness = evaluateGRN(buildGRNFromGenome(aGenome), problems);
		aGenome.setNewFitness(fitness);
		return fitness;
	}

	public double evaluateGRN(GRNModel grn, Vector<String> problems) {
		double fitness = 0;
		if (problems.contains("MountainCar")) {
			fitness += evaluateMountainCar(grn);
		}
		if (problems.contains("Maze")) {
			fitness += evaluateMaze(grn);
		}
		//System.out.println(grn+" : "+fitness);
		return fitness;
	}

	public double evaluateMountainCar(GRNModel grn) {

		FunctionProjected2D valueFunctionDisplay;
		MountainCar problem;
		SarsaControl control;
		TileCodersNoHashing projector;
		Clock clock = new Clock("SarsaMountainCar");

		problem = new MountainCar(null);
		projector = new TileCodersNoHashing(problem.getObservationRanges());
		projector.addFullTilings(10, 10);
		projector.includeActiveFeature();

		TabularAction toStateAction = new TabularAction(problem.actions(), projector.vectorNorm(), projector.vectorSize());
		toStateAction.includeActiveFeature();
		double alpha = .15 / projector.vectorNorm();
		double gamma = 0.99;
		double lambda = .3;
		//System.out.print(numEvaluations+"\t: Evaluate: "+grn);
		Sarsa sarsa;
		if (grn != null) {
			sarsa = new GRNSarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces(), grn);
		} else {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces());
		}
		double epsilon = 0.01;
		Policy acting = new EpsilonGreedy(new Random(0), problem.actions(), toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		valueFunctionDisplay = new ValueFunction2D(projector, problem, sarsa);
		Zephyr.advertise(clock, this);

		TRStep step = problem.initialize();
		int nbEpisode = 0;
		RealVector x_t = null;
		fitness = 0;
		while (clock.tick() && nbEpisode<100) {
			BinaryVector x_tp1 = projector.project(step.o_tp1);
			Action action = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
			x_t = Vectors.bufferedCopy(x_tp1, x_t);
			if (step.isEpisodeEnding() || step.time>5000) {
				//System.out.println(String.format("%d\t%d", nbEpisode, step.time));
				fitness += step.time;
				step = problem.initialize();
				x_t = null;
				nbEpisode++;
			} else {
				step = problem.step(action);
			}
		}
		//System.out.println(" = "+fitness/5);
		return -fitness/100;		

	}
	
	public double evaluateMaze(GRNModel grn) {
		MazeValueFunction mazeValueFunction;
		Maze problem = Mazes.createBookMaze();
		ControlLearner control;
		Clock clock = new Clock("QLearningMaze");
		Projector projector;
		PVector occupancy;
		LearnerAgentFA agent;

		projector = problem.getMarkovProjector();
		occupancy = new PVector(projector.vectorSize());
		TabularAction toStateAction = new TabularAction(problem.actions(), projector.vectorNorm(), projector.vectorSize());
		double alpha = .15 / projector.vectorNorm();
		double gamma = 1.0;
		double lambda = 0.6;
		Sarsa sarsa;
		if (grn == null) {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces());
		} else {
			sarsa = new GRNSarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces(), grn);
		}
		double epsilon = 0.3;
		Policy acting = new EpsilonGreedy(new Random(0), problem.actions(), toStateAction, sarsa, epsilon);
		control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, projector);
		mazeValueFunction = new MazeValueFunction(problem, sarsa, toStateAction, acting);
		Zephyr.advertise(clock, this);
		
//		double fitness = 0;
		nEpisode = 0;
	    Runner runner = new Runner(problem, agent, 100, 10000);
	    fitness = 0;
	    runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
	      @Override
	      public void listen(RunnerEvent eventInfo) {
	        //System.out.println(String.format("%d\t%d", eventInfo.nbEpisodeDone, eventInfo.step.time));
	    	nEpisode++;
	        fitness+=eventInfo.step.time;
	      }
	    });
	    while (clock.tick() && nEpisode<100) {
	      runner.step();
	      occupancy.addToSelf(agent.lastState());
	    }

		//System.out.println(" = "+fitness/5);
		return -fitness/100;			
	}

	public double evaluateActorCriticPendulum(GRNModel grn) {
		FunctionProjected2D valueFunction;
		double reward;
		SwingPendulum problem;
		Clock clock = new Clock("ActorCriticPendulum");
		LearnerAgentFA agent;
		Runner runner;
	    
		Random random = new Random(0);
	    problem = new SwingPendulum(null, false);
	    TileCodersNoHashing tileCoders = new TileCodersNoHashing(problem.getObservationRanges());
	    ((AbstractPartitionFactory) tileCoders.discretizerFactory()).setRandom(random, .2);
	    tileCoders.addFullTilings(10, 10);
		double alpha = .15 / tileCoders.vectorNorm();
		double gamma = 1.0;
		double lambda = 0.6;
	    double vectorNorm = tileCoders.vectorNorm();
	    int vectorSize = tileCoders.vectorSize();
		TabularAction toStateAction = new TabularAction(problem.actions(), tileCoders.vectorNorm(), tileCoders.vectorSize());
		Sarsa sarsa;
		if (grn == null) {
			sarsa = new Sarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces());
		} else {
			sarsa = new GRNSarsa(alpha, gamma, lambda, toStateAction.vectorSize(), new RTraces(), grn);
		}
		double epsilon = 0.3;
		Policy acting = new EpsilonGreedy(new Random(0), problem.actions(), toStateAction, sarsa, epsilon);
		ControlLearner control = new SarsaControl(acting, toStateAction, sarsa);
		agent = new LearnerAgentFA(control, tileCoders);
	    valueFunction = new ValueFunction2D(tileCoders, problem, sarsa);
	    runner = new Runner(problem, agent, 100, 10000);
	    nEpisode=0;
	    fitness=0;
	    runner.onEpisodeEnd.connect(new Listener<RunnerEvent>() {
	      @Override
	      public void listen(RunnerEvent eventInfo) {
	        System.out.println(String.format("Episode %d: %f", eventInfo.nbEpisodeDone, eventInfo.episodeReward));
	        nEpisode++;
	        fitness+=eventInfo.episodeReward;
	      }
	    });
	    
	    Zephyr.advertise(clock, this);
	    
	    while (clock.tick() && nEpisode<100) {
	        runner.step();
	    }
	    
	    return fitness/100;

	}
	
	public static void main(String[] args) throws Exception {
		GRNModel grn = GRNModel.loadFromFile("NMSarsa/run_1420560446341160000/grn_44_-128.97.grn");

		double fSarsa = new NMSarsaEvaluator().evaluateMaze(null);
		double fGRN = new NMSarsaEvaluator().evaluateMaze(grn);
		System.out.println(fGRN+"\t"+fSarsa);
	}

}
